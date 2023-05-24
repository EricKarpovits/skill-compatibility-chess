import multiprocessing
import collections
import collections.abc
import time
import zipfile
import os.path

import numpy as np

from .proto_reader import ProtoGamesReader, sampling_types
from ..leela_board import max_clock

from ..utils import printWithDate

#max_seconds = 20
move_time_splits = [1,2,4,8]

class DataFileReaderBase(collections.abc.Iterator):
    def __init__(self, readers, writers, processes, rng, verbose = False, is_multi_proc = True, sampling_type = 'uniform', sampling_options = None, board_stack_size = 8):
        self.readers = readers
        self.writers = writers
        self.processes = processes
        self.verbose = verbose
        self.sampling_type = sampling_type
        self.sampling_options = sampling_options
        self.board_stack_size = board_stack_size
        self.sample_ids = set()
        self.sample_indices = []
        self.num_samples = 0
        self.is_multi_proc = is_multi_proc
        self.current_reader_id = 0
        self.num_readers = len(self.readers)

        if self.sampling_type not in sampling_types:
            raise NotImplementedError(f"{self.sampling_type} is not a known sampling function, {sampling_types.keys().join(', ')} are the valid options")
        if self.sampling_options is None:
            self.sampling_options = sampling_types[self.sampling_type].copy()

        if rng is None:
            self.rng = np.random.default_rng(123)
        else:
            self.rng = rng

    #@profile
    def __next__(self):
        cur_pipe = self.readers[self.current_reader_id]
        if cur_pipe.poll(10):
            resp = cur_pipe.recv()
        else:
            for r in self.processes:
                if not r.is_alive():
                    d = r.join()
                    raise multiprocessing.ProcessError(f"Something happened in a reader: {d}")
            return next(self)

        self.current_reader_id = (self.current_reader_id + 1) % self.num_readers
        sample_boards, sample_avail, sample_move, sample_result, sample_move_time, sample_ids, sample_indices = resp
        self.sample_ids |= set(sample_ids)
        self.num_samples += len(sample_ids)
        self.sample_indices += sample_indices
        return sample_boards, sample_move, sample_result

    #@profile
    def sampling_iter(self):
        while True:
            yield self.chunk_converter(*next(self))

    @staticmethod
    #@profile
    def chunk_converter(boards, moves, winner):
        import tensorflow as tf
        q_np = np.zeros_like(winner)
        q_np[:,0] = .5
        q_np[:,2] = .5
        b = tf.convert_to_tensor(boards, dtype=tf.float32)
        #a = tf.convert_to_tensor(avail_time, dtype=tf.float32) / max_clock
        m = tf.convert_to_tensor(moves, dtype=tf.float32)
        w = tf.convert_to_tensor(winner, dtype=tf.float32)
        q = tf.convert_to_tensor(q_np, dtype=tf.float32)
        return b, m, w, q

    def check_workers(self, timeout = 60):
        if not self.is_multi_proc:
            return
        for i, r in enumerate(self.readers):
            if r.poll(timeout = timeout):
                self.info_print(f"{i} {os.path.basename(self.file_path)} worker ready", flush = True, end = '\r')
            else:
                for p in self.processes:
                    if not p.is_alive():
                        d = p.join()
                        raise multiprocessing.ProcessError(f"Something happened in a reader: {p} {d}")
                raise multiprocessing.TimeoutError(f"Reader {r} timed out")
        self.info_print(f"All {len(self.processes)} {os.path.basename(self.file_path)} workers fully loaded")

    def shutdown(self):
        """
        Terminates all the workers
        """
        if not self.is_multi_proc:
            return
        self.info_print(f"Terminating {os.path.basename(self.file_path)} workers")
        for i, p in enumerate(self.processes):
            p.terminate()
            p.join()
            self.info_print(f"task {i} ended", end = '\r')
        for r in self.readers:
            r.close()
        for w in self.writers:
            w.close()
        self.info_print(f"All {len(self.processes)} {os.path.basename(self.file_path)} workers terminated")

    def info_print(self, *args, colour = None, **kwargs):
        if self.verbose:
            printWithDate(*args, colour = colour, **kwargs)

    def __del__(self):
        try:
            self.shutdown()
        except:
            pass

class ZipDataFileReaderSingle(DataFileReaderBase):
    def __init__(self, file_path, batch_size, shuffle_size = 256, verbose = True, rng = None, num_workers = None, sampling_type = 'uniform', sampling_options = None, board_stack_size = 8):
        super().__init__([], [], [], rng, verbose = verbose, is_multi_proc = False, sampling_type = sampling_type, sampling_options = sampling_options, board_stack_size = board_stack_size)
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.zip_handle = zipfile.ZipFile(file_path, 'r')
        self.files_lst = [f.filename for f in self.zip_handle.filelist]
        self.main_iter = self.task_loop()

    def task_loop(self):
        next_files = []
        shuffle_buffer = {}
        while True:
            if len(next_files) < 1:
                next_files = self.files_lst.copy()
                self.rng.shuffle(next_files)
            target_file = next_files.pop()
            try:
                with self.zip_handle.open(target_file) as f:
                    games_file = ProtoGamesReader(f, rng = self.rng, sampling_type = self.sampling_type, sampling_options = self.sampling_options)
                    for (g_id, g_index), (board, ig, avail_time, move, result, move_time) in games_file.iter_sample_games(board_stack_size = self.board_stack_size):
                        if(ig):
                            continue
                        shuffle_index = self.rng.integers(0, self.shuffle_size)
                        try:
                            yield shuffle_buffer.pop(shuffle_index)
                        except KeyError:
                            pass
                        shuffle_buffer[shuffle_index] = (board, avail_time, move, result, move_time, g_id, g_index, )
            except zipfile.BadZipFile as e:
                print(target_file, e)
                continue

    def __next__(self):
        batch = []
        for i in range(self.batch_size):
            batch.append(next(self.main_iter))
        sample_boards =  np.stack([s[0] for s in batch]).astype(np.float32)
        sample_avail_time = np.stack([s[1] for s in batch]).astype(np.float32).reshape(-1, 1)
        sample_move =  np.stack([make_one_hot(s[2], 1858) for s in batch]).astype(np.float32)
        sample_result =  np.stack([np.array([1,0,0]) if s[3] else np.array([0,0,1]) for s in batch]).astype(np.float32)
        sample_move_time =  np.stack([make_mvt_hot(s[4], move_time_splits) for s in batch]).astype(np.float32)
        sample_ids = [s[5] for s in batch]
        sample_indices = [s[6] for s in batch]
        return sample_boards, sample_avail_time, sample_move, sample_result, sample_move_time, sample_ids, sample_indices

    def __del__(self):
        try:
            self.zip_handle.close()
            super().__del__()
        except:
            pass


class ZipDataFileEmbedReaderSingle(DataFileReaderBase):
    def __init__(self, player_maps, batch_size, shuffle_size = 256, verbose = True, rng = None, num_workers = None, sampling_type = 'uniform', sampling_options = None, board_stack_size = 8):
        super().__init__([], [], [], rng, verbose = verbose, is_multi_proc = False, sampling_type = sampling_type, sampling_options = sampling_options, board_stack_size = board_stack_size)
        self.file_path = '.'
        self.player_maps = player_maps
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.zip_handles = {}
        self.zip_file_lsts = {}
        for p_name, (p_path, p_vec) in player_maps.items():
            self.zip_handles[p_name] = zipfile.ZipFile(p_path, 'r')
            self.zip_file_lsts[p_name] = [f.filename for f in self.zip_handles[p_name].filelist]
        self.main_iter = self.task_loop()

    def task_loop(self):
        next_players = []
        shuffle_buffer = {}
        while True:
            if len(next_players) < 1:
                next_players = list(self.player_maps.keys())
                self.rng.shuffle(next_players)
            target_player = next_players.pop()
            target_file = self.rng.choice(self.zip_file_lsts[target_player])
            try:
                with self.zip_handles[target_player].open(target_file) as f:
                    games_file = ProtoGamesReader(f, player_name = target_player, rng = self.rng, sampling_type = self.sampling_type, sampling_options = self.sampling_options)
                    for (g_id, g_index), (board, ig, avail_time, move, result, move_time) in games_file.iter_sample_games(board_stack_size = self.board_stack_size):
                        if(ig):
                            continue
                        shuffle_index = self.rng.integers(0, self.shuffle_size)
                        try:
                            yield shuffle_buffer.pop(shuffle_index)
                        except KeyError:
                            pass
                        shuffle_buffer[shuffle_index] = (board, self.player_maps[target_player][1], move, result, move_time, g_id, g_index, )
            except zipfile.BadZipFile as e:
                print(target_file, e)
                continue

    def __next__(self):
        batch = []
        for i in range(self.batch_size):
            batch.append(next(self.main_iter))
        sample_boards =  np.stack([s[0] for s in batch]).astype(np.float32)
        sample_p_vec = np.stack([s[1] for s in batch]).astype(np.float32).reshape(-1, 512)
        sample_move =  np.stack([make_one_hot(s[2], 1858) for s in batch]).astype(np.float32)
        sample_result =  np.stack([np.array([1,0,0]) if s[3] else np.array([0,0,1]) for s in batch]).astype(np.float32)
        sample_move_time =  np.stack([make_mvt_hot(s[4], move_time_splits) for s in batch]).astype(np.float32)
        sample_ids = [s[5] for s in batch]
        sample_indices = [s[6] for s in batch]
        return sample_boards, sample_p_vec, sample_move, sample_result, sample_move_time, sample_ids, sample_indices

    def __del__(self):
        try:
            for zip_handle in self.zip_handles.values():
                zip_handle.close()
            super().__del__()
        except:
            pass

class ZipDataFileReaderMulti(DataFileReaderBase):
    def __init__(self, file_path, batch_size, shuffle_size = 256, verbose = True, rng = None, num_workers = 4, sampling_type = 'uniform', sampling_options = None, board_stack_size = 8):
        if rng is None:
            rng = np.random.default_rng(123)

        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.num_workers = num_workers
        self.verbose = verbose
        readers = []
        writers = []
        processes = []
        for i in range(num_workers):
            read, write = multiprocessing.Pipe(duplex=False)
            p = multiprocessing.Process(target=self.task, args=(file_path, batch_size, shuffle_size, verbose, rng.integers(0, 99999), write, sampling_type, sampling_options, board_stack_size))
            processes.append(p)
            p.start()
            readers.append(read)
            writers.append(write)
            self.info_print(f"{i} {len(processes)} {os.path.basename(self.file_path)} tasks started", end = '\r')
        self.info_print(f"All {len(processes)} {os.path.basename(self.file_path)} tasks started")

        super().__init__(readers, writers, processes, rng, verbose = verbose, is_multi_proc = True, sampling_type = sampling_type, sampling_options = sampling_options, board_stack_size = board_stack_size)

    @staticmethod
    def task(file_path, batch_size, shuffle_size, verbose, rng_seed, writer, sampling_type, sampling_options, board_stack_size):
        rng = np.random.default_rng(rng_seed)
        reader = ZipDataFileReaderSingle(file_path, batch_size, shuffle_size = shuffle_size, verbose = verbose, rng = rng, sampling_type = sampling_type, sampling_options = sampling_options, board_stack_size = board_stack_size)
        try:
            while True:
                writer.send(next(reader))
        except KeyboardInterrupt:
            return

class ZipDataFileEmbedReaderMulti(DataFileReaderBase):
    def __init__(self, player_maps, batch_size, shuffle_size = 256, verbose = True, rng = None, num_workers = 4, sampling_type = 'uniform', sampling_options = None, board_stack_size = 8):
        if rng is None:
            rng = np.random.default_rng(123)
        self.file_path = '.'
        self.player_maps = player_maps
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.num_workers = num_workers
        self.verbose = verbose
        readers = []
        writers = []
        processes = []
        for i in range(num_workers):
            read, write = multiprocessing.Pipe(duplex=False)
            p = multiprocessing.Process(target=self.task, args=(player_maps, batch_size, shuffle_size, verbose, rng.integers(0, 99999), write, sampling_type, sampling_options, board_stack_size))
            processes.append(p)
            p.start()
            readers.append(read)
            writers.append(write)
            self.info_print(f"{i} {len(processes)} {os.path.basename(self.file_path)} tasks started", end = '\r')
        self.info_print(f"All {len(processes)} {os.path.basename(self.file_path)} tasks started")

        super().__init__(readers, writers, processes, rng, verbose = verbose, is_multi_proc = True, sampling_type = sampling_type, sampling_options = sampling_options, board_stack_size = board_stack_size)

    @staticmethod
    def task(player_maps, batch_size, shuffle_size, verbose, rng_seed, writer, sampling_type, sampling_options, board_stack_size):
        rng = np.random.default_rng(rng_seed)
        reader = ZipDataFileEmbedReaderSingle(player_maps, batch_size, shuffle_size = shuffle_size, verbose = verbose, rng = rng, sampling_type = sampling_type, sampling_options = sampling_options, board_stack_size = board_stack_size)
        try:
            while True:
                writer.send(next(reader))
        except KeyboardInterrupt:
            return

def make_mvt_hot(index, mv_splits):
    #quants: 2, 3, 7
    a = np.zeros(len(mv_splits) + 1)
    for i, mvt in enumerate(mv_splits):
        if index <= mvt:
            a[i] = 1
            return a
    a[len(mv_splits)] = 1
    return a

def make_one_hot(index, length):
    a = np.zeros(length)
    try:
        a[index] = 1
    #implicit clamp
    except IndexError:
        a[length - 1] = 1
    return a
