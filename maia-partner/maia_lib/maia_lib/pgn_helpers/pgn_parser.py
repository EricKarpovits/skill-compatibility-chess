import bz2
import collections.abc
import datetime
import io
import multiprocessing
import multiprocessing.synchronize
import os.path
import queue
import re
import threading
import typing

import chess
import chess.pgn

from ..leela_board import LeelaBoard
from ..utils import init_non_capture_worker
from .pgn_parsing_funcs import (
    get_termination_condition,
    per_game_infos,
    per_move_eval_funcs,
    per_move_funcs,
    per_move_time_funcs,
)

_header = r"""\[([A-Za-z0-9_]+)\s+"((?:[^"]|\\")*)"\]"""
header_re = re.compile(_header)
_headers = r"(" + _header + r"\s*\n)+\n"
_moves = r"[^\[]*(\*|1-0|0-1|1/2-1/2)\s*\n"
_move = r"""([NBKRQ]?[a-h]?[1-8]?[\-x]?[a-h][1-8](?:=?[nbrqkNBRQK])?|[PNBRQK]?@[a-h][1-8]|--|Z0|O-O(?:-O)?|0-0(?:-0)?)|(\{.*)|(;.*)|(\$[0-9]+)|(\()|(\))|(\*|1-0|0-1|1\/2-1\/2)|([\?!]{1,2})"""

game_clock_re = re.compile(r"\[%clk ")
game_eval_re = re.compile(r"%eval ")
move_num_re = re.compile(r"\d{1,3}\.{1,3}( |\n)")

game_counter_re = re.compile(r'\[Event "')

game_re = re.compile(
    r"(" + _headers + r")(.*?)(\*|1-0|0-1|1\/2-1\/2)", re.MULTILINE | re.DOTALL
)


class GamesFile(collections.abc.Iterator):
    def __init__(
        self,
        path: typing.Union[str, typing.TextIO],
        multi_proc_read: bool = False,
        multi_proc_workers: int = 4,
    ) -> None:
        if isinstance(path, io.StringIO) or isinstance(path, io.TextIOWrapper):
            self.file = path
        elif not isinstance(path, str):
            raise TypeError("path must be a string or a file object")
        elif path.endswith("bz2"):
            self.file = bz2.open(path, "rt")
        elif os.path.isfile(path):
            if path.endswith(".pgn"):
                self.file = open(path, "rt")
            else:
                raise TypeError("path must be a pgn or bz2 file")
        else:
            raise FileNotFoundError(f"Games file not found: {path}")
        self.path = path
        self.multi_proc_read = multi_proc_read
        self.games_queue = multiprocessing.Queue()
        self.processed_queue = multiprocessing.Queue()
        self.end_of_file = multiprocessing.Event()
        self.process_pool = None
        if self.multi_proc_read:
            self.process_workers = [
                multiprocessing.Process(
                    target=game_process_worker,
                    args=(i, self.games_queue, self.processed_queue, self.end_of_file),
                    daemon=True,
                )
                for i in range(multi_proc_workers)
            ]
            for p in self.process_workers:
                p.start()
            self.re_iter = multi_proc_stream_iter(
                self.file,
                self.games_queue,
                self.processed_queue,
                self.end_of_file,
                len(self.process_workers),
            )
        else:
            self.re_iter = stream_iter(self.file)

    def __del__(self) -> None:
        try:
            self.file.close()
            self.end_of_file.set()
            if self.multi_proc_read:
                for w in self.process_workers:
                    w.terminate()
        except:
            pass

    def __next__(self) -> typing.Tuple[typing.Dict[str, str], str]:
        r = next(self.re_iter)
        return extract_header(r.group(1)), r.group(0)

    def __iter__(
        self,
    ) -> typing.Generator[typing.Tuple[typing.Dict[str, str], str], None, None]:
        while True:
            try:
                yield next(self)
            except StopIteration:
                break

    def iter_moves(
        self,
    ) -> typing.Generator[typing.Tuple[LeelaBoard, typing.Any], None, None]:
        return games_sample_iter(self)

    def iter_mongo_dicts(
        self,
    ) -> typing.Generator[typing.Dict[str, typing.Any], None, None]:
        for game_dict, game_str in self:
            yield make_game_info_mongo(game_dict, game_str)

    def next_raw_game(self) -> str:
        return next(self.re_iter).group(0)

    def iter_raw_games(self) -> typing.Generator[str, None, None]:
        while True:
            try:
                yield next(self.re_iter).group(0)
            except StopIteration:
                break


def extract_header(header_str: str) -> typing.Dict[str, str]:
    try:
        header = header_re.findall(header_str)
    except AttributeError:
        raise AttributeError(f"Failed to parse input as pgn header: '{header_str}'")
    return {k: v for k, v in header}


def stream_iter(file_handle: typing.TextIO) -> typing.Iterator[re.Match]:
    current_game = file_handle.readline()
    for line in file_handle:
        if line.startswith("[Event "):
            gr = game_re.match(current_game.strip())
            if gr is not None:
                yield gr
            current_game = ""
        current_game += line
    if len(current_game.strip()) > 0:
        game_match = game_re.match(current_game.strip())
        if game_match is not None:
            yield game_match


def multi_proc_stream_iter(
    file_handle: typing.TextIO,
    games_queue: multiprocessing.Queue,
    processed_queue: multiprocessing.Queue,
    end_of_file: multiprocessing.synchronize.Event,
    active_worker_count: int,
) -> typing.Iterator[re.Match]:
    current_game = file_handle.readline()
    for line in file_handle:
        if line.startswith("[Event "):
            cur_game = current_game.strip()
            while games_queue.full():
                try:
                    game_dat = processed_queue.get(timeout=0.1)
                    if game_dat is None:
                        active_worker_count -= 1
                        if active_worker_count == 0:
                            end_of_file.set()
                            break
                    else:
                        yield game_dat
                except queue.Empty:
                    pass
            # Nothing else should be pushing to this
            games_queue.put_nowait(cur_game)
            current_game = ""
            while not processed_queue.empty():
                try:
                    game_dat = processed_queue.get(timeout=0.1)
                    if game_dat is None:
                        active_worker_count -= 1
                        if active_worker_count == 0:
                            end_of_file.set()
                            break
                    else:
                        yield game_dat
                except queue.Empty:
                    break
        current_game += line
    if len(current_game.strip()) > 0:
        while games_queue.full():
            try:
                game_dat = processed_queue.get(timeout=0.1)
                if game_dat is None:
                    active_worker_count -= 1
                    if active_worker_count == 0:
                        end_of_file.set()
                        break
                else:
                    yield game_dat
            except queue.Empty:
                pass
        games_queue.put_nowait(current_game.strip())
    end_of_file.set()
    while not processed_queue.empty():
        try:
            game_dat = processed_queue.get(timeout=0.1)
            if game_dat is None:
                active_worker_count -= 1
                if active_worker_count == 0:
                    break
            else:
                yield game_dat
        except queue.Empty:
            break


def game_process_worker(
    worker_id: int,
    games_queue: multiprocessing.Queue,
    processed_queue: multiprocessing.Queue,
    end_of_file: multiprocessing.synchronize.Event,
) -> None:
    init_non_capture_worker(f"[maia_lib_internal] game processing worker {worker_id}")
    while not end_of_file.is_set() and not games_queue.empty():
        try:
            game = games_queue.get(timeout=1)
            gr = game_re.match(game.strip())
            if gr is not None:
                processed_queue.put(gr)
        except queue.Empty:
            pass
    processed_queue.put(None)


def read_games_threaded(
    file_handle: typing.TextIO,
    file_lock: threading.Lock,
    games_queue: queue.Queue,
    stopped: threading.Event,
) -> None:
    file_lock.acquire()
    current_game = file_handle.readline()
    for line in file_handle:
        if line.startswith("[Event "):
            if games_queue.full():
                file_lock.release()
                games_queue.put(game_re.match(current_game.strip()))
                file_lock.acquire()
            games_queue.put(game_re.match(current_game.strip()))
            current_game = ""
        current_game += line
        if stopped.is_set():
            break
    file_lock.release()
    if len(current_game.strip()) > 0:
        games_queue.put(game_re.match(current_game.strip()))
    stopped.set()
    games_queue.put(None)


def stream_iter_threaded(
    games: queue.Queue, stop: threading.Event
) -> typing.Iterator[re.Match]:
    while not stop.is_set():
        game_match = games.get()
        if game_match is not None:
            yield game_match
        # if len(current_game.strip()) > 0:
        #    game_match = game_re.match(current_game.strip())
        #    if game_match is not None:
        #        yield game_match


def games_sample_iter(
    game_stream: GamesFile,
) -> typing.Generator[
    typing.Tuple[LeelaBoard, typing.Dict[str, typing.Any]], None, None
]:
    for _, game_str in game_stream:
        lines = get_game_info(game_str)
        board = None
        for l in lines:
            if board is None:
                board = LeelaBoard(fen=l["board"])
            yield board.copy(), l
            board.push(board.parse_uci(l["move"]))


def get_header_info(header_dict):
    gameVals = {}
    for name, func in per_game_infos.items():
        try:
            gameVals[name] = func(header_dict)
        except KeyError:
            gameVals[name] = None
    return gameVals


def get_game_info(
    input_game: typing.Union[str, chess.pgn.Game], no_clock=False
) -> typing.List[typing.Dict[str, typing.Any]]:
    if isinstance(input_game, str):
        game = chess.pgn.read_game(io.StringIO(input_game))
        assert game is not None
    else:
        game = input_game

    gameVals = {}
    for name, func in per_game_infos.items():
        try:
            gameVals[name] = func(game.headers)
        except KeyError:
            gameVals[name] = None

    gameVals["num_ply"] = len(list(game.mainline()))
    gameVals["termination_condition"] = get_termination_condition(
        gameVals["termination"], game
    )

    moves_values: typing.List[typing.Dict[str, typing.Any]] = []
    for i, node in enumerate(game.mainline()):
        # Caching
        board = node.parent.board()
        node_dict = gameVals.copy()
        node_dict["move_ply"] = i
        for name, func in per_move_funcs.items():
            node_dict[name] = func(node, gameVals)
        if len(node.comment) > 0:
            if r"%clk" in node.comment and not no_clock:
                for name, func in per_move_time_funcs.items():
                    node_dict[name] = func(node, gameVals)
            if r"%eval" in node.comment:
                for name, func in per_move_eval_funcs.items():
                    node_dict[name] = func(node, gameVals)
        moves_values.append(node_dict)
    return moves_values


def get_game_info_quick(
    input_game: str, no_clock=False
) -> typing.List[typing.Dict[str, typing.Any]]:
    game = chess.pgn.read_game(io.StringIO(input_game))
    assert game is not None
    gameVals = {}
    for name, func in per_game_infos.items():
        try:
            gameVals[name] = func(game.headers)
        except KeyError:
            gameVals[name] = None

    gameVals["num_ply"] = len(list(game.mainline()))
    moves_values = []
    for i, node in enumerate(game.mainline()):
        node_dict = gameVals.copy()
        node_dict["move_ply"] = i
        if i % 2 == 0:
            node_dict["active_player"] = gameVals["white_player"]
            node_dict["active_won"] = gameVals["white_won"]
            if i == 0:
                assert (
                    node.board().turn == chess.BLACK
                ), f"First move is not white, this is not a standard game: {input_game}"
        else:
            node_dict["active_player"] = gameVals["black_player"]
            node_dict["active_won"] = gameVals["black_won"]
        node_dict["move"] = per_move_funcs["move"](node, gameVals)
        if len(node.comment) > 0:
            if r"%clk" in node.comment and not no_clock:
                node_dict["pre_move_clock"] = per_move_time_funcs["pre_move_clock"](
                    node, gameVals
                )
        moves_values.append(node_dict)
    return moves_values


def make_game_info_mongo(header_dict: typing.Dict[str, str], game_str: str):
    data_dict = get_header_info(header_dict)
    if data_dict["white_elo"] == "?":
        data_dict["white_elo"] = None
    if data_dict["black_elo"] == "?":
        data_dict["black_elo"] = None
    data_dict["has_clock"] = game_clock_re.search(game_str) is not None
    data_dict["has_eval"] = game_eval_re.search(game_str) is not None
    data_dict["game_ply"] = len(move_num_re.findall(game_str))
    if data_dict["white_title"] == "":
        data_dict["white_title"] = None
    if data_dict["black_title"] == "":
        data_dict["black_title"] = None
    data_dict["full_date"] = datetime.datetime.fromisoformat(
        header_dict["UTCDate"].replace(".", "-") + " " + header_dict["UTCTime"]
    )
    data_dict["termination_condition"] = get_termination_condition(
        data_dict["termination"], chess.pgn.read_game(io.StringIO(game_str))
    )
    del data_dict["url"]
    data_dict["_id"] = f'{data_dict["game_source"]}_{data_dict["game_id"]}'
    data_dict["pgn"] = game_str
    return data_dict
