import numpy as np

from ..proto import GameSet_pb, GameInfo_pb, game_header_bytes, move_bytes_count,IgnorableGameSet_pb, IgnorableGameInfo_pb, ignorable_game_header_bytes, ignorable_move_bytes_count
from ..leela_board._leela_board import LeelaBoard

sampling_types = {
    'uniform' : {},
    'beta' : [1.4, 2.0],
    'probabilistic' : {'fraction' : 32}
}

max_clock = 600

class ProtoGamesReader():
    def __init__(self, p_handle, player_name = None, rng = None, sampling_type = 'uniform', sampling_options = None):
        self.close_after = False
        self.sampling_type = sampling_type
        if player_name is not None and sampling_type != 'probabilistic':
            raise NotImplementedError(f"{sampling_type} sampling not supported on player embeds")
        self.sampling_options = sampling_options
        self.player_name = player_name
        self.p_handle = p_handle
        if rng is None:
            self.rng = np.random.default_rng(123)
        else:
            self.rng = rng
        if isinstance(p_handle, str):
            self.handle = open(p_handle, 'rb')
            self.close_after = True
        else:
            self.handle = p_handle

    @staticmethod
    def get_lc0_board(target_index, moves, board_stack_size = 8, ignore_games =True):
        start_index = max(target_index - board_stack_size, 0)
        move_set = moves[start_index: target_index]
        #if ignore_games and move_set[0].ignore_move:
        #    raise ValueError(f"Ignorable move found at index {target_index}")
        m_strip = move_set[0].board.strip()
        lboard = LeelaBoard(m_strip)
        if len(move_set) > 1:
            for m in move_set[:-1]:
                mv_str = m.move.strip()
                lboard.push_uci(mv_str)
        if(ignore_games==True):
            return lboard.lcz_features(board_stack_size = board_stack_size), move_set[-1].ignore_move, move_set[-1].pre_move_clock / max_clock, lboard.lcz_uci_to_idx([move_set[-1].move.strip()])[0], move_set[-1].active_won, move_set[-1].move_time
        return lboard.lcz_features(board_stack_size = board_stack_size), move_set[-1].pre_move_clock / max_clock, lboard.lcz_uci_to_idx([move_set[-1].move.strip()])[0], move_set[-1].active_won, move_set[-1].move_time

    def iter_sample_games(self, board_stack_size = 8, ignore_games = True):
        if ignore_games:
            gs = IgnorableGameSet_pb()
        else:
            gs = GameSet_pb()
        self.handle.seek(0)
        gs.ParseFromString(self.handle.read())
        last_index = 0
        player_white = None
        for game_inf in gs.game_infos:
            if self.player_name is not None:
                if game_inf.white_player.strip() == self.player_name:
                    player_white = True
                elif game_inf.black_player.strip() == self.player_name:
                    player_white = False
                else:
                    raise KeyError(f"{self.player_name} not found in {self.p_handle}:\n{game_inf}")
            try:
                if self.sampling_type == 'beta':
                    r_index = int(self.rng.beta(*self.sampling_options) * game_inf.num_ply) + 1
                    yield (game_inf.game_id, r_index), self.get_lc0_board(r_index, gs.moves[last_index:last_index + game_inf.num_ply], board_stack_size = board_stack_size)
                elif self.sampling_type == 'uniform':
                    r_index = self.rng.integers(0, game_inf.num_ply) + 1
                    yield (game_inf.game_id, r_index), self.get_lc0_board(r_index, gs.moves[last_index:last_index + game_inf.num_ply], board_stack_size = board_stack_size)
                elif self.sampling_type == 'probabilistic':
                    if self.player_name is None:
                        for i in range(game_inf.num_ply):
                            if self.rng.integers(0, self.sampling_options['fraction']) == 0:
                                r_index = i + 1
                                yield (game_inf.game_id, r_index), self.get_lc0_board(r_index, gs.moves[last_index:last_index + game_inf.num_ply], board_stack_size = board_stack_size)
                    else:
                        for i in range(game_inf.num_ply):
                            if i % 2 and player_white:
                                continue
                            elif i % 2 == 0 and not player_white:
                                continue
                            if self.rng.integers(0, self.sampling_options['fraction']) == 0:
                                r_index = i + 1
                                yield (game_inf.game_id, r_index), self.get_lc0_board(r_index, gs.moves[last_index:last_index + game_inf.num_ply], board_stack_size = board_stack_size)
                else:
                    raise NotImplementedError(f"{self.sampling_type} is not a known sampling function, {sampling_types.keys().join(', ')} are the valid options")
            except ValueError:
                if ignore_games:
                    continue
            last_index += game_inf.num_ply

    def get_games_info(self, ignore_games = True):
        self.handle.seek(0)
        if ignore_games:
            gs_tmp = IgnorableGameSet_pb()
        else:
            gs_tmp = GameSet_pb()
        gs_tmp.ParseFromString(self.handle.read(5))
        num_games = gs_tmp.num_games
        game_infos = {}
        game_count = 0
        self.handle.read(2)
        if ignore_games:
            for i in range(num_games):
                gi = IgnorableGameInfo_pb()
                gi.ParseFromString(self.handle.read(ignorable_move_bytes_count))
                start_byte = 7 + ( num_games * (ignorable_move_bytes_count + 2) ) + (game_count * (ignorable_game_header_bytes + 2))
                game_infos[gi.game_id] = (gi.num_ply, start_byte)
                self.handle.read(2)
                game_count += gi.num_ply
        else:
            for i in range(num_games):
                gi = GameInfo_pb()
                gi.ParseFromString(self.handle.read(move_bytes_count))
                start_byte = 7 + ( num_games * (move_bytes_count + 2) ) + (game_count * (game_header_bytes + 2))
                game_infos[gi.game_id] = (gi.num_ply, start_byte)
                self.handle.read(2)
                game_count += gi.num_ply
        return num_games, game_infos

    def __del__(self):
        if self.close_after:
            try:
                self.handle.close()
            except:
                pass
