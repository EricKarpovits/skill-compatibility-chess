import dateutil
import dateutil.parser

from .data_files_pb2 import GameSet as GameSet_pb, GameInfo as GameInfo_pb, Move as Move_pb
from .data_files_ignorable_pb2 import IgnorableGameSet as IgnorableGameSet_pb, IgnorableGameInfo as IgnorableGameInfo_pb, IgnorableMove as IgnorableMove_pb
from ..pgn_handling.pgn_parser import get_game_info_minumal, get_game_info_mongo

game_header_bytes = 126
move_bytes_count = 84
ignorable_game_header_bytes = 128
ignorable_move_bytes_count = 84

#game_header_bytes = 116
#move_bytes_count = 64

class NoClockError(Exception):
    pass

def write_games_mongo_batch(f_path, batch):
    gs = GameSet_pb()
    for game_str, result, bitstring, valar in batch:
        #print(bitstring)
        inf, moves = make_game_inf_mongo(game_str, result,bitstring=bitstring ,valar=valar)
        if len(moves) > 0:
            for m in moves:
                gs.moves.append(m)
            gs.game_infos.append(inf)
            gs.num_games += 1
    with open(f_path, 'wb') as f:
        f.write(gs.SerializeToString())
def write_games_mongo_batch2(f_path, batch):
    gs = GameSet_pb()
    for game_str, result, valar in batch:
        inf, moves = make_game_inf_mongo(game_str, result,valar)
        if len(moves) > 0:
            for m in moves:
                gs.moves.append(m)
            gs.game_infos.append(inf)
            gs.num_games += 1
    with open(f_path, 'wb') as f:
        f.write(gs.SerializeToString())
def write_games_batch(f_path, batch, min_clock, ignore_no_clock = False):
    gs = GameSet_pb()
    for header_dict, game_str in batch:
        try:
            inf, moves = make_game_inf(header_dict, game_str, min_clock)
        except NoClockError:
            if ignore_no_clock:
                continue
            else:
                raise
        if len(moves) > 0:
            for m in moves:
                gs.moves.append(m)
            gs.game_infos.append(inf)
            gs.num_games += 1
    with open(f_path, 'wb') as f:
        f.write(gs.SerializeToString())


def write_games_batch_ignore(f_path, batch):
    gs = IgnorableGameSet_pb()
    for game_str, result, bitstring, valar in batch:
        inf, moves = make_game_inf( game_str, result, bitstring=bitstring, valar=valar)
        if len(moves) > 0:
            for m in moves:
                gs.moves.append(m)
            gs.game_infos.append(inf)
            gs.num_games += 1
    with open(f_path, 'wb') as f:
        f.write(gs.SerializeToString())

def make_game_inf(game_str, result, valar=[], bitstring=""):
    valar2=valar.copy()
    moves = get_game_info_mongo(game_str, result,valar2,bitstring)
    #print(bitstring)
    if bitstring!="":
        inf =  IgnorableGameInfo_pb()
        target_game_header_bytes = ignorable_game_header_bytes
        target_move_bytes_count = ignorable_move_bytes_count
    else:
        inf =  GameInfo_pb()
        target_game_header_bytes = game_header_bytes
        target_move_bytes_count = move_bytes_count
    #print(target_game_header_bytes,ignorable_game_header_bytes)
    if len(moves) < 1:
        return None, []
    inf.game_id = "12345678"
    inf.white_player = ''.ljust(20)
    inf.black_player = ''.ljust(20)
    inf.white_elo = 0
    inf.black_elo = 0
    inf.timestamp = 0

    inf.start_time = 0
    inf.increment = 0

    ret_moves = []
    move_count = 0
    last_wclock = 0
    last_bclock = 0
    for i, m in enumerate(moves):
        move_count += 1
        if bitstring!="":
            mov = IgnorableMove_pb()
            mov.ignore_move=bool(m['ignore'])
        else:
            mov = Move_pb()
        mov.is_white = (i + 1) % 2
        mov.active_won = int(m['active_won'])
        mov.no_winner = int(m['no_winner'])
        mov.move_ply = i
        mov.board = m['board'].strip().ljust(90)
        mov.move = m['move'].strip().ljust(6)
        mov.pre_move_clock = 0
        mov.opp_clock = 0
        mov.move_time = 0
        
        try:
            assert len(mov.SerializeToString()) == target_game_header_bytes
        except AssertionError:
            print(m, repr(m['board']),mov.SerializeToString(), len(mov.SerializeToString()))
            raise
        ret_moves.append(mov)
    inf.num_ply = move_count
    try:
        assert len(inf.SerializeToString()) == target_move_bytes_count
    except AssertionError:
        print(len(inf.SerializeToString()), move_bytes_count)
        raise
    return inf, ret_moves

def make_game_inf_mongo(game_str, result,valar=[], bitstring=""):
    valar2=valar.copy()
    moves = get_game_info_mongo(game_str, result,valar2,bitstring)
    if len(moves) < 1:
        return None, []
    inf =  GameInfo_pb()
    
    inf.game_id = "12345678"
    inf.white_player = ''.ljust(20)
    inf.black_player = ''.ljust(20)
    inf.white_elo = 0
    inf.black_elo = 0
    inf.timestamp = 0

    inf.start_time = 0
    inf.increment = 0

    ret_moves = []
    move_count = 0
    last_wclock = 0
    last_bclock = 0
    for i, m in enumerate(moves):
        move_count += 1
        mov = Move_pb()
        mov.is_white = (i + 1) % 2
        mov.active_won = int(m['active_won'])
        mov.no_winner = int(m['no_winner'])
        mov.move_ply = i
        mov.board = m['board'].strip().ljust(90)
        mov.move = m['move'].strip().ljust(6)
        mov.pre_move_clock = 0
        mov.opp_clock = 0
        mov.move_time = 0
        
        try:
            assert len(mov.SerializeToString()) == game_header_bytes #those myt not be ryt
        except AssertionError:
            print(m, repr(m['board']),mov.SerializeToString(), len(mov.SerializeToString()))
            raise
        ret_moves.append(mov)
    inf.num_ply = move_count
    try:
        assert len(inf.SerializeToString()) == bytes_count
    except AssertionError:
        print(len(inf.SerializeToString()), game_move_bytes_count) #that too
        raise
    return inf, ret_moves
