import re
import bz2
import io
import collections.abc
import numpy as np
import chess.pgn

from .pgn_parsing_funcs import per_game_infos, per_move_funcs,per_move_time_funcs, per_move_eval_funcs

from ..leela_board import LeelaBoard

_header=r'''\[([A-Za-z0-9_]+)\s+"((?:[^"]|\\")*)"\]'''
header_re =re.compile(_header)
_headers=r'(' + _header + r'\s*\n)+\n'
_moves=r'[^\[]*(\*|1-0|0-1|1/2-1/2)\s*\n'
_move = r"""([NBKRQ]?[a-h]?[1-8]?[\-x]?[a-h][1-8](?:=?[nbrqkNBRQK])?|[PNBRQK]?@[a-h][1-8]|--|Z0|O-O(?:-O)?|0-0(?:-0)?)|(\{.*)|(;.*)|(\$[0-9]+)|(\()|(\))|(\*|1-0|0-1|1\/2-1\/2)|([\?!]{1,2})"""

game_counter_re = re.compile('\[Event "')

game_re=re.compile("(" + _headers +")(.*?)(\*|1-0|0-1|1\/2-1\/2)", re.MULTILINE | re.DOTALL)

class GamesFile(collections.abc.Iterator):
    def __init__(self, path):
        if isinstance(path, io.StringIO):
            self.file_string = path.read()
        elif path.endswith('bz2'):
            with bz2.open(path, 'rt') as f:
                self.file_string = f.read()
        else:
            with open(path, 'r') as f:
                self.file_string = f.read()
        self.path = path
        self._len = None
        self.re_iter =game_re.finditer(self.file_string)

    def __len__(self):
        if self._len is None:
            self._len = len(game_counter_re.findall(self.file_string))
        return self._len

    def __next__(self):
        r = next(self.re_iter)
        try:
            header = header_re.findall(r.group(1))
        except AttributeError:
            print(r)
        return { k : v for k, v in header}, r.group(0)

    def iter_moves(self, no_clock = False):
        return games_move_iter(self, no_clock = no_clock)

    def iter_samples(self):
        return games_sample_iter(self)

class GamesFileStream(GamesFile):
    def __init__(self, path):
        if isinstance(path, io.StringIO) or isinstance(path, io.TextIOWrapper):
            self.file = path
        elif path.endswith('bz2'):
            self.file = bz2.open(path, 'rt')
        else:
            self.file = open(path, 'rt')
        self.path = path
        self._len = -1
        self.re_iter = stream_iter(self.file)#game_re.finditer(self.file_string)

    def __del__(self):
        try:
            self.file.close()
        except:
            pass

def stream_iter(file_handle):
    current_game=file_handle.readline()
    for line in file_handle:
        if line.startswith('[Event '):
            gr = game_re.match(current_game.strip())
            if gr is not None:
                yield gr
            current_game = ''
        current_game += line
    if len(current_game.strip()) > 0:
        yield game_re.match(current_game.strip())

def games_move_iter(game_stream, no_clock = False):
    for _, game_str in game_stream:
        lines = get_game_info(game_str, no_clock = no_clock)
        for l in lines:
            yield l

def games_sample_iter(game_stream):
    for _, game_str in game_stream:
        lines = get_game_info(game_str)
        board = None
        for l in lines:
            if board is None:
                board = LeelaBoard(fen = l['board'])
            yield board.copy(), l
            board.push_uci(l['move'])

def get_header_info(header_dict):
    gameVals = {}
    for name, func in per_game_infos.items():
        try:
            gameVals[name] = func(header_dict)
        except KeyError:
            gameVals[name] = None
    return gameVals

def get_game_info_minumal(input_game):
    """For protobuf/training data"""
    if isinstance(input_game, str):
        game = chess.pgn.read_game(io.StringIO(input_game))
    else:
        game = input_game

    gameVals = {}
    for value in ['game_id', 'no_winner', 'black_won', 'white_won', 'increment', 'white_elo', 'black_elo', 'time_control']:
        gameVals[value] = per_game_infos[value](game.headers)

    moves_values = []
    parent_board = game.board()
    for i, node in enumerate(game.mainline()):
        node_dict = gameVals.copy()
        parent_board.push(node.move)
        board = node.parent.board()
        node_dict['board'] = board.fen()
        node_dict['move'] = per_move_funcs['move'](node, gameVals)
        node_dict['active_won'] = gameVals['white_won'] if board.turn == chess.WHITE else gameVals['black_won']
        if len(node.comment) > 0 and r'%clk' in node.comment:
            node_dict['pre_move_clock'] = per_move_time_funcs['pre_move_clock'](node, gameVals)
            node_dict['opp_clock'] = per_move_time_funcs['opp_clock'](node, gameVals)
            node_dict['move_time'] = per_move_time_funcs['move_time'](node, gameVals)
        moves_values.append(node_dict)
    return moves_values

def get_game_info(input_game, no_clock = False):
    if isinstance(input_game, str):
        game = chess.pgn.read_game(io.StringIO(input_game))
    else:
        game = input_game

    gameVals = {}
    for name, func in per_game_infos.items():
        try:
            gameVals[name] = func(game.headers)
        except KeyError:
            gameVals[name] = None

    gameVals['num_ply'] = len(list(game.mainline()))

    moves_values = []
    for i, node in enumerate(game.mainline()):
        #Caching
        board = node.parent.board()
        node_dict = gameVals.copy()
        node_dict['move_ply'] = i
        for name, func in per_move_funcs.items():
            node_dict[name] = func(node, gameVals)
        if len(node.comment) > 0:
            if r'%clk' in node.comment and not no_clock:
                for name, func in per_move_time_funcs.items():
                    node_dict[name] = func(node, gameVals)
            if r'%eval' in node.comment:
                for name, func in per_move_eval_funcs.items():
                    node_dict[name] = func(node, gameVals)
        moves_values.append(node_dict)
    return moves_values
def get_game_info_mongo(input_game, result, valar=[], bitstring=""):
    """For protobuf/training data"""
    if isinstance(input_game, str):
        game = chess.pgn.read_game(io.StringIO(input_game))
    else:
        game = input_game
    gameVals = {
        'no_winner' : True if result == 0 else False,
        'black_won' : True if result == -1 else False,
         'white_won' : True if result == 1 else False,
        }
    moves_values = []
    parent_board = game.board()
    for i, node in enumerate(game.mainline()):
        parent_board.push(node.move)
        board = node.parent.board()
        #print(i,board.fen())
        if(len(valar)>0):
            winner=np.sign(valar[i])
            if(abs(valar[i])<10):
                winner=0
            gameVals = {
            'no_winner' : True if winner == 0 else False,
            'black_won' : True if winner == -1 else False,
            'white_won' : True if winner == 1 else False,
            }
        node_dict = gameVals.copy()
        node_dict['board'] = board.fen()
        node_dict['move'] = per_move_funcs['move'](node, gameVals)
        node_dict['active_won'] = gameVals['white_won'] if board.turn == chess.WHITE else gameVals['black_won']
        if(bitstring=="" or bitstring[i]=='1'):
            node_dict['ignore']=False
        else:
            node_dict['ignore']=True
        moves_values.append(node_dict)
    return moves_values