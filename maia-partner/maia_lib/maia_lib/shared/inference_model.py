import io
import random
import typing

import chess
import numpy as np

from ..leela_board import LeelaBoard
from ..pgn_helpers import GamesFile


class MaiaNet_Base(object):
    def __init__(self, model_path) -> None:
        self.model_path = model_path

    def infer_board(
        self, board: typing.Union[str, chess.Board, LeelaBoard, np.ndarray]
    ):
        if isinstance(board, str) or isinstance(board, chess.Board):
            board = (
                LeelaBoard(board).lcz_features().reshape(1, 112, 64).astype("float32")
            )
        elif isinstance(board, LeelaBoard):
            board = board.lcz_features().reshape(1, 112, 64).astype("float32")
        elif isinstance(board, np.ndarray):
            board = board.reshape(-1, 112, 64).astype("float32")
        return self.model_eval(board)

    def model_eval(self, board):
        return NotImplemented

    def eval_boards(
        self,
        boards: typing.Union[
            typing.List[str], typing.List[chess.Board], typing.List[LeelaBoard]
        ],
        policy_softmax_temp: float = 1,
    ) -> typing.List[typing.Tuple[typing.Dict[str, float], float]]:

        boards = [self.board_convert(b) for b in boards]
        boards_arr = np.stack([b.lcz_features() for b in boards])

        pols, vals = self.infer_board(boards_arr)
        ret_vals = []
        for policy, val, board in zip(pols, vals, boards):
            ret_vals.append(
                self.make_outputs(
                    policy, val, board, policy_softmax_temp=policy_softmax_temp
                )
            )
        return ret_vals

    def eval_board(
        self, board, policy_softmax_temp: float = 1
    ) -> typing.Tuple[typing.Dict[str, float], float]:
        board = self.board_convert(board)
        pol, val = self.infer_board(board)
        return self.make_outputs(
            pol[0], val[0], board, policy_softmax_temp=policy_softmax_temp
        )

    def eval_board_search(
        self,
        board: typing.Union[str, chess.Board, LeelaBoard],
        rollouts: int,
        policy_temp: float = 1.0,
        c_puct: float = 1.0,
    ) -> typing.Tuple[typing.Dict[str, float], float]:
        board = self.board_convert(board)
        # assert isinstance(self, MaiaNet)
        # import here to prevent circular import
        from ..model_utils import MCTSNode

        root = MCTSNode.start_search(board, working_model=self, c_puct=c_puct)
        root.pick_move(rollouts, temperature=policy_temp)
        return root.make_policy(), (root.Q_sum / root.N) / 2.0 + 0.5

    def game_analysis(self, game_pgn):
        game = GamesFile(io.StringIO(game_pgn))
        boards = list(game.iter_moves())
        results = self.eval_boards([b for b, l in boards])
        ret_list = []
        for (pol, val), (_, row) in zip(results, boards):
            row["model_name"] = str(self)
            model_d = make_model_results_dict(pol, val, row)
            row.update(model_d)
            ret_list.append(row)
        return ret_list

    def get_top_move(
        self, board: typing.Union[str, chess.Board, LeelaBoard], policy_softmax_temp=1
    ) -> typing.Tuple[str, typing.Dict[str, float], float]:
        p, v = self.eval_board(board, policy_softmax_temp=policy_softmax_temp)
        return sorted(p.items(), key=lambda x: x[1], reverse=True)[0][0], p, v

    def get_top_move_search(
        self,
        board: typing.Union[str, chess.Board, LeelaBoard],
        rollouts: int,
        policy_temp: float = 1.0,
        c_puct: float = 1.0,
    ) -> typing.Tuple[str, typing.Dict[str, float], float]:
        p, v = self.eval_board_search(
            board, rollouts, policy_temp=policy_temp, c_puct=c_puct
        )
        return sorted(p.items(), key=lambda x: x[1], reverse=True)[0][0], p, v

    def get_random_move(
        self, board: typing.Union[str, chess.Board, LeelaBoard], policy_softmax_temp=1
    ):
        p, v = self.eval_board(board, policy_softmax_temp=policy_softmax_temp)
        p_l = list(p.items())
        return random.choices([m for m, p in p_l], weights=[p for m, p in p_l])[0], p, v

    @staticmethod
    def board_convert(board: typing.Union[str, chess.Board, LeelaBoard]) -> LeelaBoard:
        if isinstance(board, str) or isinstance(board, chess.Board):
            return LeelaBoard(board)
        return board

    @staticmethod
    def make_outputs(
        policy: typing.Dict[str, float],
        val: float,
        board,
        policy_softmax_temp: float = 1,
    ) -> typing.Tuple[typing.Dict[str, float], float]:
        try:
            return convert_outputs(
                policy, val, board, policy_softmax_temp=policy_softmax_temp
            )
        except ValueError:
            if board.pc_board.outcome():
                raise ValueError(
                    f"Terminal Board: {board.pc_board.outcome()} in '{board.pc_board.fen()}'"
                )
            else:
                raise


def _softmax(x, softmax_temp):
    e_x = np.exp((x - np.max(x)) / softmax_temp)
    return e_x / e_x.sum(axis=0)


def convert_outputs(
    policy, val, board, policy_softmax_temp: float = 1
) -> typing.Tuple[typing.Dict[str, float], float]:
    legal_uci = [m.uci() for m in board.generate_legal_moves()]
    legal_indexes = board.lcz_uci_to_idx(legal_uci)
    softmaxed = _softmax(policy[legal_indexes], policy_softmax_temp)

    return (
        {m: float(v) for v, m in sorted(zip(softmaxed, legal_uci), reverse=True)},
        _softmax(val, 1.0)[0],
    )


def maia_entropy(p_dict):
    a = np.array(list(p_dict.values()))
    return -1.0 * np.nansum(a * np.log2(a))


def make_model_results_dict(p_dict, val, row):
    model_moves = sorted(p_dict.items(), key=lambda x: x[1])
    model_move = model_moves[-1][0]
    model_p_1 = model_moves[-1][1]
    row_batch = {
        "model_value": val,
        "model_move": model_move,
        "player_move": row["move"],
        "model_top_policy": model_p_1,
        "player_move_policy": p_dict[row["move"]],
        "model_policy_dict": p_dict,
        "model_entropy": maia_entropy(p_dict),
        "num_above_player": len([x for x in model_moves if x[1] > p_dict[row["move"]]]),
        "num_below_player": len([x for x in model_moves if x[1] < p_dict[row["move"]]]),
    }
    row_batch["model_correct"] = row["move"] == row_batch["model_move"]
    return row_batch
