import io
import re

import chess
import chess.pgn
import numpy as np

from ._leela_board import LeelaBoard

piece_lookup = {
    1: "P",
    2: "N",
    3: "B",
    4: "R",
    5: "Q",
    6: "K",
    0: "1",
}
piece_lookup.update({-k: v.lower() for k, v in piece_lookup.items()})

oneone_re = re.compile("11+")


def counter(s):
    return str(len(s.group(0)))


def leela_game(input_game):
    if isinstance(input_game, str):
        game = chess.pgn.read_game(io.StringIO(input_game))
    else:
        game = input_game
    lc_board = LeelaBoard(game.board())
    features = [lc_board.lcz_features()]
    for node in game.mainline():
        lc_board.push(node.move)
        features.append(lc_board.lcz_features())
    return np.stack(features[:-1], axis=0)


class HTMLWrapper(str):
    def _repr_html_(self):
        return self


def display_arr(arr, as_grid=False):
    table_vals = []
    is_white = int(arr[108].mean())
    for i in range(4):
        row_vals = []
        for k in range(2):
            ap = arr[i * 26 + (13 * k) : i * 26 + 12 + (13 * k)]
            board_arr = make_board_arr(ap)
            if as_grid:
                row_vals.append(
                    (
                        grid_plot(ap),
                        arr[i * 26 + 12 + (13 * k)].mean() > 0,
                    )
                )
            else:
                row_vals.append(
                    (
                        str(
                            chess.svg.board(
                                chess.Board(
                                    boar_arr_to_fen(
                                        board_arr,
                                        active_white=is_white % 2,
                                        no_reorder=True,
                                    )
                                )
                            )
                        ),
                        arr[i * 26 + 12 + (13 * k)].mean() > 0,
                    )
                )
            is_white += 1
        if i < 1:
            extra_style = """style="background-color: coral" """
        else:
            extra_style = ""
        table_vals.append(
            f"""
                    <tr>
                    <td {extra_style} >{row_vals[0][0]} <p>Is Repetition: {row_vals[0][1]}</p></td>
                    <td>{row_vals[1][0]} <p>Is Repitition: {row_vals[0][1]}</p></td>
                    <td>{get_extra_info(i, arr)}</td>
                    </tr>"""
        )
    return HTMLWrapper(
        f"""<table style="width: 600px">
    <tr>
    <th>Active Player</th>
    <th>Opponent</th>
    <th>Info</th>
    </tr>
    {' '.join(table_vals)}
    </table>
    """
    )


def arr_to_board(arr):
    return chess.Board(boar_arr_to_fen(make_board_arr(arr)))


def make_board_arr(arr):
    a_ret = np.zeros([8, 8])
    for i in range(6):
        a_ret += arr[i] * (i + 1)
        a_ret -= arr[i + 6] * (i + 1)
    return a_ret


def boar_arr_to_fen(arr, active_white=True, no_reorder=False):
    ret_s = []
    for i in range(8):
        row_s = ""
        for j in range(8):
            row_s += piece_lookup[int(arr[i, j])]
        ret_s.append(row_s)
    board_st = oneone_re.sub(counter, "/".join(ret_s))
    if active_white:
        return board_st + " w"
    else:
        if no_reorder:
            return board_st.swapcase() + " b"
        else:
            return "/".join(board_st.swapcase().split("/")[::-1]) + " b"


def annot_arr(arr):
    ret_s = []
    for i in range(8):
        row_s = []
        for j in range(8):
            row_s.append(piece_lookup[int(arr[i, j])])
        ret_s.append([s.replace("1", "") for s in row_s])
    return ret_s


def l_plt(arr):
    import matplotlib.pyplot as plt
    import seaborn

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=[4, 8])
    for i in range(4):
        for k in range(2):
            ap = arr[i * 26 + (13 * k) : i * 26 + 12 + (13 * k)]
            board_arr = make_board_arr(ap)
            seaborn.heatmap(
                make_board_arr(ap),
                ax=axes[i, k],
                cbar=False,
                annot=annot_arr(board_arr),
            )


def get_extra_info(i, arr):
    if i == 0:
        if arr[108].mean():
            return "<p>Active is Black</p>"
        else:
            return "<p>Active is White</p>"
    elif i == 1:
        options = []
        if arr[104].mean():
            options.append("White can O-O-O")
        else:
            options.append("White cannot O-O-O")
        if arr[105].mean():
            options.append("White can O-O")
        else:
            options.append("Black cannot O-O")
        if arr[106].mean():
            options.append("Black can O-O-O")
        else:
            options.append("Black cannot O-O-O")
        if arr[107].mean():
            options.append("Black can O-O-O")
        else:
            options.append("Black cannot O-O-O")
        return "<p>" + "</p><p>".join(options) + "</p>"
    elif i == 2:
        return f"<p>Rull 50: {arr[109].mean() * 50:.0f}/50 = {arr[109].mean() :.2f}</p>"
    else:
        return ""


def grid_plot(arr):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=[5, 2])
    for i in range(6):
        for j in range(2):
            axes[j, i].imshow(
                arr[i + (6 * j)],
                cmap="Greys",
                vmax=1,
                vmin=-1,
                interpolation="nearest",
            )
            axes[j, i].set_axis_off()
    f = io.BytesIO()
    plt.savefig(f, transparent=True, format="svg", pad_inches=0, bbox_inches="tight")
    fig.tight_layout()
    plt.close()
    return f.getvalue().decode("utf8")
