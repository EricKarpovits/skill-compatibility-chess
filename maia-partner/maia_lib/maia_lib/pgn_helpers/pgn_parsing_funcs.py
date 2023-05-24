import functools
import re
import typing
import uuid

import chess

from .cp_to_winrate import cp_to_winrate

time_regex = re.compile(r"\[%clk\s(\d+):(\d+):(\d+(\.\d+)?)\]", re.MULTILINE)
eval_regex = re.compile(r"\[%eval\s([0-9.+-]+)|(#(-)?[0-9]+)\]", re.MULTILINE)

white_opening_re = re.compile(r"(?P<d4>(?:A[4-9])|D|E)|(?P<other>A[0-3])|(?P<e4>B|C)")

black_openings_map = {
    "other": "(?:A[0-3]|A4[0-4]|B[01])",
    "Nf6": "A(?:4[5-9]|[567])",
    "f5": "A[89]",
    "c5": "B[2-9]",
    "e6": "C[01]",
    "e5": "C[2-9]",
    "d5": "D[0-6]",
    "grunfeld": "D[7-9]",
    "indian": "E",
}

black_strs = []
for k, v in black_openings_map.items():
    black_strs.append(f"(?P<{k}>{v})")
black_opening_re = re.compile(f"{'|'.join(black_strs)}")

low_time_threshold = 30

# Header


def gen_game_id(header: typing.Dict[str, str]) -> str:
    if "Link" in header:
        return header["Link"].split("/")[-1]
    elif "lichess" in header["Site"]:
        return header["Site"].split("/")[-1]
    else:
        return str(uuid.uuid4())


def get_game_source(header: typing.Dict[str, str]):
    if "lichess" in header["Site"]:
        return "lichess"
    elif "Chess.com" == header["Site"]:
        return "chess.com"
    return "unknown"


"""
https://github.com/lichess-org/lila/blob/da0c8241a07e003491baea0208db62e08af33b26/modules/tournament/src/main/Schedule.scala#L232
if (time < 30) UltraBullet
    else if (time < 60) HyperBullet
    else if (time < 120) Bullet
    else if (time < 180) HippoBullet
    else if (time < 480) Blitz
    else if (time < 1500) Rapid
    else Classical
"""


def gen_game_type(header: dict) -> typing.Union[str, None]:
    if "lichess" in header["Site"]:
        return (
            header["Event"]
            .split(" tournament")[0]
            .replace(" game", "")
            .replace("Rated ", "")
        )
    elif "Chess.com" == header["Site"]:
        increment = get_increment(header["TimeControl"])
        if increment < 60:
            return "UltraBullet"
        elif increment < 120:
            return "Bullet"
        elif increment < 480:
            return "Blitz"
        elif increment < 1500:
            return "Rapid"
        else:
            return "Classical"
    else:
        return None


def gen_url(header: typing.Dict[str, str]):
    if "lichess" in header["Site"]:
        return header["Site"]
    else:
        return header["Link"]


def safe_to_int(s):
    try:
        return int(s)
    except ValueError:
        return 0


def eco_white_move(eco_code):
    reg = white_opening_re.match(eco_code)
    try:
        for k, v in reg.groupdict().items():
            if v is not None:
                return k
        raise AttributeError("code not found")
    except AttributeError:
        raise KeyError(
            f"ECO code: '{eco_code}' not known, regex = '{white_opening_re.pattern}'"
        ) from None


def eco_black_move(eco_code):
    reg = black_opening_re.match(eco_code)
    try:
        for k, v in reg.groupdict().items():
            if v is not None:
                return k
        raise AttributeError("code not found")
    except AttributeError:
        raise KeyError(
            f"ECO code: '{eco_code}' not known, regex = '{black_opening_re.pattern}'"
        ) from None


def get_increment(ctr_str):
    try:
        return int(ctr_str.split("+")[-1])
    except ValueError:
        if ctr_str == "-":
            return 0
        else:
            return -1


per_game_infos = {
    "game_id": gen_game_id,
    "game_type": gen_game_type,
    "game_source": get_game_source,
    "date": lambda x: x["UTCDate"],
    "time": lambda x: x["UTCTime"],
    "datetime": lambda x: f"{x['UTCDate']} {x['UTCTime']}",
    "url": gen_url,
    "result": lambda x: x["Result"],
    "white_player": lambda x: x["White"],
    "black_player": lambda x: x["Black"],
    "white_title": lambda x: x.get("WhiteTitle", ""),
    "black_title": lambda x: x.get("BlackTitle", ""),
    "white_elo": lambda x: safe_to_int(x["WhiteElo"]),
    "black_elo": lambda x: safe_to_int(x["BlackElo"]),
    "eco": lambda x: x["ECO"],
    "time_control": lambda x: x["TimeControl"],
    "increment": lambda x: get_increment(x["TimeControl"]),
    "termination": lambda x: x["Termination"],
    "white_won": lambda x: x["Result"] == "1-0",
    "black_won": lambda x: x["Result"] == "0-1",
    "no_winner": lambda x: x["Result"] not in ["1-0", "0-1"],
}


"""
https://github.com/lichess-org/lila/blob/c48da4e2cd22837a9f6e8b7eb4a59924967aec3f/modules/game/src/main/PgnDump.scala#L136
case Created | Started                             => "Unterminated"
case Aborted | NoStart                             => "Abandoned"
case Timeout | Outoftime                           => "Time forfeit"
case Resign | Draw | Stalemate | Mate | VariantEnd => "Normal"
case Cheat                                         => "Rules infraction"
case UnknownFinish                                 => "Unknown"
"""


def get_lichess_termination(termination, game):
    if termination == "Normal":
        if "#" in str(game.mainline_moves())[-40:]:
            return "mate"
        elif game.headers["Result"] == "1/2-1/2":
            return "draw"
        else:
            return "resign"
    elif termination == "Time forfeit":
        return "time"
    elif termination == "Abandoned":
        return "abandoned"
    elif termination == "Unterminated":
        return "unterminated"
    elif termination == "Rules infraction":
        return "cheat"
    else:
        return "unknown"


"""
Known strings
won on time"]
won by resignation"]
won by checkmate"]
drawn by timeout vs insufficient material"]
drawn by stalemate"]
drawn by repetition"]
drawn by insufficient material"]
drawn by agreement"]
    """


def get_chesscom_termination(termination, game):
    if "won by resignation" in termination:
        return "resign"
    elif "won by checkmate" in termination:
        return "mate"
    elif "won on time" in termination:
        return "time"
    elif "drawn by" in termination:
        return "draw"
    else:
        return "unknown"


def get_termination_condition(termination, game):
    source = get_game_source(game.headers)
    if source == "lichess":
        return get_lichess_termination(termination, game)
    elif source == "chess.com":
        return get_chesscom_termination(termination, game)
    else:
        return "unknown"


# Moves

## No comment


@functools.lru_cache(maxsize=128)
def white_active(node):
    return node.board().turn == chess.BLACK


def get_previous_move(node, d):
    if node.parent is None:
        return ""
    else:
        return str(node.parent.move)


per_move_funcs = {
    "move": lambda n, d: str(n.move),
    "move_san": lambda n, d: str(n.parent.board().san(n.move)),
    "previous_move": get_previous_move,
    "board": lambda n, d: n.parent.board().fen(),
    "white_active": lambda n, d: white_active(n),
    "active_player": lambda n, d: d["white_player"]
    if white_active(n)
    else d["black_player"],
    "is_capture": lambda n, d: n.parent.board().is_capture(n.move),
    "is_check": lambda n, d: n.board().is_check(),
    "is_mate": lambda n, d: n.board().is_checkmate(),
    "active_won": lambda n, d: d["white_won"] if white_active(n) else d["black_won"],
    "active_elo": lambda n, d: d["white_elo"] if white_active(n) else d["black_elo"],
    "opponent_elo": lambda n, d: d["white_elo"]
    if not white_active(n)
    else d["black_elo"],
}

# Clock


@functools.lru_cache(maxsize=1024)
def get_move_clock(comment: str) -> int:
    timesRe = time_regex.search(comment)
    return (
        int(timesRe.group(1)) * 60 * 60
        + int(timesRe.group(2)) * 60
        + int(float(timesRe.group(3)))
    )


def get_opp_clck(node) -> int:
    pc = node.parent.comment
    if len(pc) > 0:
        return get_move_clock(pc)
    else:
        # Start of game
        return get_move_clock(node.comment)


def get_pre_move_clock(node) -> int:
    np = node.parent.parent
    # Start of game
    if np is None or len(np.comment) < 1:
        return get_move_clock(node.comment)
    return get_move_clock(np.comment)


def get_move_time(node, increment):
    delta = get_pre_move_clock(node) - get_move_clock(node.comment) + increment

    # Sometimes the opponent can give time, but only in 15 second increments
    if delta < 0:
        # why do math when you can use a for loop?
        while delta < 0:
            delta += 15
    return delta


def time_control_to_secs(timeStr, moves_per_game=35):
    if timeStr == "-":
        return 10800  # 180 minutes per side max on lichess
    else:
        try:
            t_base, t_add = timeStr.split("+")
            return int(t_base) + int(t_add) * moves_per_game
        except ValueError:
            return int(timeStr)


per_move_time_funcs = {
    "pre_move_clock": lambda n, d: get_pre_move_clock(n),
    "post_move_clock": lambda n, d: get_move_clock(n.comment),
    "pre_clock_percent": lambda n, d: get_pre_move_clock(n)
    / time_control_to_secs(d["time_control"]),
    "post_clock_percent": lambda n, d: get_move_clock(n.comment)
    / time_control_to_secs(d["time_control"]),
    "opp_clock": lambda n, d: get_opp_clck(n),
    "opp_clock_percent": lambda n, d: get_opp_clck(n)
    / time_control_to_secs(d["time_control"]),
    "move_time": lambda n, d: get_move_time(n, d["increment"]),
    "low_time": lambda n, d: get_move_clock(n.comment) < low_time_threshold,
}

# Eval
def get_move_eval(comment):
    if len(comment) < 1:
        return 0.1
    try:
        cp_str = eval_regex.search(comment).group(1)
    except AttributeError:
        return float("inf")
    try:
        return float(cp_str)
    except TypeError:
        if "-" in comment:
            return float("-inf")
        else:
            return float("inf")


def get_cp_loss(node):
    is_white = white_active(node)
    cp_par = get_move_eval(node.parent.comment) * (1 if is_white else -1)
    cp_aft = get_move_eval(node.comment) * (1 if is_white else -1)
    return cp_par - cp_aft


def get_move_wr(comment):
    return cp_to_winrate(get_move_eval(comment))


per_move_eval_funcs = {
    "cp": lambda n, d: get_move_eval(n.parent.comment),
    "cp_rel": lambda n, d: get_move_eval(n.parent.comment)
    * (1 if white_active(n) else -1),
    "cp_loss": lambda n, d: get_cp_loss(n),
    "winrate": lambda n, d: get_move_wr(n.parent.comment)
    if white_active(n)
    else (1 - get_move_wr(n.parent.comment)),
    "opp_winrate": lambda n, d: (1 - get_move_wr(n.parent.comment))
    if white_active(n)
    else get_move_wr(n.parent.comment),
    "winrate_loss": lambda n, d: (
        get_move_wr(n.parent.comment) - get_move_wr(n.comment)
    )
    if white_active(n)
    else (-get_move_wr(n.parent.comment) + get_move_wr(n.comment)),
}


full_funcs_lst = (
    ["move_ply"]
    + list(per_game_infos)
    + list(per_move_funcs)
    + list(per_move_time_funcs)
    + list(per_move_eval_funcs)
)
