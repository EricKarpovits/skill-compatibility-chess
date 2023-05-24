import datetime
import os.path
import signal
import typing

import __main__
import pytz
import setproctitle

tz = pytz.timezone("Canada/Eastern")

__version__ = "0.0.8"


def get_cur_datetime() -> str:
    return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def print_with_date(string: str, flush: bool = False, **kwargs: dict) -> None:
    """Print a string prefixed with the current date and time in Toronto. kwargs are passed to print()"""
    print(
        f"{get_cur_datetime()} {string}",
        flush=flush,
        **kwargs,
    )


def init_non_capture_worker(title: typing.Optional[str] = None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if title is not None:
        set_proc_title(title)


def set_proc_title(title: str) -> str:
    if hasattr(__main__, "__file__"):
        set_title = f"{os.path.basename(__main__.__file__)} {title}"
    else:
        set_title = title
    setproctitle.setproctitle(set_title)
    return set_title
