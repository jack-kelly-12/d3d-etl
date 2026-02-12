from enum import IntEnum, StrEnum


class BattedBallType(StrEnum):
    GROUND_BALL = "GB"
    LINE_DRIVE = "LD"
    FLY_BALL = "FB"
    POP_UP = "PU"
    BUNT = "BU"


class EventType(IntEnum):
    UNKNOWN = 0
    NO_PLAY = 1
    GENERIC_OUT = 2
    STRIKEOUT = 3
    STOLEN_BASE = 4
    DEFENSIVE_INDIFF = 5
    CAUGHT_STEALING = 6
    PICKOFF_ERROR = 7
    PICKOFF = 8
    WILD_PITCH = 9
    PASSED_BALL = 10
    BALK = 11
    OTHER_ADVANCE = 12
    FOUL_ERROR = 13
    WALK = 14
    INTENTIONAL_WALK = 15
    HIT_BY_PITCH = 16
    INTERFERENCE = 17
    ERROR = 18
    FIELDERS_CHOICE = 19
    SINGLE = 20
    DOUBLE = 21
    TRIPLE = 22
    HOME_RUN = 23
    STRIKEOUT_PASSED_BALL = 24
    STRIKEOUT_WILD_PITCH = 25


POS_MAP = {
    "p": "p", "pitcher": "p",
    "c": "c", "catcher": "c",
    "1b": "1b", "first baseman": "1b", "first base": "1b",
    "2b": "2b", "second baseman": "2b", "second base": "2b",
    "3b": "3b", "third baseman": "3b", "third base": "3b",
    "ss": "ss", "shortstop": "ss",
    "lf": "lf", "left fielder": "lf", "left field": "lf",
    "cf": "cf", "center fielder": "cf", "center field": "cf",
    "rf": "rf", "right fielder": "rf", "right field": "rf",
    "dh": "dh", "designated hitter": "dh",
    "ph": "ph", "pinch hitter": "ph",
    "pr": "pr", "pinch runner": "pr",
}


def canon_pos(x: str) -> str:
    if not isinstance(x, str):
        return ""
    s = x.strip().lower()
    return POS_MAP.get(s, s)


__all__ = ["BattedBallType", "EventType", "POS_MAP", "canon_pos"]
