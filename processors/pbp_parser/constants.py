from enum import StrEnum


class BattedBallType(StrEnum):
    GROUND_BALL = "GB"
    LINE_DRIVE = "LD"
    FLY_BALL = "FB"
    POP_UP = "PU"
    BUNT = "BU"


class EventType(StrEnum):
    UNKNOWN = "UNK"
    NO_PLAY = "NP"
    GENERIC_OUT = "OUT"
    STRIKEOUT = "SO"
    STOLEN_BASE = "SB"
    DEFENSIVE_INDIFF = "DEF_IND"
    CAUGHT_STEALING = "CS"
    PICKOFF_ERROR = "POE"
    PICKOFF = "PO"
    WILD_PITCH = "WP"
    PASSED_BALL = "PB"
    BALK = "BK"
    OTHER_ADVANCE = "ADV"
    FOUL_ERROR = "FE"
    WALK = "BB"
    INTENTIONAL_WALK = "IBB"
    HIT_BY_PITCH = "HBP"
    INTERFERENCE = "CI"
    ERROR = "E"
    FIELDERS_CHOICE = "FC"
    SINGLE = "1B"
    DOUBLE = "2B"
    TRIPLE = "3B"
    HOME_RUN = "HR"
    STRIKEOUT_PASSED_BALL = "SO_PB"
    STRIKEOUT_WILD_PITCH = "SO_WP"


POS_MAP = {
    "p": "p",
    "pitcher": "p",
    "c": "c",
    "catcher": "c",
    "1b": "1b",
    "first baseman": "1b",
    "first base": "1b",
    "2b": "2b",
    "second baseman": "2b",
    "second base": "2b",
    "3b": "3b",
    "third baseman": "3b",
    "third base": "3b",
    "ss": "ss",
    "shortstop": "ss",
    "lf": "lf",
    "left fielder": "lf",
    "left field": "lf",
    "cf": "cf",
    "center fielder": "cf",
    "center field": "cf",
    "rf": "rf",
    "right fielder": "rf",
    "right field": "rf",
    "dh": "dh",
    "designated hitter": "dh",
    "ph": "ph",
    "pinch hitter": "ph",
    "pr": "pr",
    "pinch runner": "pr",
}


def canon_pos(x: str) -> str:
    if not isinstance(x, str):
        return ""
    s = x.strip().lower()
    return POS_MAP.get(s, s)


__all__ = ["BattedBallType", "EventType", "POS_MAP", "canon_pos"]
