import re

import numpy as np
import pandas as pd

from .regexes import (
    RX_BAT_OUT,
    RX_BATTER_NAME,
    RX_BATTER_VERBS,
    RX_BB,
    RX_DOUBLE,
    RX_DP,
    RX_FC,
    RX_HBP,
    RX_HR,
    RX_OUT,
    RX_PLAY_VERB,
    RX_REACH,
    RX_REACHED,
    RX_RUNNER_NAME,
    RX_RUNNER_ONLY_VERBS,
    RX_RUNNER_P1_NAME,
    RX_SINGLE,
    RX_SUB_LINE,
    RX_TO_2,
    RX_TO_3,
    RX_TO_H,
    RX_TP,
    RX_TRIPLE,
)


def split_players_text(desc):
    if not isinstance(desc, str) or not desc:
        return "", "", "", ""
    parts = re.split(r"(?:;|3a|:)", desc)
    parts = [p.strip() for p in parts if p is not None]
    parts = (parts + ["", "", "", ""])[:4]
    return parts[0], parts[1], parts[2], parts[3]


def infer_outs_from_fc(p1_text: str, has_p2: bool, has_p3: bool, has_p4: bool, outs_already: int) -> int:
    if outs_already:
        return 0
    if not isinstance(p1_text, str) or not p1_text:
        return 0
    if RX_REACHED.search(p1_text):
        return 0
    if not RX_FC.search(p1_text):
        return 0
    if RX_DP.search(p1_text) or RX_TP.search(p1_text):
        return 0
    if not has_p2 and not has_p3 and not has_p4:
        return 1
    return 0

def bases_str(r1: str, r2: str, r3: str) -> str:
    return f"{'Y' if str(r1).strip() else 'N'}{'Y' if str(r2).strip() else 'N'}{'Y' if str(r3).strip() else 'N'}"

def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x).strip()

def blank_if_sub_or_meta(p1: str, sub_fl: int) -> bool:
    if sub_fl == 1:
        return True
    if not p1:
        return True
    if RX_SUB_LINE.search(p1):
        return True
    if p1.startswith("("):
        return True
    if not RX_PLAY_VERB.search(p1):
        return True
    return False

def is_runner_only_event(p1_text: str) -> bool:
    p1 = _s(p1_text)
    if not p1:
        return False
    if RX_BATTER_VERBS.search(p1):
        return False
    if RX_RUNNER_ONLY_VERBS.search(p1):
        return True
    return False

def extract_runner_name_from_p1(p1_text: str) -> str:
    p1 = _s(p1_text)
    if not p1:
        return ""
    m = RX_RUNNER_P1_NAME.search(p1)
    return m.group("name").strip() if m else ""

def extract_batter_name(p1_text: str, sub_fl: int) -> str:
    p1 = _s(p1_text)
    if blank_if_sub_or_meta(p1, sub_fl):
        return ""
    if is_runner_only_event(p1):
        return ""
    m = RX_BATTER_NAME.search(p1)
    return m.group("name").strip() if m else ""

def extract_runner_name(px_text: str) -> str:
    t = _s(px_text)
    if not t:
        return ""
    m = RX_RUNNER_NAME.search(t)
    return m.group("name").strip() if m else ""

def bat_order_id(df: pd.DataFrame) -> pd.Series:
    side = np.where(df["half"] == "Top", "A", "H")

    is_bat = df["batter_name"].fillna("").astype(str).str.strip().ne("")

    pa_idx = is_bat.astype("int32").groupby([df["contest_id"], side], sort=False).cumsum()

    bo = ((pa_idx - 1) % 9 + 1).where(is_bat, pd.NA)

    return bo.astype("Int16")

def bat_order_fill(df: pd.DataFrame, bat_order: pd.Series) -> pd.Series:
    side = np.where(df["half"] == "Top", "A", "H")
    keys = [df["contest_id"], pd.Series(side, index=df.index)]

    filled = bat_order.groupby(keys, sort=False).ffill()

    filled = filled.groupby(keys, sort=False).bfill()

    return filled.astype("Int16")

def batter_dest(p1_text: str) -> str:
    p1 = _s(p1_text)
    if not p1:
        return ""
    if RX_HR.search(p1):
        return "H"   # batter scores; we'll clear bases conservatively
    if RX_TRIPLE.search(p1):
        return "3"
    if RX_DOUBLE.search(p1):
        return "2"
    if RX_SINGLE.search(p1):
        return "1"
    if RX_BB.search(p1) or RX_HBP.search(p1) or RX_REACH.search(p1):
        return "1"
    if RX_BAT_OUT.search(p1):
        return "OUT"
    return ""

def runner_dest(px_text: str) -> str:
    t = _s(px_text)
    if not t:
        return ""
    if RX_OUT.search(t):
        return "OUT"
    if RX_TO_H.search(t):
        return "H"
    if RX_TO_3.search(t):
        return "3"
    if RX_TO_2.search(t):
        return "2"
    return ""


__all__ = [
    "split_players_text",
    "infer_outs_from_fc",
    "bases_str",
    "extract_batter_name",
    "extract_runner_name",
    "extract_runner_name_from_p1",
    "is_runner_only_event",
    "batter_dest",
    "runner_dest",
    "blank_if_sub_or_meta",
    "bat_order_id",
    "bat_order_fill",
]
