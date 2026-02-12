import re

import numpy as np
import pandas as pd

from .constants import BattedBallType, EventType, canon_pos
from .helpers import (
    bat_order_fill,
    bat_order_id,
    batter_dest,
    blank_if_sub_or_meta,
    extract_batter_name,
    extract_runner_name,
    extract_runner_name_from_p1,
    infer_outs_from_fc,
    is_runner_only_event,
    runner_dest,
    split_players_text,
)
from .regexes import (
    RX_ADVANCE,
    RX_BALK,
    RX_BATTER_OUT,
    RX_BB,
    RX_CAUGHT_STEALING,
    RX_CI,
    RX_DEFENSIVE_INDIFF,
    RX_DOUBLE,
    RX_DP,
    RX_DROPPED_FOUL,
    RX_ERROR,
    RX_FC,
    RX_HBP,
    RX_HR,
    RX_IBB,
    RX_INTERFERENCE,
    RX_K,
    RX_K_PB,
    RX_K_SAFE,
    RX_K_WP,
    RX_LINEUP_CHANGE,
    RX_NO_PLAY,
    RX_PASSED_BALL,
    RX_PICKOFF,
    RX_PICKOFF_ERROR,
    RX_REACHED,
    RX_RUNNER_OUT,
    RX_SINGLE,
    RX_STOLEN_BASE,
    RX_TP,
    RX_TRIPLE,
    RX_WILD_PITCH,
)


def outs_on_play(p1_text: pd.Series, p2_text: pd.Series, p3_text: pd.Series, p4_text: pd.Series) -> tuple[pd.Series, pd.Series]:
    def _calc_outs(row):
        texts = [row.iloc[0] if isinstance(row.iloc[0], str) else "",
                 row.iloc[1] if isinstance(row.iloc[1], str) else "",
                 row.iloc[2] if isinstance(row.iloc[2], str) else "",
                 row.iloc[3] if isinstance(row.iloc[3], str) else ""]
        texts = [t.strip() for t in texts]
        full = " ".join(t for t in texts if t)

        if not full:
            return 0, ""

        if RX_CI.search(full):
            return 0, "CATCH_INTERF"
        if RX_TP.search(full):
            return 3, "TRIPLE_PLAY"
        if RX_DP.search(full):
            return 2, "DOUBLE_PLAY"

        outs = 0
        reasons = []

        for t in texts:
            if not t:
                continue

            if RX_RUNNER_OUT.search(t):
                outs += 1
                reasons.append("RUNNER_OUT")
                continue

            if RX_K.search(t) and RX_K_SAFE.search(t):
                continue

            if RX_K.search(t):
                outs += 1
                reasons.append("K")
                continue

            if RX_BATTER_OUT.search(t) and not RX_REACHED.search(t):
                outs += 1
                reasons.append("BATTER_OUT")
                continue

        has_p2 = bool(texts[1])
        has_p3 = bool(texts[2])
        has_p4 = bool(texts[3])

        fc_out = infer_outs_from_fc(texts[0], has_p2, has_p3, has_p4, outs)
        if fc_out:
            outs += fc_out
            reasons.append("FC_OUT")

        outs = min(outs, 3)
        return (outs, "+".join(reasons)) if outs else (0, "")

    combined = pd.DataFrame({"p1": p1_text, "p2": p2_text, "p3": p3_text, "p4": p4_text})
    results = combined.apply(_calc_outs, axis=1, result_type="expand")
    return results[0], results[1]


def metadata(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    df['half'] = np.where(df["home_text"].isna() | (df["home_text"] == ""), "Top", "Bottom")
    df['play_description'] = (df["away_text"].fillna("") + df["home_text"].fillna("")).str.strip()
    df['play_description'] = df['play_description'].replace("", pd.NA)
    df = df.dropna(subset=["play_description"])
    df['play_id'] = np.arange(1, len(df) + 1)

    return df


def outs_before(df: pd.DataFrame) -> pd.Series:
    keys = ["contest_id", "inning", "half"]
    return df.groupby(keys)["outs_on_play"].transform(lambda s: s.shift(fill_value=0).cumsum()).astype("int16")


def outs_after(df: pd.DataFrame) -> pd.Series:
    return (df["outs_before"] + df["outs_on_play"]).astype("int16")


def score_before(game_end_fl: pd.Series, runs_on_play: pd.Series, half: pd.Series, home_team: int) -> pd.Series:
    runs = pd.to_numeric(runs_on_play, errors="coerce").fillna(0).astype(int)
    is_top = pd.Series(half).eq("Top")

    if home_team == 1:
        scoring_mask = ~is_top
    else:
        scoring_mask = is_top

    scored_runs = runs.where(scoring_mask, 0)
    end_fl = pd.Series(game_end_fl).fillna(False).astype(bool)
    game_ids = end_fl.shift(fill_value=False).cumsum()
    cum_runs = scored_runs.groupby(game_ids, sort=False).cumsum()
    return cum_runs - scored_runs


def score_after(home_score_before: pd.Series, away_score_before: pd.Series, runs_on_play: pd.Series, half: pd.Series) -> tuple[pd.Series, pd.Series]:
    home_score_after = home_score_before + np.where(half == "Bottom", runs_on_play, 0)
    away_score_after = away_score_before + np.where(half == "Top", runs_on_play, 0)
    return home_score_after, away_score_after

def bat_order(df: pd.DataFrame) -> pd.Series:
    bat_order = bat_order_id(df)
    bat_order = bat_order_fill(df, bat_order)
    return bat_order

def runs_on_play(play_description: pd.Series) -> pd.Series:
    explicit_runs = (
        play_description.str.count("homered", flags=re.IGNORECASE) +
        play_description.str.count("homers", flags=re.IGNORECASE) +
        play_description.str.count("scored", flags=re.IGNORECASE) +
        play_description.str.count("scores", flags=re.IGNORECASE) +
        play_description.str.count("advanced to home", flags=re.IGNORECASE) +
        play_description.str.count("advances to home", flags=re.IGNORECASE) +
        play_description.str.count("steals home", flags=re.IGNORECASE) +
        play_description.str.count("stole home", flags=re.IGNORECASE) -
        play_description.str.count("scored, scored", flags=re.IGNORECASE)
    )

    rbi_count = (
        play_description
        .str.extract(r"(\d+)\s*RBI", flags=re.IGNORECASE)[0]
        .astype("float")
        .fillna(1)
    )

    has_rbi = play_description.str.contains(r"\bRBI\b", flags=re.IGNORECASE)

    return (
        explicit_runs.where(explicit_runs > 0, 0) +
        np.where((explicit_runs == 0) & has_rbi, rbi_count, 0)
    ).astype(int)


def runs_this_inn(end_inn: pd.Series, runs_on_play: pd.Series) -> pd.Series:
    m = len(end_inn)
    runs = np.zeros(m, dtype=int)
    endinnloc = [i for i, val in enumerate(end_inn) if val == 1]
    endinnloc = [-1] + endinnloc
    for j in range(1, len(endinnloc)):
        start = endinnloc[j-1] + 1
        end = endinnloc[j] + 1
        total = int(runs_on_play.iloc[start:end].sum())
        runs[start:end] = total
    return pd.Series(runs, index=end_inn.index)

def runs_rest_of_inn(end_inn: pd.Series, runs_on_play: pd.Series, runs_this_inn_s: pd.Series) -> pd.Series:
    m = len(end_inn)
    runs = np.zeros(m, dtype=int)
    endinnloc = [i for i, val in enumerate(end_inn) if val == 1]
    endinnloc = [-1] + endinnloc
    for j in range(1, len(endinnloc)):
        start = endinnloc[j-1] + 1
        end = endinnloc[j] + 1
        for k in range(start, end):
            runs[k] = int(runs_this_inn_s.iloc[k]) - int(runs_on_play.iloc[start:k+1].sum())
    runs = runs + runs_on_play.values
    return pd.Series(runs, index=end_inn.index)


def flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["contest_id", "play_id"]).copy()

    p = df["play_description"].map(split_players_text)
    df[["p1_text", "p2_text", "p3_text", "p4_text"]] = pd.DataFrame(p.tolist(), index=df.index)

    df["game_end_fl"] = False
    df["inn_end_fl"] = False
    df["new_game_fl"] = False
    df["new_inn_fl"] = False

    df.loc[df.groupby("contest_id", sort=False).tail(1).index, "game_end_fl"] = True
    df.loc[df.groupby(["contest_id", "inning", "half"], sort=False).tail(1).index, "inn_end_fl"] = True

    df.loc[df.groupby("contest_id", sort=False).head(1).index, "new_game_fl"] = True
    df.loc[df.groupby(["contest_id", "inning", "half"], sort=False).head(1).index, "new_inn_fl"] = True

    txt = df["play_description"].fillna("").astype(str)
    txt_norm = txt.str.replace(r"\s+", " ", regex=True).str.strip()

    rx_to_for = re.compile(
        r"(?i)^\s*(?:lineup changed:\s*)?"
        r"(?P<in>.+?)\s+to\s+(?P<pos>p|c|1b|2b|3b|ss|lf|cf|rf|dh|pitcher|catcher|first base|second base|third base|shortstop|left field|center field|right field|first baseman|second baseman|third baseman)\s+for\s+(?P<out>.+?)\s*$"
    )
    rx_in_for = re.compile(
        r"(?i)^\s*(?:lineup changed:\s*)?"
        r"(?P<in>.+?)\s+in\s+for\s+(?:(?P<pos>p|c|1b|2b|3b|ss|lf|cf|rf|dh|pitcher|catcher|first base|second base|third base|shortstop|left field|center field|right field|first baseman|second baseman|third baseman)\s+)?(?P<out>.+?)\s*$"
    )
    rx_pinch = re.compile(
        r"(?i)^\s*(?:lineup changed:\s*)?"
        r"(?P<in>.+?)\s+pinch\s+(?P<ptype>hit|ran)\s+for\s+(?P<out>.+?)\s*$"
    )

    m_to_for = txt_norm.str.extract(rx_to_for)
    m_in_for = txt_norm.str.extract(rx_in_for)
    m_pinch = txt_norm.str.extract(rx_pinch)

    has_to_for = m_to_for["in"].notna()
    has_in_for = (~has_to_for) & m_in_for["in"].notna()
    has_pinch = (~has_to_for) & (~has_in_for) & m_pinch["in"].notna()

    df["sub_fl"] = (has_to_for | has_in_for | has_pinch).astype(int)

    sub_in = np.where(
        has_to_for, m_to_for["in"],
        np.where(has_in_for, m_in_for["in"],
                 np.where(has_pinch, m_pinch["in"], ""))
    )
    sub_out = np.where(
        has_to_for, m_to_for["out"],
        np.where(has_in_for, m_in_for["out"],
                 np.where(has_pinch, m_pinch["out"], ""))
    )

    pinch_ptype = m_pinch.get("ptype")
    pinch_is_hit = pinch_ptype.notna() & pinch_ptype.astype(str).str.lower().eq("hit")

    sub_pos = np.where(
        has_to_for, m_to_for["pos"],
        np.where(has_in_for, m_in_for["pos"],
                 np.where(has_pinch, np.where(pinch_is_hit, "ph", "pr"), ""))
    )

    df["sub_in"] = pd.Series(sub_in, index=df.index).fillna("").astype(str).str.strip()
    df["sub_out"] = pd.Series(sub_out, index=df.index).fillna("").astype(str).str.strip()
    df["sub_pos"] = pd.Series(sub_pos, index=df.index).fillna("").astype(str).str.strip().map(canon_pos)

    p1 = df["p1_text"].fillna("").astype(str)
    df["int_bb_fl"] = txt_norm.str.contains("intentionally ", regex=False).astype(int)
    df["sh_fl"] = (p1.str.contains("SAC", regex=False) & ~p1.str.contains(r"(?:flied|popped)", regex=True)).astype(int)
    df["sf_fl"] = (
        (p1.str.contains("SAC", regex=False) & p1.str.contains(r"(?:flied|popped)", regex=True)) |
         (~p1.str.contains("SAC", regex=False) & p1.str.contains(r"(?:flied|popped)", regex=True) & p1.str.contains("RBI", regex=False))
    ).astype(int)

    df["top_inning_fl"] = df["half"].astype(str).eq("Top").astype(int)
    df["pitcher_sub_fl"] = df["sub_pos"].fillna("").astype(str).eq("p").astype(int)

    df.loc[df["sub_fl"] == 0, ["sub_in", "sub_out", "sub_pos"]] = ""

    return df


def determine_batter_and_runners(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    new_game = df["new_game_fl"].to_numpy(dtype=bool) if "new_game_fl" in df.columns else np.zeros(n, bool)
    new_inn  = df["new_inn_fl"].to_numpy(dtype=bool) if "new_inn_fl" in df.columns else np.zeros(n, bool)

    sub_fl = df["sub_fl"].to_numpy(dtype=np.int8) if "sub_fl" in df.columns else np.zeros(n, np.int8)
    sub_in  = df["sub_in"].fillna("").astype(str).to_numpy() if "sub_in" in df.columns else np.array([""] * n, dtype=object)
    sub_out = df["sub_out"].fillna("").astype(str).to_numpy() if "sub_out" in df.columns else np.array([""] * n, dtype=object)

    p1 = df["p1_text"].fillna("").astype(str).to_numpy()
    p2 = df["p2_text"].fillna("").astype(str).to_numpy() if "p2_text" in df.columns else np.array([""] * n, dtype=object)
    p3 = df["p3_text"].fillna("").astype(str).to_numpy() if "p3_text" in df.columns else np.array([""] * n, dtype=object)
    p4 = df["p4_text"].fillna("").astype(str).to_numpy() if "p4_text" in df.columns else np.array([""] * n, dtype=object)

    bat_out = np.empty(n, dtype=object)
    player_of_interest = np.empty(n, dtype=object)

    r1_before = np.empty(n, dtype=object)
    r2_before = np.empty(n, dtype=object)
    r3_before = np.empty(n, dtype=object)

    r1_after = np.empty(n, dtype=object)
    r2_after = np.empty(n, dtype=object)
    r3_after = np.empty(n, dtype=object)

    bases_before = np.empty(n, dtype=object)
    bases_after  = np.empty(n, dtype=object)

    def _norm(x):
        return x.strip() if isinstance(x, str) else ""

    def bases_str_local(a, b, c):
        return ("Y" if a else "N") + ("Y" if b else "N") + ("Y" if c else "N")

    r1 = r2 = r3 = ""

    for i in range(n):
        if new_game[i] or new_inn[i]:
            r1 = r2 = r3 = ""

        if sub_fl[i] == 1:
            si = _norm(sub_in[i])
            so = _norm(sub_out[i])
            if si and so:
                if _norm(r1) == so:
                    r1 = si
                if _norm(r2) == so:
                    r2 = si
                if _norm(r3) == so:
                    r3 = si

        r1_before[i], r2_before[i], r3_before[i] = r1, r2, r3
        bases_before[i] = bases_str_local(bool(_norm(r1)), bool(_norm(r2)), bool(_norm(r3)))

        p1i = p1[i].strip()
        subf = int(sub_fl[i])

        r1a, r2a, r3a = r1, r2, r3

        is_runner_event = is_runner_only_event(p1i)

        if is_runner_event:
            bat_out[i] = ""
            p1_runner = extract_runner_name_from_p1(p1i)
            player_of_interest[i] = p1_runner
        else:
            bat = extract_batter_name(p1i, subf)
            bat_out[i] = bat
            player_of_interest[i] = bat

        if blank_if_sub_or_meta(p1i, subf) and not is_runner_event:
            r1_after[i], r2_after[i], r3_after[i] = r1a, r2a, r3a
            bases_after[i] = bases_str_local(bool(_norm(r1a)), bool(_norm(r2a)), bool(_norm(r3a)))
            r1, r2, r3 = r1a, r2a, r3a
            continue

        def remove_runner(name):
            nonlocal r1a, r2a, r3a
            if _norm(r1a) == name:
                r1a = ""
            if _norm(r2a) == name:
                r2a = ""
            if _norm(r3a) == name:
                r3a = ""

        if is_runner_event and player_of_interest[i]:
            p1_dst = runner_dest(p1i)
            p1_runner_name = player_of_interest[i]
            if p1_dst == "OUT":
                remove_runner(p1_runner_name)
            elif p1_dst == "H":
                remove_runner(p1_runner_name)
            elif p1_dst == "2":
                remove_runner(p1_runner_name)
                r2a = p1_runner_name
            elif p1_dst == "3":
                remove_runner(p1_runner_name)
                r3a = p1_runner_name

        moves = []
        for px in (p2[i], p3[i], p4[i]):
            t = px.strip()
            if not t:
                continue
            nm = extract_runner_name(t)
            if not nm:
                continue
            dst = runner_dest(t)
            if dst:
                moves.append((nm, dst))

        for nm, dst in moves:
            if dst == "OUT" or dst == "H":
                remove_runner(nm)

        for nm, dst in moves:
            if dst == "2":
                remove_runner(nm)
                r2a = nm
            elif dst == "3":
                remove_runner(nm)
                r3a = nm

        if not is_runner_event:
            bdst = batter_dest(p1i)
            bat = bat_out[i]

            if bdst == "H":
                r1a = r2a = r3a = ""
            elif bdst == "2":
                if not _norm(r2a):
                    r2a = bat
            elif bdst == "3":
                if not _norm(r3a):
                    r3a = bat
            elif bdst == "1":
                if not _norm(r1a):
                    r1a = bat
                else:
                    if (not _norm(r2a)) and (not _norm(r3a)):
                        r2a = r1a
                        r1a = bat
                    elif (not _norm(r2a)) and _norm(r3a):
                        r2a = r1a
                        r1a = bat
                    elif (not _norm(r3a)) and _norm(r2a):
                        r3a = r2a
                        r2a = r1a
                        r1a = bat
                    else:
                        pass

        r1_after[i], r2_after[i], r3_after[i] = r1a, r2a, r3a
        bases_after[i] = bases_str_local(bool(_norm(r1a)), bool(_norm(r2a)), bool(_norm(r3a)))

        r1, r2, r3 = r1a, r2a, r3a

    df["batter_name"] = bat_out
    df["player_of_interest"] = player_of_interest

    df["r1_name"] = r1_before
    df["r2_name"] = r2_before
    df["r3_name"] = r3_before
    df["bases_before"] = bases_before

    df["r1_after"] = r1_after
    df["r2_after"] = r2_after
    df["r3_after"] = r3_after
    df["bases_after"] = bases_after

    return df

def classify_event_type(df: pd.DataFrame) -> pd.Series:
    text = df["play_description"].fillna("").astype(str)
    p1 = df["p1_text"].fillna("").astype(str)
    sub_fl = df["sub_fl"] if "sub_fl" in df.columns else pd.Series(0, index=df.index)

    def _classify(row_text: str, row_p1: str, row_sub: int) -> int:
        t = row_text.strip()
        p = row_p1.strip()

        if row_sub == 1:
            return EventType.NO_PLAY
        if p.startswith("("):
            return EventType.NO_PLAY
        if RX_LINEUP_CHANGE.search(t):
            return EventType.NO_PLAY
        if RX_NO_PLAY.search(t):
            return EventType.NO_PLAY

        if RX_HR.search(t):
            return EventType.HOME_RUN
        if RX_TRIPLE.search(t):
            return EventType.TRIPLE
        if RX_DOUBLE.search(t):
            return EventType.DOUBLE
        if RX_SINGLE.search(t):
            return EventType.SINGLE

        if RX_K_WP.search(t):
            return EventType.STRIKEOUT_WILD_PITCH
        if RX_K_PB.search(t):
            return EventType.STRIKEOUT_PASSED_BALL
        if RX_K.search(t) and not RX_K_SAFE.search(t):
            return EventType.STRIKEOUT
        if RX_K_SAFE.search(t):
            return EventType.STRIKEOUT

        if RX_IBB.search(t):
            return EventType.INTENTIONAL_WALK
        if RX_BB.search(t):
            return EventType.WALK
        if RX_HBP.search(t):
            return EventType.HIT_BY_PITCH

        if RX_DEFENSIVE_INDIFF.search(t):
            return EventType.DEFENSIVE_INDIFF
        if RX_STOLEN_BASE.search(t) and not RX_CAUGHT_STEALING.search(t):
            return EventType.STOLEN_BASE
        if RX_CAUGHT_STEALING.search(t):
            return EventType.CAUGHT_STEALING
        if RX_PICKOFF_ERROR.search(t):
            return EventType.PICKOFF_ERROR
        if RX_PICKOFF.search(t):
            return EventType.PICKOFF

        if RX_WILD_PITCH.search(t):
            return EventType.WILD_PITCH
        if RX_PASSED_BALL.search(t):
            return EventType.PASSED_BALL
        if RX_BALK.search(t):
            return EventType.BALK

        if RX_CI.search(t) or RX_INTERFERENCE.search(t):
            return EventType.INTERFERENCE
        if RX_DROPPED_FOUL.search(t):
            return EventType.FOUL_ERROR
        if RX_FC.search(t):
            return EventType.FIELDERS_CHOICE

        if RX_ERROR.search(t) and not RX_BATTER_OUT.search(t):
            return EventType.ERROR

        if RX_TP.search(t):
            return EventType.GENERIC_OUT
        if RX_DP.search(t):
            return EventType.GENERIC_OUT
        if RX_BATTER_OUT.search(t):
            return EventType.GENERIC_OUT
        if RX_RUNNER_OUT.search(t):
            return EventType.GENERIC_OUT

        if RX_ADVANCE.search(t):
            return EventType.OTHER_ADVANCE

        return EventType.UNKNOWN

    result = [
        _classify(t, p, s)
        for t, p, s in zip(text, p1, sub_fl, strict=True)
    ]

    return pd.Series(result, index=df.index, dtype="int16")


BATTED_BALL_EVENTS = {
    EventType.SINGLE,
    EventType.DOUBLE,
    EventType.TRIPLE,
    EventType.HOME_RUN,
    EventType.GENERIC_OUT,
    EventType.FIELDERS_CHOICE,
    EventType.ERROR,
}

_RX_BBTYPE = [
    (re.compile(r"\b(?:grounded|grounds|ground(?:ed)?\s+out|ground\s+ball)\b", re.I), BattedBallType.GROUND_BALL),
    (re.compile(r"\b(?:bunt(?:ed)?|sacrifice\s+bunt)\b", re.I), BattedBallType.BUNT),
    (re.compile(r"\b(?:lined|lines|lin(?:ed|es)\s+out|line\s+drive)\b", re.I), BattedBallType.LINE_DRIVE),
    (re.compile(r"\b(?:popped|pops|pop(?:ped)?\s+(?:out|up)|pop\s+up|infield\s+fly)\b", re.I), BattedBallType.POP_UP),
    (re.compile(r"\b(?:fouled\s+out|foul(?:ed|s)\s+out)\b", re.I), BattedBallType.POP_UP),
    (re.compile(r"\b(?:flied|flies|fli(?:ed|es)\s+out|fly\s+(?:out|ball)|flyout|home run|homers|)\b", re.I), BattedBallType.FLY_BALL),
    (re.compile(r"\b(?:sacrifice\s+fly)\b", re.I), BattedBallType.FLY_BALL),
]


def classify_batted_ball_type(df: pd.DataFrame) -> pd.Series:
    text = df["play_description"].fillna("").astype(str)
    event_type = df["event_type"] if "event_type" in df.columns else pd.Series(EventType.UNKNOWN, index=df.index)

    def _classify(row_text: str, row_event: int) -> str | None:
        if row_event not in BATTED_BALL_EVENTS:
            return None

        t = row_text.strip()
        for rx, bb_type in _RX_BBTYPE:
            if rx.search(t):
                return bb_type.value
        return None

    result = [
        _classify(t, e)
        for t, e in zip(text, event_type, strict=True)
    ]

    return pd.Series(result, index=df.index, dtype="object")


__all__ = [
    "bat_order",
    "classify_batted_ball_type",
    "classify_event_type",
    "determine_batter_and_runners",
    "flags",
    "metadata",
    "outs_after",
    "outs_before",
    "outs_on_play",
    "runs_on_play",
    "runs_rest_of_inn",
    "runs_this_inn",
    "score_after",
    "score_before",
]
