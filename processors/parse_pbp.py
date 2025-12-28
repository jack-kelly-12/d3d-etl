import multiprocessing as mp
import re
from pathlib import Path

import numpy as np
import pandas as pd


def stripwhite(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def strip_punc(x):
    if pd.isna(x):
        return ""
    x = stripwhite(x)
    if x.endswith("."):
        x = re.sub(r"\.$", "", x)
    return x

def inn_end(top_inn: pd.Series) -> pd.Series:
    m = len(top_inn)
    out = np.zeros(m, dtype=int)
    for i in range(m - 1):
        out[i] = 1 if top_inn.iloc[i] != top_inn.iloc[i + 1] else 0
    out[-1] = 1
    return pd.Series(out, index=top_inn.index)

def game_end(contest_id: pd.Series) -> pd.Series:
    m = len(contest_id)
    out = np.zeros(m, dtype=int)
    for i in range(1, m):
        if contest_id.iloc[i] != contest_id.iloc[i - 1]:
            out[i - 1] = 1
    out[-1] = 1
    return pd.Series(out, index=contest_id.index)

def new_game(game_end_s: pd.Series) -> pd.Series:
    m = len(game_end_s)
    out = np.zeros(m, dtype=int)
    if m > 0:
        out[0] = 1
    for i in range(1, m):
        out[i] = int(game_end_s.iloc[i-1]) if not pd.isna(game_end_s.iloc[i-1]) else 0
    return pd.Series(out, index=game_end_s.index)

def new_inn(inn_end_s: pd.Series) -> pd.Series:
    m = len(inn_end_s)
    out = np.zeros(m, dtype=int)
    if m > 0:
        out[0] = 1
    for i in range(1, m):
        out[i] = int(inn_end_s.iloc[i-1]) if not pd.isna(inn_end_s.iloc[i-1]) else 0
    return pd.Series(out, index=inn_end_s.index)

def outs_before(outs_on_play: pd.Series, new_game_s: pd.Series, new_inn_s: pd.Series) -> pd.Series:
    m = len(outs_on_play)
    out = np.zeros(m, dtype=int)
    for i in range(1, m):
        if (
            not pd.isna(new_game_s.iloc[i]) and
            not pd.isna(new_inn_s.iloc[i]) and
            new_game_s.iloc[i] == 0 and
            new_inn_s.iloc[i] == 0
        ):
            out[i] = (out[i-1] + int(outs_on_play.iloc[i-1])) % 3
    return pd.Series(out, index=outs_on_play.index)

def score_before(new_game_s: pd.Series, runs_on_play_s: pd.Series, top_inning: pd.Series, home_team: int = 1) -> pd.Series:
    m = len(new_game_s)
    home_score_before = np.zeros(m, dtype=int)
    away_score_before = np.zeros(m, dtype=int)
    for i in range(1, m):
        if pd.isna(new_game_s.iloc[i]) or pd.isna(top_inning.iloc[i-1]) or pd.isna(runs_on_play_s.iloc[i-1]):
            home_score_before[i] = home_score_before[i-1]
            away_score_before[i] = away_score_before[i-1]
            continue
        if new_game_s.iloc[i] == 0:
            if top_inning.iloc[i-1] == 0:
                home_score_before[i] = home_score_before[i-1] + int(runs_on_play_s.iloc[i-1])
                away_score_before[i] = away_score_before[i-1]
            else:
                away_score_before[i] = away_score_before[i-1] + int(runs_on_play_s.iloc[i-1])
                home_score_before[i] = home_score_before[i-1]
        else:
            home_score_before[i] = 0
            away_score_before[i] = 0
    if home_team == 1:
        return pd.Series(home_score_before, index=new_game_s.index)
    else:
        return pd.Series(away_score_before, index=new_game_s.index)

def runs_play(home_score: pd.Series, away_score: pd.Series,
              home_score_before: pd.Series, away_score_before: pd.Series,
              top_inn: pd.Series) -> pd.Series:
    n = len(home_score)
    out = np.zeros(n, dtype=int)
    for i in range(1, n):
        if top_inn.iloc[i] == 0:
            out[i] = int(home_score.iloc[i]) - int(home_score_before.iloc[i])
        else:
            out[i] = int(away_score.iloc[i]) - int(away_score_before.iloc[i])
    return pd.Series(out, index=home_score.index)

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

def bat_order_id(new_game_s: pd.Series, top_inn: pd.Series, bat_name: pd.Series) -> pd.Series:
    m = len(top_inn)
    batorder = ["" for _ in range(m)]
    newgameloc = [i for i, val in enumerate(new_game_s) if val == 1] + [m]
    for j in range(1, len(newgameloc)):
        kk = 0
        jj = 0
        for i in range(newgameloc[j-1], newgameloc[j]):
            if top_inn.iloc[i] == 1 and stripwhite(bat_name.iloc[i]) != "":
                batorder[i] = (kk % 9) + 1
                kk += 1
            elif top_inn.iloc[i] == 0 and stripwhite(bat_name.iloc[i]) != "":
                batorder[i] = (jj % 9) + 1
                jj += 1
            else:
                batorder[i] = ""
    return pd.Series(batorder, index=top_inn.index)

def bat_order_fill(bat_order: pd.Series, end_game: pd.Series) -> pd.Series:
    bo = bat_order.copy()
    bo = bo.replace("", np.nan)
    m = len(bo)
    for i in range(m-1, 0, -1):
        if pd.isna(bo.iloc[i-1]) and end_game.iloc[i-1] == 0:
            bo.iloc[i-1] = bo.iloc[i]
    for i in range(1, m):
        if pd.isna(bo.iloc[i]):
            bo.iloc[i] = bo.iloc[i-1]
    return bo.fillna("").astype(str)

def determine_r1_name(bat_text, bat_name, r1_text, r1_name, inn_end_s, game_end_s, sub_in, sub_out):
    m = len(bat_text)
    out = ["" for _ in range(m)]
    for i in range(1, m):
        if pd.isna(inn_end_s.iloc[i-1]) or pd.isna(game_end_s.iloc[i-1]):
            out[i] = None
            continue
        if inn_end_s.iloc[i-1] == 0 and game_end_s.iloc[i-1] == 0:
            bt = str(bat_text.iloc[i-1])
            bn = str(bat_name.iloc[i-1])
            r1t = str(r1_text.iloc[i-1])
            r1n_prev = str(r1_name.iloc[i-1])
            si = str(sub_in.iloc[i-1])
            so = str(sub_out.iloc[i-1])
            if so != "" and so.strip() == stripwhite(r1n_prev):
                out[i] = si
            elif re.search(r"(singled|walked|hit by pitch|reached)", bt) and not re.search(r"(doubled|tripled|homered|advanced|scored|out|stole)", bt):
                out[i] = bn
            elif re.search(r"(reached first)", bt) and re.search(r"(struck out)", bt):
                out[i] = bn
            elif (r1t == "" or not re.search(r"(advanced to second|stole second|advanced to third|stole third|scored|out)", r1t)) and not re.search(r"(double play|advanced to second|stole second|advanced to third|stole third|scored|caught stealing|picked off|homered)", bt):
                out[i] = r1n_prev
            elif not re.search(r"(singled|doubled|tripled|advanced to second|stole second|advanced to third|stole third|scored|homered|out at second c to)", bt) and re.search(r"(advanced to third|stole third|scored|out at third)", r1t):
                base_r1t = re.sub(r"(advanced to second|stole second|stole third|advanced to third|scored|out).*$", "", r1t).strip()
                base_r1n = re.sub(r"(singled|reached).*$", "", r1n_prev).strip()
                if base_r1t != base_r1n:
                    out[i] = r1n_prev
            elif r1t == "" and re.sub(r"(advanced to second|stole second|stole third|advanced to third|scored|out|failed|Failed|picked off).*$", "", bt).strip() != r1n_prev.strip():
                out[i] = r1n_prev
            else:
                out[i] = None
        else:
            out[i] = None
    return pd.Series([stripwhite(x) if x is not None else None for x in out], index=bat_text.index)

def determine_r2_name(bat_text, bat_name, r1_text, r1_name, r2_text, r2_name_s, inn_end_s, game_end_s, sub_in, sub_out):
    m = len(bat_text)
    out = ["" for _ in range(m)]
    for i in range(1, m):
        if pd.isna(inn_end_s.iloc[i-1]) or pd.isna(game_end_s.iloc[i-1]):
            out[i] = None
            continue
        if inn_end_s.iloc[i-1] == 0 and game_end_s.iloc[i-1] == 0:
            bt = str(bat_text.iloc[i-1])
            r1t = str(r1_text.iloc[i-1])
            r2t = str(r2_text.iloc[i-1])
            r2n_prev = str(r2_name_s.iloc[i-1])
            si = str(sub_in.iloc[i-1])
            so = str(sub_out.iloc[i-1])
            if so != "" and so.strip() == stripwhite(r2n_prev):
                out[i] = si
            elif re.search(r"(doubled|advanced to second|stole second)", bt) and not re.search(r"(advanced to third|scored|out|stole third)", bt):
                out[i] = stripwhite(re.sub(r"(doubled|advanced to second|stole second).*$", "", bt))
            elif re.search(r"(advanced to second|stole second)", r1t) and not re.search(r"(advanced to third|scored|out|stole third)", r1t):
                out[i] = stripwhite(re.sub(r"(advanced to second|stole second).*$", "", r1t))
            elif r2t == "" and stripwhite(re.sub(r"(stole third|advanced to third|scored|out).*$", "", r1t)) != stripwhite(r2n_prev) and not re.search(r"(advanced to third|stole third|scored|picked off|caught stealing)", bt):
                out[i] = r2n_prev
            elif r2t == "" and stripwhite(re.sub(r"(out on the play).*$", "", r1t)) != stripwhite(r2n_prev) and re.search(r"(double play)", bt):
                out[i] = r2n_prev
            elif r1t == "" and not re.search(r"(stole third|advanced to third|scored|picked off|homered|caught stealing)", bt):
                out[i] = r2n_prev
            else:
                out[i] = None
        else:
            out[i] = None
    cleaned = []
    for val in out:
        if val is None:
            cleaned.append(None)
        else:
            cleaned.append(stripwhite(re.sub(r"(singled|reached).*$", "", val)))
    return pd.Series(cleaned, index=bat_text.index)

def determine_r3_name(bat_text, bat_name, r1_text, r1_name, r2_text, r2_name, r3_text, r3_name_s, inn_end_s, game_end_s, sub_in, sub_out):
    m = len(bat_text)
    out = ["" for _ in range(m)]
    for i in range(1, m):
        if pd.isna(inn_end_s.iloc[i-1]) or pd.isna(game_end_s.iloc[i-1]):
            out[i] = None
            continue
        if inn_end_s.iloc[i-1] == 0 and game_end_s.iloc[i-1] == 0:
            bt = str(bat_text.iloc[i-1])
            r1t = str(r1_text.iloc[i-1])
            r2t = str(r2_text.iloc[i-1])
            r3t = str(r3_text.iloc[i-1])
            r3n_prev = str(r3_name_s.iloc[i-1])
            si = str(sub_in.iloc[i-1])
            so = str(sub_out.iloc[i-1])
            if so != "" and so.strip() == stripwhite(r3n_prev):
                out[i] = si
            elif re.search(r"(tripled|advanced to third|stole third)", bt) and not re.search(r"(scored|out)", bt):
                out[i] = stripwhite(re.sub(r"(tripled|advanced to third|stole third).*$", "", bt))
            elif re.search(r"(advanced to third|stole third)", r1t) and not re.search(r"(scored|out)", r1t):
                out[i] = stripwhite(re.sub(r"(advanced to third|stole third).*$", "", r1t))
            elif re.search(r"(advanced to third|stole third)", r2t) and not re.search(r"(scored|out)", r2t):
                out[i] = stripwhite(re.sub(r"(advanced to third|stole third).*$", "", r2t))
            elif r1t == "" and not re.search(r"(scored|stole home|homered)", bt):
                out[i] = r3n_prev
            elif r2t == "" and stripwhite(re.sub(r"(scored|stole home|out).*$", "", r1t)) != stripwhite(r3n_prev) and not re.search(r"(scored|stole home)", bt):
                out[i] = r3n_prev
            elif r3t == "" and not re.search(r"(scored|stole home|out)", r2t) and not re.search(r"(scored|stole home|out)", r1t) and not re.search(r"(scored|stole home)", bt):
                out[i] = r3n_prev
            else:
                out[i] = None
        else:
            out[i] = None
    cleaned = []
    for val in out:
        if val is None:
            cleaned.append(None)
        else:
            cleaned.append(stripwhite(re.sub(r"(singled|doubled|reached|advanced|stole|failed|Failed|picked off).*$", "", val)))
    return pd.Series(cleaned, index=bat_text.index)

def classify_event_cd(df: pd.DataFrame) -> pd.Series:
    text = df["tmp_text"].fillna("").astype(str).str.lower()
    bat_text = df["bat_text"].fillna("").astype(str).str.lower()
    conds = []
    choices = []
    conds.append(df["sub_fl"] == 1); choices.append(1)
    conds.append(bat_text.str.startswith("(")); choices.append(1)
    conds.append(text.str.contains("hitting out of turn| for |no play|halted|delay|postponed|ejected|suspended|coach|sunny|review|challenged|hc|\\*\\*")); choices.append(1)
    conds.append(text.str.contains("struck out")); choices.append(3)
    conds.append(text.str.contains("stole")); choices.append(4)
    conds.append(text.str.contains("caught stealing|out at second c to|out at third c to") & ~text.str.contains("bunt|grounded")); choices.append(6)
    conds.append(text.str.contains("picked off")); choices.append(8)
    conds.append(text.str.contains("wild pitch")); choices.append(9)
    conds.append(text.str.contains("passed ball")); choices.append(10)
    conds.append(text.str.contains("balk")); choices.append(11)
    conds.append(text.str.contains("dropped foul")); choices.append(13)
    conds.append(text.str.contains("walked")); choices.append(14)
    conds.append(text.str.contains("hit by pitch")); choices.append(16)
    conds.append(text.str.contains("interference")); choices.append(17)
    conds.append(text.str.contains("error|muffed|dropped")); choices.append(18)
    conds.append(text.str.contains("fielder's choice")); choices.append(19)
    conds.append(text.str.contains("singled")); choices.append(20)
    conds.append(text.str.contains("doubled")); choices.append(21)
    conds.append(text.str.contains("tripled")); choices.append(22)
    conds.append(text.str.contains("homered")); choices.append(23)
    conds.append(text.str.contains("flied out|grounded out|popped|fouled out|lined out| infield fly|double play|triple play|out at first|out at second|out at third|out at home")); choices.append(2)
    conds.append(text.str.contains("advanced")); choices.append(12)
    return pd.Series(np.select(conds, choices, default=0), index=df.index)

def hit_type_from_text(bat_text: pd.Series, event_cd: pd.Series) -> pd.Series:
    bt = bat_text.fillna("").astype(str)
    out = []
    for i in range(len(bt)):
        t = bt.iloc[i]
        if event_cd.iloc[i] == 3:
            out.append("K")
        elif re.search(r"(bunt)", t):
            out.append("B")
        elif (not re.search(r"(bunt)", t)) and re.search(r"(SAC)", t) and (not re.search(r"(flied|popped)", t)):
            out.append("B")
        elif re.search(r"(grounded out|(?:p|3b|2b|ss|1b) to (?:p|3b|2b|ss|1b|c))", t):
            out.append("GO")
        elif re.search(r"(flied|fouled out to (?:lf|rf))", t):
            out.append("FO")
        elif re.search(r"(lined)", t):
            out.append("LO")
        elif re.search(r"(popped|infield fly|fouled out to (?:p|3b|2b|ss|1b|c))", t):
            out.append("PO")
        else:
            out.append("")
    return pd.Series(out, index=bat_text.index)

def parse_chunk(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["tmp_text"] = (d["away_text"].fillna("") + " " + d["home_text"].fillna("")).str.strip()

    d["sub_fl"] = np.where(
        (d["tmp_text"].str.contains(r"(?:singled|doubled|tripled|homered|walked|reached|struck out|grounded|flied|lined|popped| hit|infield fly|out|double play|triple play)")) &
        (~d["tmp_text"].str.contains("pinch hit")),
        0,
        np.where(
            d["tmp_text"].str.contains(r"(?:to (?:p|c|1b|2b|3b|ss|lf|rf|cf|dh))|pinch hit|pinch ran"),
            1,
            0
        )
    )

    d["bat_text"] = d["tmp_text"].str.replace(r"(;|3a|:).*$", "", regex=True)
    d["r1_text"] = np.where(
        d["tmp_text"].str.contains(r"(?:;|3a|:)"),
        d["tmp_text"].str.replace(r"^.*?(?:;|3a|:)", "", regex=True).str.strip(),
        ""
    )
    d["r2_text"] = np.where(
        d["r1_text"].str.contains(r"(?:;|3a|:)"),
        d["r1_text"].str.replace(r"^.*?(?:;|3a|:)", "", regex=True).str.strip(),
        ""
    )
    d["r3_text"] = np.where(
        d["r2_text"].str.contains(r"(?:;|3a|:)"),
        d["r2_text"].str.replace(r"^.*?(?:;|3a|:)", "", regex=True).str.strip(),
        ""
    )
    d["r1_text"] = d["r1_text"].str.replace(r"(;|3a|:).*$", "", regex=True).str.strip()
    d["r2_text"] = d["r2_text"].str.replace(r"(;|3a|:).*$", "", regex=True).str.strip()

    d["event_cd"] = classify_event_cd(d)

    bt = d["bat_text"].fillna("")
    r1t = d["r1_text"].fillna("")
    d["bat_name"] = np.select(
        [
            d["event_cd"].isin([0, 1]),
            bt.str.contains(r"(?:Batter|Runner's interference)"),
            (~bt.str.contains(r"(?:walked|singled|doubled|tripled|reached|struck out|grounded out)")) &
            (bt.str.contains(r"(?:advanced|caught stealing|stole|picked off|out at (?:first|second|third|home)|tagged out)")),
            bt.str.contains(r"(?:singled|doubled|tripled|homered|walked|reached|struck out|grounded|flied|lined|popped|hit | out |fouled out|pinch hit|infield fly|intentionally walked|was intentionally walked|fouled into double play)"),
            r1t.str.contains(r"caught stealing  c to (?:2b|3b), double play\.")
        ],
        [
            "",
            "",
            "",
            bt.str.replace(r"((?:singled|doubled|tripled|homered|walked|reached|struck out|grounded|flied|lined|popped|hit | out |fouled out|pinch hit|infield fly|intentionally walked|was intentionally walked|fouled into double play).*$)", "", regex=True),
            d["bat_text"]
        ],
        default=""
    )

    d["sub_in"] = np.select(
        [
            (d["sub_fl"] == 1) & bt.str.contains(r"to (?:p|c|1b|2b|3b|ss|lf|rf|cf|dh)"),
            (d["sub_fl"] == 1) & bt.str.contains("pinch ran for"),
            (d["sub_fl"] == 1) & bt.str.contains("pinch hit for")
        ],
        [
            bt.str.replace(r"(to (?:p|c|1b|2b|3b|ss|lf|rf|cf|dh).*$)", "", regex=True).str.strip(),
            bt.str.replace(r"pinch ran for.*$", "", regex=True).str.strip(),
            bt.str.replace(r"pinch hit for.*$", "", regex=True).str.strip()
        ],
        default=""
    )
    d["sub_out"] = pd.Series(np.select(
        [
            (d["sub_fl"] == 1) & bt.str.contains(r"to (?:p|c|1b|2b|3b|ss|lf|rf|cf|dh) for"),
            (d["sub_fl"] == 1) & bt.str.contains("pinch ran for"),
            (d["sub_fl"] == 1) & bt.str.contains("pinch hit")
        ],
        [
            bt.str.replace(r"^.*to (?:p|c|1b|2b|3b|ss|lf|rf|cf|dh) for", "", regex=True),
            bt.str.replace(r"^.*pinch ran for", "", regex=True),
            bt.str.replace(r"^.*pinch hit for", "", regex=True)
        ],
        default=""
    )).map(strip_punc)

    d["game_end_flag"] = game_end(d["contest_id"])
    d["new_game"] = new_game(d["game_end_flag"])
    d["top_inning"] = np.where(d["away_text"].fillna("") == "", 0, 1)
    d["inn_end_flag"] = inn_end(d["top_inning"])

    d["r1_name"] = determine_r1_name(d["bat_text"], d["bat_name"], d["r1_text"], d.get("r1_name", pd.Series([""] * len(d))), d["inn_end_flag"], d["game_end_flag"], d["sub_in"], d["sub_out"])
    d["r2_name"] = determine_r2_name(d["bat_text"], d["bat_name"], d["r1_text"], d["r1_name"], d["r2_text"], d.get("r2_name", pd.Series([""] * len(d))), d["inn_end_flag"], d["game_end_flag"], d["sub_in"], d["sub_out"])
    d["r3_name"] = determine_r3_name(d["bat_text"], d["bat_name"], d["r1_text"], d["r1_name"], d["r2_text"], d["r2_name"], d["r3_text"], d.get("r3_name", pd.Series([""] * len(d))), d["inn_end_flag"], d["game_end_flag"], d["sub_in"], d["sub_out"])
    for col in ["r1_name", "r2_name", "r3_name"]:
        d[col] = d[col].fillna("").map(stripwhite)

    for runner in ["r1_name", "r2_name", "r3_name"]:
        mask = (d["bat_name"] != "") & (d["bat_name"].map(stripwhite) == d[runner].map(stripwhite))
        d.loc[mask, "bat_name"] = ""

    d["outs_on_play"] = np.select(
        [
            d["event_cd"].isin([0, 1]),
            d["bat_text"].str.contains("triple play"),
            d["bat_text"].str.contains("double play"),
            (d["bat_text"].str.contains(r"(?: out|popped)")) & (d["bat_text"].str.contains("reached"))
        ],
        [0, 3, 2, 0],
        default=np.nan
    )

    cond_1out = (
        ((d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (~d["r1_text"].str.contains(r"(?: out |popped)")) & (~d["r2_text"].str.contains(r"(?: out |popped)")) & (~d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((~d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (d["r1_text"].str.contains(r"(?: out |popped)")) & (~d["r2_text"].str.contains(r"(?: out |popped)")) & (~d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((~d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (~d["r1_text"].str.contains(r"(?: out |popped)")) & (d["r2_text"].str.contains(r"(?: out |popped)")) & (~d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((~d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (~d["r1_text"].str.contains(r"(?: out |popped)")) & (~d["r2_text"].str.contains(r"(?: out |popped)")) & (d["r3_text"].str.contains(r"(?: out |popped)")))
    )
    cond_2out = (
        ((d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (d["r1_text"].str.contains(r"(?: out |popped)")) & (~d["r2_text"].str.contains(r"(?: out |popped)")) & (~d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (~d["r1_text"].str.contains(r"(?: out |popped)")) & (d["r2_text"].str.contains(r"(?: out |popped)")) & (~d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (~d["r1_text"].str.contains(r"(?: out |popped)")) & (~d["r2_text"].str.contains(r"(?: out |popped)")) & (d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((~d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (d["r1_text"].str.contains(r"(?: out |popped)")) & (d["r2_text"].str.contains(r"(?: out |popped)")) & (~d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((~d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (d["r1_text"].str.contains(r"(?: out |popped)")) & (~d["r2_text"].str.contains(r"(?: out |popped)")) & (d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((~d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (~d["r1_text"].str.contains(r"(?: out |popped)")) & (d["r2_text"].str.contains(r"(?: out |popped)")) & (d["r3_text"].str.contains(r"(?: out |popped)")))
    )
    cond_3out = (
        ((d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (d["r1_text"].str.contains(r"(?: out |popped)")) & (d["r2_text"].str.contains(r"(?: out |popped)")) & (~d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (~d["r1_text"].str.contains(r"(?: out |popped)")) & (d["r2_text"].str.contains(r"(?: out |popped)")) & (d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((d["bat_text"].str.contains(r"(?: out |popped|infield fly)")) & (d["r1_text"].str.contains(r"(?: out |popped)")) & (~d["r2_text"].str.contains(r"(?: out |popped)")) & (d["r3_text"].str.contains(r"(?: out |popped)"))) |
        ((~d["bat_text"].str.contains(r"(?: out|popped)")) & (d["r1_text"].str.contains(r"(?: out |popped)")) & (d["r2_text"].str.contains(r"(?: out |popped)")) & (d["r3_text"].str.contains(r"(?: out |popped)")))
    )

    d.loc[cond_1out & d["outs_on_play"].isna(), "outs_on_play"] = 1
    d.loc[cond_2out & d["outs_on_play"].isna(), "outs_on_play"] = 2
    d.loc[cond_3out & d["outs_on_play"].isna(), "outs_on_play"] = 3
    d["outs_on_play"] = d["outs_on_play"].fillna(0).astype(int)

    d["new_inn"] = new_inn(d["inn_end_flag"])
    d["outs_before"] = outs_before(d["outs_on_play"], d["new_game"], d["new_inn"])
    d["outs_after"] = d["outs_before"] + d["outs_on_play"]
    d["base_cd_before"] = np.select(
        [
            (d["r1_name"].str.strip() != "") & (d["r2_name"] == "") & (d["r3_name"] == ""),
            (d["r1_name"] == "") & (d["r2_name"].str.strip() != "") & (d["r3_name"] == ""),
            (d["r1_name"].str.strip() != "") & (d["r2_name"].str.strip() != "") & (d["r3_name"] == ""),
            (d["r1_name"] == "") & (d["r2_name"] == "") & (d["r3_name"].str.strip() != ""),
            (d["r1_name"].str.strip() != "") & (d["r2_name"] == "") & (d["r3_name"].str.strip() != ""),
            (d["r1_name"] == "") & (d["r2_name"].str.strip() != "") & (d["r3_name"].str.strip() != ""),
            (d["r1_name"].str.strip() != "") & (d["r2_name"].str.strip() != "") & (d["r3_name"].str.strip() != "")
        ],
        [1, 2, 3, 4, 5, 6, 7],
        default=0
    )

    d["bat_order"] = bat_order_id(d["new_game"], d["top_inning"], d["bat_name"])
    d["hit_type"] = hit_type_from_text(d["bat_text"], d["event_cd"])
    d["runs_on_play"] = (
        d["tmp_text"].str.count("advanced to home") +
        d["tmp_text"].str.count("scored") +
        d["tmp_text"].str.count("homered") +
        d["tmp_text"].str.count("stole home") -
        d["tmp_text"].str.count("scored, scored")
    ).astype(int)
    d["away_score_before"] = score_before(d["new_game"], d["runs_on_play"], d["top_inning"], home_team=0)
    d["home_score_before"] = score_before(d["new_game"], d["runs_on_play"], d["top_inning"], home_team=1)
    d["away_score"] = d["away_score_before"]
    d["home_score"] = d["home_score_before"]
    d["away_score_after"] = np.where(d["top_inning"] == 1, d["away_score_before"] + d["runs_on_play"], d["away_score_before"])
    d["home_score_after"] = np.where(d["top_inning"] == 0, d["home_score_before"] + d["runs_on_play"], d["home_score_before"])
    d["runs_this_inn"] = runs_this_inn(d["inn_end_flag"], d["runs_on_play"])
    d["runs_roi"] = runs_rest_of_inn(d["inn_end_flag"], d["runs_on_play"], d["runs_this_inn"])

    d["int_bb_fl"] = np.where(d["tmp_text"].str.contains("intentionally "), 1, 0)
    d["sh_fl"] = np.where((d["bat_text"].str.contains("SAC")) & (~d["bat_text"].str.contains(r"(?:flied|popped)")), 1, 0)
    d["sf_fl"] = np.where(
        ((d["bat_text"].str.contains("SAC")) & d["bat_text"].str.contains(r"(?:flied|popped)")) |
        ((~d["bat_text"].str.contains("SAC")) & d["bat_text"].str.contains(r"(?:flied|popped)") & d["bat_text"].str.contains("RBI")),
        1, 0
    )

    d["bat_order"] = bat_order_fill(d["bat_order"], d["game_end_flag"])

    cols_front = [
        "date","contest_id","play_id","away_team","home_team","inning","top_inning","away_score","home_score",
        "bat_order","bat_name","r1_name","r2_name","r3_name","sub_fl","sub_in","sub_out","base_cd_before",
        "outs_before","event_cd","hit_type","outs_on_play","outs_after","runs_this_inn","runs_roi",
        "runs_on_play","away_score_after","home_score_after","new_inn","inn_end_flag","new_game","game_end_flag"
    ]
    keep = [c for c in cols_front if c in d.columns] + [
        c for c in d.columns if c not in cols_front + ["tmp_text","bat_text","r1_text","r2_text","r3_text"]
    ]
    return d[keep]

def ncaa_parse_parallel(pbp_df: pd.DataFrame, num_cores: int = None) -> pd.DataFrame:
    if num_cores is None:
        num_cores = max(1, mp.cpu_count() - 1)
    chunks = np.array_split(pbp_df, num_cores)
    with mp.Pool(num_cores) as pool:
        results = pool.map(parse_chunk, chunks)
    return pd.concat(results, ignore_index=True)

def pick_right(df, base):
    for c in (base, f"{base}_sched", f"{base}_y"):
        if c in df.columns:
            return df[c]
    return pd.Series([pd.NA]*len(df), index=df.index)

def pick_left(df, base):
    for c in (base, f"{base}_pbp", f"{base}_x"):
        if c in df.columns:
            return df[c]
    return pd.Series([pd.NA]*len(df), index=df.index)

def convert_to_legacy_cols(pbp: pd.DataFrame, sched: pd.DataFrame):
    away_flag = sched["away"].astype(str).str.lower().isin(["1","true","t","yes","y"])

    sched["away_team"] = np.where(away_flag, sched["team_name"], sched["opponent"])
    sched["home_team"] = np.where(away_flag, sched["opponent"], sched["team_name"])

    sched["away_team_id"] = np.where(away_flag, sched["team_id"], sched["opponent_team_id"])
    sched["home_team_id"] = np.where(away_flag, sched["opponent_team_id"], sched["team_id"])

    merged = pbp.merge(
        sched[[
            "contest_id","year","division","date",
            "away_team","home_team","away_team_id","home_team_id"
        ]],
        on="contest_id",
        how="left",
        suffixes=("_pbp","_sched")
    )

    return pd.DataFrame({
        "year": pick_right(merged, "year"),
        "date": pick_right(merged, "date"),
        "contest_id": pick_left(merged, "contest_id"),
        "inning": pick_left(merged, "inning"),
        "away_team": pick_right(merged, "away_team"),
        "home_team": pick_right(merged, "home_team"),
        "away_team_id": pick_right(merged, "away_team_id"),
        "home_team_id": pick_right(merged, "home_team_id"),
        "away_score": pick_left(merged, "away_score"),
        "home_score": pick_left(merged, "home_score"),
        "away_text": pick_left(merged, "away_text"),
        "home_text": pick_left(merged, "home_text"),
        "division": pick_right(merged, "division")
    })

def main(data_dir: str, year: int, divisions: list = None):
    if divisions is None:
        divisions = [1, 2, 3]
    for division in divisions:
        div_name = f"d{division}"
        input_path = Path(data_dir) / f"pbp/{div_name}_pbp_{year}.csv"
        sched_path = Path(data_dir) / f"schedules/{div_name}_schedules_{year}.csv"
        if not input_path.exists():
            print(f"PBP file not found: {input_path}, skipping division {division}")
            continue
        pbp = pd.read_csv(input_path)
        sched = pd.read_csv(sched_path)
        df = convert_to_legacy_cols(pbp, sched)
        df["play_id"] = np.arange(1, len(df)+1)
        parsed = ncaa_parse_parallel(df)
        if parsed.empty:
            print(f"No play-by-play processed for division {division}")
            continue
        output_path = Path(data_dir) / f"pbp/{div_name}_parsed_pbp_{year}.csv"
        parsed.to_csv(output_path, index=False)
        print(f"Saved {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--divisions', nargs='+', type=int, default=[1, 2, 3],
                        help='Divisions to process (default: 1 2 3)')
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
