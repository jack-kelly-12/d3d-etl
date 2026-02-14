import re

# =============================================================================
# BASE VERB PATTERNS (strings for composition, not compiled)
# =============================================================================

# Batter out verbs - accounts for past/present tense
_GROUNDED_OUT = r"ground(?:ed|s) out"
_FLIED_OUT = r"fli(?:ed|es) out"
_LINED_OUT = r"lin(?:ed|es) out"
_POPPED_OUT = r"pop(?:ped|s) (?:out|up)"
_FOULED_OUT = r"foul(?:ed|s) out"
_STRUCK_OUT = r"str(?:uck|ikes) out"
_INFIELD_FLY = r"infield fly"

# All batter out verbs combined
_BATTER_OUT_VERBS = (
    f"(?:{_GROUNDED_OUT}|{_FLIED_OUT}|{_LINED_OUT}|{_POPPED_OUT}|{_FOULED_OUT}|{_INFIELD_FLY})"
)

# Hit/reach verbs (past tense primary, but include present for some)
_SINGLED = r"singled"
_DOUBLED = r"doubled"
_TRIPLED = r"tripled"
_HOMERED = r"homer(?:ed|s)"
_HOME_RUN = r"home run"
_WALKED = r"walked"
_IBB = r"intentionally walked|was intentionally walked"
_HBP = r"hit by pitch"
_REACHED = r"reached"

# All batter reach verbs combined
_BATTER_REACH_VERBS = f"(?:{_SINGLED}|{_DOUBLED}|{_TRIPLED}|{_HOMERED}|{_WALKED}|{_HBP}|{_REACHED})"

# Runner verbs - accounts for past/present tense
_ADVANCED = r"advanc(?:ed|es)"
_STOLE = r"st(?:ole|eals)"
_SCORED = r"scor(?:ed|es)"
_PICKED_OFF = r"picked off"
_CAUGHT_STEALING = r"caught stealing"
_TAGGED_OUT = r"tagged out"
_OUT_AT_BASE = r"out at (?:first|second|third|home)"
_OUT_ON_PLAY = r"out on the play"

# All runner action verbs combined
_RUNNER_ACTION_VERBS = f"(?:{_ADVANCED}|{_STOLE}|{_SCORED}|{_PICKED_OFF}|{_CAUGHT_STEALING}|{_TAGGED_OUT}|{_OUT_AT_BASE})"

# Sacrifice
_SAC_FLY = r"sacrifice fly"
_SAC_BUNT = r"sacrific(?:e bunt|es|ed)"

# =============================================================================
# COMPILED REGEXES - Built from base patterns
# =============================================================================

# Multi-out plays
RX_TP = re.compile(r"\btriple play\b", re.I)
RX_DP = re.compile(r"\bdouble play\b", re.I)

# Strikeouts
RX_K = re.compile(rf"\b{_STRUCK_OUT}\b", re.I)
RX_K_SAFE = re.compile(
    rf"\b{_STRUCK_OUT}\b.*\b("
    r"reached first|reached base|reached on|safe at first|"
    r"wild pitch|passed ball|dropped 3rd strike|dropped third strike|"
    r"fielder'?s choice|error\(|\bE\d\b|bobble|advanced on"
    r")\b",
    re.I,
)
RX_K_WP = re.compile(rf"\b{_STRUCK_OUT}\b.*\bwild pitch\b", re.I)
RX_K_PB = re.compile(rf"\b{_STRUCK_OUT}\b.*\bpassed ball\b", re.I)

# Batter outs (non-K)
RX_BATTER_OUT = re.compile(rf"\b(?:{_SAC_FLY}|{_SAC_BUNT}|{_BATTER_OUT_VERBS})\b", re.I)
RX_GROUNDED_OUT = re.compile(rf"\b{_GROUNDED_OUT}\b", re.I)
RX_FLIED_OUT = re.compile(rf"\b{_FLIED_OUT}\b", re.I)
RX_LINED_OUT = re.compile(rf"\b{_LINED_OUT}\b", re.I)
RX_POPPED_OUT = re.compile(rf"\b{_POPPED_OUT}\b", re.I)
RX_FOULED_OUT = re.compile(rf"\b{_FOULED_OUT}\b", re.I)
RX_INFIELD_FLY = re.compile(rf"\b{_INFIELD_FLY}\b", re.I)

# Sacrifice
RX_SAC_FLY = re.compile(rf"\b{_SAC_FLY}\b", re.I)
RX_SAC_BUNT = re.compile(rf"\b{_SAC_BUNT}\b", re.I)

# Hits and reaches
RX_SINGLE = re.compile(rf"\b{_SINGLED}\b", re.I)
RX_DOUBLE = re.compile(rf"\b{_DOUBLED}\b", re.I)
RX_TRIPLE = re.compile(rf"\b{_TRIPLED}\b", re.I)
RX_HR = re.compile(rf"\b(?:{_HOMERED}|{_HOME_RUN})\b", re.I)
RX_BB = re.compile(rf"\b{_WALKED}\b", re.I)
RX_IBB = re.compile(rf"\b(?:{_IBB})\b", re.I)
RX_HBP = re.compile(rf"\b{_HBP}\b", re.I)
RX_REACHED = re.compile(rf"\b{_REACHED}\b", re.I)
RX_REACH = RX_REACHED  # alias

# Fielder's choice
RX_FC = re.compile(r"\bfielder'?s choice\b", re.I)

# Runner outs
RX_RUNNER_OUT = re.compile(
    rf"\b(?:{_OUT_AT_BASE}|{_PICKED_OFF}|{_CAUGHT_STEALING}|{_OUT_ON_PLAY})\b", re.I
)
RX_STOLEN_BASE = re.compile(rf"\b{_STOLE}\s+(?:second|third|home)\b", re.I)
RX_CAUGHT_STEALING = re.compile(
    rf"\b(?:{_CAUGHT_STEALING}|out at (?:second|third|home)\s+c\s+to)\b", re.I
)
RX_PICKOFF = re.compile(rf"\b{_PICKED_OFF}\b", re.I)
RX_PICKOFF_ERROR = re.compile(rf"\b{_PICKED_OFF}\b.*\b(?:error|E\d)\b", re.I)

# Runner advances
RX_ADVANCE = re.compile(rf"\b{_ADVANCED}\b", re.I)
RX_TO_2 = re.compile(rf"\b(?:{_ADVANCED}|{_STOLE})\s+to\s+second\b|\bstole second\b", re.I)
RX_TO_3 = re.compile(rf"\b(?:{_ADVANCED}|{_STOLE})\s+to\s+third\b|\bstole third\b", re.I)
RX_TO_H = re.compile(rf"\b(?:{_ADVANCED})\s+to\s+home\b|\bstole home\b|\b{_SCORED}\b", re.I)
RX_OUT = re.compile(
    rf"\b(?:{_OUT_AT_BASE}|out at second|out at third|out at home|{_PICKED_OFF}|{_CAUGHT_STEALING}|{_TAGGED_OUT})\b",
    re.I,
)

# Pitcher/catcher events
RX_WILD_PITCH = re.compile(r"\bwild pitch\b", re.I)
RX_PASSED_BALL = re.compile(r"\bpassed ball\b", re.I)
RX_BALK = re.compile(r"\bbalk\b", re.I)
RX_CI = re.compile(r"\bcatcher'?s? interference\b", re.I)

# Errors
RX_ERROR = re.compile(r"\b(?:error|muffed|dropped|bobbled|E\d)\b", re.I)
RX_DROPPED_FOUL = re.compile(r"\bdropped foul\b", re.I)
RX_INTERFERENCE = re.compile(r"\binterference\b", re.I)

# Defensive indifference
RX_DEFENSIVE_INDIFF = re.compile(r"\bdefensive indifference\b", re.I)

# Non-play events
RX_NO_PLAY = re.compile(
    r"\b(?:no play|halted|delay|postponed|ejected|suspended|coach visit|mound visit|"
    r"timeout|injury|review|challenged|overturned|confirmed|stands|sunny|rain|"
    r"hitting out of turn)\b",
    re.I,
)
RX_LINEUP_CHANGE = re.compile(r"^\s*(?:lineup changed|pinch (?:hit|ran)|to\s+\w+\s+for)\b", re.I)
RX_SUB_LINE = re.compile(
    r"^\s*(?:lineup changed:\s*)?.*?\b(?:in for|to\b.*\bfor\b|pinch (?:hit|ran) for)\b", re.I
)

# =============================================================================
# COMPOSITE PATTERNS - For classification and name extraction
# =============================================================================

# All batter action verbs (for detecting batter-focused plays)
RX_BATTER_VERBS = re.compile(
    rf"\b(?:{_BATTER_REACH_VERBS}|{_STRUCK_OUT}|{_BATTER_OUT_VERBS}|{_IBB}|fouled into double play|pinch hit)\b",
    re.I,
)

# All runner-only verbs (plays where p1 is about a runner, not the batter)
RX_RUNNER_ONLY_VERBS = re.compile(rf"\b{_RUNNER_ACTION_VERBS}\b", re.I)

# Combined verb for detecting any play action
RX_PLAY_VERB = re.compile(
    rf"\b(?:{_BATTER_REACH_VERBS}|{_STRUCK_OUT}|{_BATTER_OUT_VERBS}|{_RUNNER_ACTION_VERBS}|double play|triple play)\b",
    re.I,
)

# Batter out shorthand (for helpers - K or fielded out)
RX_BAT_OUT = re.compile(rf"\b(?:{_STRUCK_OUT}|{_BATTER_OUT_VERBS})\b", re.I)

# =============================================================================
# NAME EXTRACTION PATTERNS
# =============================================================================

# Extract batter name from p1_text
RX_BATTER_NAME = re.compile(
    rf"^\s*(?P<name>.+?)\s+(?:{_BATTER_REACH_VERBS}|{_STRUCK_OUT}|{_GROUNDED_OUT}|grounded|{_FLIED_OUT}|flied|{_LINED_OUT}|lined|{_POPPED_OUT}|popped|{_FOULED_OUT}|{_INFIELD_FLY}|out)\b",
    re.I,
)

# Extract runner name from p2/p3/p4 text
RX_RUNNER_NAME = re.compile(
    rf"^\s*(?P<name>.+?)\s+(?:{_ADVANCED}|{_STOLE}|{_SCORED}|out|{_PICKED_OFF}|{_CAUGHT_STEALING})\b",
    re.I,
)

# Extract runner name from p1_text (when p1 is a runner event)
RX_RUNNER_P1_NAME = re.compile(
    rf"^\s*(?P<name>.+?)\s+(?:{_ADVANCED}|{_STOLE}|{_SCORED}|out at|{_PICKED_OFF}|{_CAUGHT_STEALING}|{_TAGGED_OUT})\b",
    re.I,
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Multi-out plays
    "RX_TP",
    "RX_DP",
    # Strikeouts
    "RX_K",
    "RX_K_SAFE",
    "RX_K_WP",
    "RX_K_PB",
    # Batter outs
    "RX_BATTER_OUT",
    "RX_GROUNDED_OUT",
    "RX_FLIED_OUT",
    "RX_LINED_OUT",
    "RX_POPPED_OUT",
    "RX_FOULED_OUT",
    "RX_INFIELD_FLY",
    "RX_SAC_FLY",
    "RX_SAC_BUNT",
    # Hits/reaches
    "RX_SINGLE",
    "RX_DOUBLE",
    "RX_TRIPLE",
    "RX_HR",
    "RX_BB",
    "RX_IBB",
    "RX_HBP",
    "RX_REACHED",
    "RX_REACH",
    "RX_FC",
    # Runner events
    "RX_RUNNER_OUT",
    "RX_STOLEN_BASE",
    "RX_CAUGHT_STEALING",
    "RX_PICKOFF",
    "RX_PICKOFF_ERROR",
    "RX_ADVANCE",
    "RX_TO_2",
    "RX_TO_3",
    "RX_TO_H",
    "RX_OUT",
    # Pitcher/catcher
    "RX_WILD_PITCH",
    "RX_PASSED_BALL",
    "RX_BALK",
    "RX_CI",
    # Errors
    "RX_ERROR",
    "RX_DROPPED_FOUL",
    "RX_INTERFERENCE",
    # Other
    "RX_DEFENSIVE_INDIFF",
    "RX_NO_PLAY",
    "RX_LINEUP_CHANGE",
    "RX_SUB_LINE",
    # Composite
    "RX_BATTER_VERBS",
    "RX_RUNNER_ONLY_VERBS",
    "RX_PLAY_VERB",
    "RX_BAT_OUT",
    # Name extraction
    "RX_BATTER_NAME",
    "RX_RUNNER_NAME",
    "RX_RUNNER_P1_NAME",
]
