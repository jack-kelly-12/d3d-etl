"""
WAR calculation Pydantic models and output schemas.

Usage via Python::

    from processors.war_calculation.calculator import WARCalculator

    calculator = WARCalculator(
        batting_df=batting,
        pitching_df=pitching,
        pbp_df=pbp,
        guts_df=guts,
        park_factors_df=park_factors,
        lineups_df=lineups,
        rankings_df=rankings,
        mappings_df=mappings,
        division="ncaa_1",
        year=2025,
    )
    results = calculator.run()

    results.batting        # player batting WAR DataFrame
    results.pitching       # player pitching WAR DataFrame
    results.batting_team   # team batting stats DataFrame
    results.pitching_team  # team pitching stats DataFrame

Usage via CLI::

    python -m processors.get_war --data_dir data --year 2025
    python -m processors.get_war --data_dir data --year 2025 --divisions ncaa_1 ncaa_2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class GutsConstants(BaseModel):
    """Validated linear weights and run environment constants.

    Constructed from a guts DataFrame row via ``from_dataframe``.
    Raises ``ValidationError`` if any required constant is missing or
    non-numeric.
    """

    wbb: float
    whbp: float
    w1b: float
    w2b: float
    w3b: float
    whr: float
    woba: float
    woba_scale: float
    runs_win: float
    runs_out: float

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> GutsConstants:
        if df.empty:
            raise ValueError("Cannot create GutsConstants from empty DataFrame")
        row = df.iloc[0]
        return cls(**{k: float(row[k]) for k in cls.model_fields})


class _WARSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def _column_map(cls) -> dict[str, Any]:
        """Map of *DataFrame column name* -> default value."""
        result: dict[str, Any] = {}
        for name, info in cls.model_fields.items():
            col = info.alias if info.alias else name
            result[col] = info.default
        return result

    @classmethod
    def columns(cls) -> list[str]:
        """Ordered list of output column names."""
        return list(cls._column_map().keys())

    @classmethod
    def ensure_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add any missing schema columns with their default values."""
        for col, default in cls._column_map().items():
            if col not in df.columns:
                df[col] = default
        return df

    @classmethod
    def select(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* with only the columns defined in the schema, in order.

        Assumes ``ensure_columns`` has been called so all schema columns exist.
        """
        return df[cls.columns()]


# ---------------------------------------------------------------------------
# Input schemas — cube stats columns the pipeline expects at the start.
# These do NOT include PBP-derived columns (wgdp, bfh, webt, rea, …) which
# are added via merges during processing.
# ---------------------------------------------------------------------------

class BattingInputSchema(_WARSchema):

    # -- player bio --
    player_name: str = ""
    cube_player_id: int | None = None
    player_id: str = ""
    player_class: str = Field(default="", alias="class")
    bats: str = ""
    pos: str = ""

    # -- team --
    team_name: str = ""
    team_id: str = ""
    conference: str = ""

    # -- counting stats --
    gp: int = 0
    gs: int = 0
    ab: int = 0
    pa: int = 0
    h: int = 0
    singles: int = Field(default=0, alias="1b")
    doubles: int = Field(default=0, alias="2b")
    triples: int = Field(default=0, alias="3b")
    hr: int = 0
    r: int = 0
    rbi: int = 0
    bb: int = 0
    ibb: int = 0
    hbp: int = 0
    k: int = 0
    sf: int = 0
    sh: int = 0
    sb: int = 0
    cs: int = 0
    gdp: int = 0

    # -- rates from cube stats --
    ba: float = 0.0
    ob_pct: float = 0.0
    slg_pct: float = 0.0
    ops: float = 0.0
    iso: float = 0.0
    babip: float = 0.0
    bb_pct: float = 0.0
    k_pct: float = 0.0


class PitchingInputSchema(_WARSchema):

    # -- player bio --
    player_name: str = ""
    player_id: str = ""
    player_class: str = Field(default="", alias="class")
    throws: str = ""

    # -- team --
    team_name: str = ""
    team_id: str = ""
    conference: str = ""

    # -- record / counting stats --
    app: int = 0
    gs: int = 0
    w: int = 0
    losses: int = Field(default=0, alias="l")
    sv: int = 0
    ip: float = 0.0
    cg: int = 0
    sho: int = 0
    h: int = 0
    r: int = 0
    er: int = 0
    bb: int = 0
    so: int = 0
    hbp: int = 0
    hr_a: int = 0
    ibb: int = 0
    sfa: int = 0
    sha: int = 0
    pitches: int = 0

    # -- rates from cube stats --
    era: float = 0.0
    ra9: float = 0.0
    k9: float = 0.0
    bb9: float = 0.0
    h9: float = 0.0
    hr9: float = 0.0
    whip: float = 0.0


# ---------------------------------------------------------------------------
# Output schemas — full WAR output shape including computed + PBP columns.
# Applied at the end of the pipeline via ensure_columns + select.
# ---------------------------------------------------------------------------


class BattingWarSchema(_WARSchema):
    player_name: str = ""
    cube_player_id: int | None = None
    player_id: str = ""
    player_class: str = Field(default="", alias="class")
    bats: str = ""
    pos: str = ""

    team_name: str = ""
    team_id: str = ""
    conference: str = ""
    division: str = ""
    year: int = 0

    gp: int = 0
    gs: int = 0
    ab: int = 0
    pa: int = 0
    h: int = 0
    singles: int = Field(default=0, alias="1b")
    doubles: int = Field(default=0, alias="2b")
    triples: int = Field(default=0, alias="3b")
    hr: int = 0
    r: int = 0
    rbi: int = 0
    bb: int = 0
    ibb: int = 0
    hbp: int = 0
    k: int = 0
    sf: int = 0
    sh: int = 0
    sb: int = 0
    cs: int = 0

    ba: float = 0.0
    ob_pct: float = 0.0
    slg_pct: float = 0.0
    ops: float = 0.0
    iso: float = 0.0
    babip: float = 0.0
    ops_plus: float = 0.0
    bb_pct: float = 0.0
    k_pct: float = 0.0
    bb_per_k: float = 0.0
    sb_pct: float = 0.0
    runs_created: float = 0.0
    rc_per_pa: float = 0.0

    woba: float = 0.0
    wrc: float = 0.0
    wraa: float = 0.0
    wrc_plus: float = 0.0

    wsb: float = 0.0
    wgdp: float = 0.0
    gdp: int = 0
    gdp_opps: int = 0
    webt: float = 0.0
    ebt_opps: int = 0
    ebt: int = 0
    baserunning: float = 0.0
    bfh: int = 0

    batting: float = 0.0
    adjustment: float = 0.0
    league_adjustment: float = 0.0

    rea: float = 0.0
    wpa: float = 0.0
    wpa_li: float = 0.0
    clutch: float | None = None

    war: float = 0.0
    sos_adj_war: float = 0.0


class PitchingWarSchema(_WARSchema):

    player_name: str = ""
    player_id: str = ""
    player_class: str = Field(default="", alias="class")
    throws: str = ""
    ht: str = ""

    team_name: str = ""
    team_id: str = ""
    conference: str = ""
    division: str = ""
    year: int = 0

    app: int = 0
    gs: int = 0
    w: int = 0
    losses: int = Field(default=0, alias="l")
    sv: int = 0
    ip: float = 0.0
    ip_float: float = 0.0
    cg: int = 0
    sho: int = 0
    h: int = 0
    r: int = 0
    er: int = 0
    bb: int = 0
    so: int = 0
    hbp: int = 0
    bf: float = 0.0
    hr_a: int = 0
    ibb: int = 0
    sfa: int = 0
    sha: int = 0
    pitches: int = 0

    era: float = 0.0
    ra9: float = 0.0
    k9: float = 0.0
    bb9: float = 0.0
    h9: float = 0.0
    hr9: float = 0.0
    whip: float = 0.0
    k_pct: float = 0.0
    bb_pct: float = 0.0
    k_minus_bb_pct: float = 0.0
    hr_div_fb: float = 0.0
    babip_against: float = 0.0
    ba_against: float = 0.0
    obp_against: float = 0.0
    era_plus: float | None = None

    fo: int = 0
    go: int = 0
    fb: int = 0

    gmli: float = 0.0
    prea: float = 0.0
    pwpa: float = 0.0
    pwpa_li: float = 0.0
    clutch: float | None = None

    war: float = 0.0
    sos_adj_war: float = 0.0


batting_columns = BattingWarSchema.columns()
pitching_columns = PitchingWarSchema.columns()


@dataclass
class WarResults:
    batting: pd.DataFrame
    pitching: pd.DataFrame
    batting_team: pd.DataFrame
    pitching_team: pd.DataFrame
    sos_missing: list[str] = field(default_factory=list)
