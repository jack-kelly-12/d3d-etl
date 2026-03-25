"""
Clean the `positions` column in cube_player_info.csv.

Run interactively in a Jupyter notebook to verify before saving:

    import importlib, scripts.clean_cube_positions as m
    df, cleaned = m.load_and_clean(PATH)
    cleaned["positions"].value_counts().head(40)   # inspect
    m.save(cleaned, PATH)                           # write when satisfied
"""

import re
from pathlib import Path

import pandas as pd

PATH = Path("/Users/jackkelly/Desktop/d3d-etl/data/cube_stats/cube_player_info.csv")

VALID_POSITIONS = {"P", "C", "1B", "2B", "3B", "SS", "OF", "IF", "UT", "DH", "LF", "CF", "RF"}


def _clean_position(raw: str | None) -> str | None:
    if not raw or str(raw).strip() in ("", "nan"):
        return None
    # Remove count suffixes like "(29)"
    s = re.sub(r"\(\d+\)", "", str(raw))
    # Split on dashes, dots, and whitespace
    tokens = re.split(r"[-.\s]+", s)
    seen: list[str] = []
    for t in tokens:
        t = t.strip().upper()
        if t in VALID_POSITIONS and t not in seen:
            seen.append(t)
    return ", ".join(seen) if seen else None


def load_and_clean(path: Path = PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (original_df, cleaned_df). Inspect before saving."""
    df = pd.read_csv(path, dtype=str)
    cleaned = df.copy()
    cleaned["positions"] = cleaned["positions"].apply(_clean_position)
    return df, cleaned


def diff(original: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Rows where positions changed — useful for spot-checking."""
    mask = original["positions"].fillna("") != cleaned["positions"].fillna("")
    return pd.concat(
        [
            original.loc[mask, ["cube_player_id", "player_name", "positions"]].rename(
                columns={"positions": "positions_before"}
            ),
            cleaned.loc[mask, ["positions"]].rename(columns={"positions": "positions_after"}),
        ],
        axis=1,
    )


def save(cleaned: pd.DataFrame, path: Path = PATH) -> None:
    cleaned.to_csv(path, index=False)
    print(f"Saved {len(cleaned)} rows to {path}")


if __name__ == "__main__":
    original, cleaned = load_and_clean()
    changes = diff(original, cleaned)
    print(f"{len(changes)} rows with position changes")
    print(changes.head(20).to_string(index=False))
    print("\nTop cleaned positions:")
    print(cleaned["positions"].value_counts().head(20).to_string())
