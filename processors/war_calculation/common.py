import numpy as np
import pandas as pd


def safe_divide(num, denom, fill=0.0):
    return np.where(denom > 0, num / denom, fill)


def ip_to_float(ip_str) -> float:
    try:
        ip = str(ip_str)
        if '.' not in ip:
            return float(ip)
        whole, partial = ip.split('.')
        thirds = {0: 0, 1: 1/3, 2: 2/3}.get(int(partial), 0)
        return int(whole) + thirds
    except (ValueError, TypeError):
        return 0.0


def float_to_ip(decimal_ip: float) -> float:
    if pd.isna(decimal_ip) or decimal_ip < 0:
        return 0.0
    whole = int(decimal_ip)
    frac = decimal_ip - whole
    if frac < 0.17:
        partial = 0
    elif frac < 0.5:
        partial = 1
    elif frac < 0.84:
        partial = 2
    else:
        whole += 1
        partial = 0
    return float(f"{whole}.{partial}")


def aggregate_team(df: pd.DataFrame, sum_cols: list[str], first_cols: list[str] = None) -> pd.DataFrame:
    if first_cols is None:
        first_cols = ['conference', 'team_id']

    agg = {}
    for col in sum_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = 'sum'
    for col in first_cols:
        if col in df.columns:
            agg[col] = 'first'

    return df.groupby('team_name').agg(agg).reset_index()


def fill_missing(df: pd.DataFrame, cols: list[str], fill_value=0) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    return df
