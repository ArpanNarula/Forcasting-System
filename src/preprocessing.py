"""
preprocessing.py
----------------
Loads raw Excel data, resamples to strict weekly frequency,
engineers all required features, and provides train/val splits.
"""

import pandas as pd
import numpy as np
import holidays
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "Forecasting_Case-_Study.xlsx"
US_HOLIDAYS = holidays.US()


# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)
    return df


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample each state's sales to strict weekly (Sunday-anchored) frequency.
    Gaps are forward-filled; remaining NaNs (leading) are back-filled.
    """
    records = []
    for state, grp in df.groupby("State"):
        grp = grp.set_index("Date")["Total"]
        grp = grp.resample("W").sum()          # sum within week if duplicates
        grp = grp.replace(0, np.nan)
        grp = grp.ffill().bfill()
        grp = grp.reset_index()
        grp.columns = ["Date", "Total"]
        grp["State"] = state
        records.append(grp)
    out = pd.concat(records, ignore_index=True)
    return out[["State", "Date", "Total"]]


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def _is_holiday_week(date: pd.Timestamp) -> int:
    """Return 1 if any day in the week ending on `date` is a US holiday."""
    week_days = pd.date_range(date - pd.Timedelta(days=6), date)
    return int(any(d.date() in US_HOLIDAYS for d in week_days))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag features, rolling stats, and calendar features.
    Operates on the full dataset; NaN rows introduced by lags are kept
    so callers can decide to drop them.
    """
    out = []
    for state, grp in df.groupby("State"):
        g = grp.sort_values("Date").copy()

        # --- lag features ---
        g["lag_1"]  = g["Total"].shift(1)
        g["lag_4"]  = g["Total"].shift(4)   # ~1 month
        g["lag_7"]  = g["Total"].shift(7)
        g["lag_30"] = g["Total"].shift(30)
        g["lag_52"] = g["Total"].shift(52)  # ~1 year

        # --- rolling statistics (min_periods avoids leading NaN cascade) ---
        g["roll_mean_4"]  = g["Total"].shift(1).rolling(4,  min_periods=2).mean()
        g["roll_mean_12"] = g["Total"].shift(1).rolling(12, min_periods=4).mean()
        g["roll_std_4"]   = g["Total"].shift(1).rolling(4,  min_periods=2).std()
        g["roll_std_12"]  = g["Total"].shift(1).rolling(12, min_periods=4).std()

        # --- calendar features ---
        g["day_of_week"]  = g["Date"].dt.dayofweek
        g["week_of_year"] = g["Date"].dt.isocalendar().week.astype(int)
        g["month"]        = g["Date"].dt.month
        g["quarter"]      = g["Date"].dt.quarter
        g["year"]         = g["Date"].dt.year

        # --- holiday flag ---
        g["holiday_flag"] = g["Date"].apply(_is_holiday_week)

        out.append(g)

    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Train / validation split (time-series safe — no leakage)
# ---------------------------------------------------------------------------

FORECAST_WEEKS = 8

def train_val_split(state_df: pd.DataFrame):
    """
    Split a single-state DataFrame into train and validation sets.
    Validation = last FORECAST_WEEKS rows.
    Returns (train_df, val_df).
    """
    state_df = state_df.sort_values("Date").reset_index(drop=True)
    split_idx = len(state_df) - FORECAST_WEEKS
    train = state_df.iloc[:split_idx].copy()
    val   = state_df.iloc[split_idx:].copy()
    return train, val


# ---------------------------------------------------------------------------
# 4. Convenience: return clean per-state dict
# ---------------------------------------------------------------------------

def get_state_data(df_featured: pd.DataFrame) -> dict:
    """
    Returns {state: df_sorted_with_features} for all states.
    Drops rows where lag_1 is NaN (insufficient history).
    """
    state_data = {}
    for state, grp in df_featured.groupby("State"):
        grp = grp.sort_values("Date").dropna(subset=["lag_1"]).reset_index(drop=True)
        state_data[state] = grp
    return state_data


# ---------------------------------------------------------------------------
# 5. Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline() -> dict:
    """End-to-end: load → resample → feature-engineer → split."""
    print("[preprocessing] Loading raw data...")
    raw = load_raw()

    print("[preprocessing] Resampling to weekly frequency...")
    weekly = resample_weekly(raw)

    print("[preprocessing] Engineering features...")
    featured = add_features(weekly)

    print(f"[preprocessing] Done. Shape: {featured.shape}")
    state_data = get_state_data(featured)
    print(f"[preprocessing] States available: {len(state_data)}")
    return state_data


if __name__ == "__main__":
    data = run_pipeline()
    sample_state = list(data.keys())[0]
    print(f"\nSample state: {sample_state}")
    print(data[sample_state].tail())
