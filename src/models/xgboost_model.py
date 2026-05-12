"""
models/xgboost_model.py
-----------------------
XGBoost regressor trained on lag + rolling + calendar features.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

FORECAST_WEEKS = 8

FEATURE_COLS = [
    "lag_1", "lag_4", "lag_7", "lag_30", "lag_52",
    "roll_mean_4", "roll_mean_12",
    "roll_std_4", "roll_std_12",
    "day_of_week", "week_of_year", "month", "quarter", "year",
    "holiday_flag",
]


def _is_holiday_week(date: pd.Timestamp) -> int:
    import holidays

    us_holidays = holidays.US(years=[date.year])
    week_days = pd.date_range(date - pd.Timedelta(days=6), date)
    return int(any(d.date() in us_holidays for d in week_days))


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN in feature columns."""
    return df.dropna(subset=FEATURE_COLS).copy()


def train(train_df: pd.DataFrame) -> dict:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("Install xgboost: pip install xgboost")

    clean = _clean_features(train_df)
    X = clean[FEATURE_COLS].values
    y = clean["Total"].values

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)

    return {
        "model": model,
        "last_train_df": train_df,   # needed for recursive forecasting
        "last_train_date": train_df["Date"].max(),
    }


def _recursive_forecast(model_dict: dict, n_periods: int) -> np.ndarray:
    """
    Recursive multi-step forecast: each predicted value is fed back as lag.
    """
    from xgboost import XGBRegressor
    model: XGBRegressor = model_dict["model"]
    history = model_dict["last_train_df"].copy().sort_values("Date")

    preds = []
    for _ in range(n_periods):
        last = history.iloc[-1:]
        next_date = last["Date"].values[0] + pd.Timedelta(weeks=1)

        # build feature row
        vals = history["Total"].values
        lag1  = vals[-1]
        lag4  = vals[-4]  if len(vals) >= 4  else vals[0]
        lag7  = vals[-7]  if len(vals) >= 7  else vals[0]
        lag30 = vals[-30] if len(vals) >= 30 else vals[0]
        lag52 = vals[-52] if len(vals) >= 52 else vals[0]

        roll4  = pd.Series(vals[-4:]).mean()
        roll12 = pd.Series(vals[-12:]).mean()
        std4   = pd.Series(vals[-4:]).std() if len(vals) >= 4 else 0.0
        std12  = pd.Series(vals[-12:]).std() if len(vals) >= 12 else 0.0

        dt = pd.Timestamp(next_date)
        feat = np.array([[
            lag1, lag4, lag7, lag30, lag52,
            roll4, roll12, std4, std12,
            dt.dayofweek,
            dt.isocalendar()[1],  # week_of_year
            dt.month,
            (dt.month - 1) // 3 + 1,  # quarter
            dt.year,
            _is_holiday_week(dt),
        ]])

        pred = float(model.predict(feat)[0])
        pred = max(pred, 0)
        preds.append(pred)

        # append predicted row to history
        new_row = pd.DataFrame({"Date": [dt], "Total": [pred], "State": [history["State"].iloc[0]]})
        history = pd.concat([history, new_row], ignore_index=True)

    return np.array(preds)


def predict(model_dict: dict, n_periods: int = FORECAST_WEEKS) -> np.ndarray:
    return _recursive_forecast(model_dict, n_periods)


def evaluate(model_dict: dict, val_df: pd.DataFrame) -> dict:
    actuals = val_df["Total"].values
    preds   = predict(model_dict, n_periods=len(actuals))

    mape = np.mean(np.abs((actuals - preds) / np.where(actuals == 0, 1, actuals))) * 100
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))

    return {
        "model_name": "XGBoost",
        "mape": round(float(mape), 4),
        "rmse": round(float(rmse), 4),
        "predictions": preds.tolist(),
    }


def save(model_dict: dict, path: Path):
    with open(path, "wb") as f:
        pickle.dump(model_dict, f)


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
