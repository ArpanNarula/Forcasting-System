"""
models/prophet_model.py
-----------------------
Facebook Prophet with weekly + yearly seasonality and US holidays.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

FORECAST_WEEKS = 8


def train(train_df: pd.DataFrame) -> dict:
    """Fit Prophet model."""
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Install prophet: pip install prophet")

    import holidays as hol

    # Build US holidays DataFrame for Prophet
    us_hols = hol.US(years=range(2018, 2026))
    holidays_df = pd.DataFrame(
        [(str(date), name) for date, name in us_hols.items()],
        columns=["ds", "holiday"],
    )
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])

    prophet_df = train_df[["Date", "Total"]].rename(columns={"Date": "ds", "Total": "y"})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
        holidays_prior_scale=10.0,
        interval_width=0.95,
    )
    model.fit(prophet_df)

    return {"model": model, "last_train_date": train_df["Date"].max()}


def predict(model_dict: dict, n_periods: int = FORECAST_WEEKS) -> np.ndarray:
    model = model_dict["model"]
    future = model.make_future_dataframe(periods=n_periods, freq="W")
    forecast = model.predict(future)
    preds = forecast.tail(n_periods)["yhat"].values
    return np.maximum(preds, 0)


def evaluate(model_dict: dict, val_df: pd.DataFrame) -> dict:
    actuals = val_df["Total"].values
    preds   = predict(model_dict, n_periods=len(actuals))

    mape = np.mean(np.abs((actuals - preds) / np.where(actuals == 0, 1, actuals))) * 100
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))

    return {
        "model_name": "Prophet",
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
