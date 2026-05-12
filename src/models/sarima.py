"""
models/sarima.py
----------------
SARIMA model using pmdarima's auto_arima for automatic order selection.
"""

import warnings
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

FORECAST_WEEKS = 8


def train(train_df: pd.DataFrame) -> dict:
    """
    Fit SARIMA on the training series.
    Returns a dict with model and metadata.
    """
    try:
        import pmdarima as pm
    except ImportError:
        raise ImportError("Install pmdarima: pip install pmdarima")

    series = train_df.set_index("Date")["Total"].asfreq("W")

    model = pm.auto_arima(
        series,
        seasonal=True,
        m=52,                  # weekly seasonality
        stepwise=True,
        approximation=True,    # faster for large m
        suppress_warnings=True,
        error_action="ignore",
        max_p=3, max_q=3,
        max_P=1, max_Q=1,
        D=1,
        information_criterion="aic",
        n_jobs=-1,
    )

    return {"model": model, "last_train_date": train_df["Date"].max()}


def predict(model_dict: dict, n_periods: int = FORECAST_WEEKS) -> pd.Series:
    """Generate n_periods ahead forecasts."""
    model = model_dict["model"]
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    return np.maximum(forecast, 0)   # sales can't be negative


def evaluate(model_dict: dict, val_df: pd.DataFrame) -> dict:
    """Compute MAPE and RMSE on validation set."""
    actuals = val_df["Total"].values
    preds   = predict(model_dict, n_periods=len(actuals))

    mape = np.mean(np.abs((actuals - preds) / np.where(actuals == 0, 1, actuals))) * 100
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))

    return {
        "model_name": "SARIMA",
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
