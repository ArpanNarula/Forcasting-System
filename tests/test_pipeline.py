import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import preprocessing


def _synthetic_weekly_state(state="California", periods=130):
    dates = pd.date_range("2021-01-03", periods=periods, freq="W")
    trend = np.linspace(1000, 1600, periods)
    seasonal = 120 * np.sin(np.arange(periods) * 2 * np.pi / 52)
    total = trend + seasonal
    return pd.DataFrame({"State": state, "Date": dates, "Total": total})


def _featured_state(periods=130):
    raw = _synthetic_weekly_state(periods=periods)
    featured = preprocessing.add_features(raw)
    return preprocessing.get_state_data(featured)["California"]


def test_resample_weekly_has_no_missing_values():
    raw = pd.DataFrame(
        {
            "State": ["California", "California", "California"],
            "Date": pd.to_datetime(["2024-01-07", "2024-01-21", "2024-01-28"]),
            "Total": [100.0, np.nan, 130.0],
        }
    )
    weekly = preprocessing.resample_weekly(raw)
    assert weekly["Total"].isna().sum() == 0
    assert weekly["Date"].diff().dropna().eq(pd.Timedelta(days=7)).all()


def test_feature_engineering_lag_1_matches_shifted_total():
    state_df = _featured_state()
    assert "lag_1" in state_df.columns
    assert "lag_7" in state_df.columns
    assert "lag_30" in state_df.columns
    sample = state_df.iloc[10]
    previous = state_df.iloc[9]
    assert sample["lag_1"] == pytest.approx(previous["Total"])


def test_train_validation_split_has_no_date_leakage():
    state_df = _featured_state()
    train_df, val_df = preprocessing.train_val_split(state_df)
    assert train_df["Date"].max() < val_df["Date"].min()
    assert len(val_df) == preprocessing.FORECAST_WEEKS


@pytest.mark.parametrize(
    "module_name",
    [
        "src.models.sarima",
        "src.models.prophet_model",
        "src.models.xgboost_model",
        "src.models.lstm_model",
    ],
)
def test_model_train_predict_smoke(module_name, monkeypatch):
    module = pytest.importorskip(module_name)
    if module_name.endswith("sarima"):
        pytest.importorskip("pmdarima")
    if module_name.endswith("prophet_model"):
        pytest.importorskip("prophet")
    if module_name.endswith("xgboost_model"):
        try:
            pytest.importorskip("xgboost")
        except Exception as exc:
            pytest.skip(f"xgboost native runtime unavailable: {exc}")
    if module_name.endswith("lstm_model"):
        pytest.importorskip("tensorflow")
        monkeypatch.setattr(module, "EPOCHS", 1)
        monkeypatch.setattr(module, "BATCH", 4)

    state_df = _featured_state(periods=120)
    train_df, _ = preprocessing.train_val_split(state_df)
    model_dict = module.train(train_df)
    preds = module.predict(model_dict, n_periods=2)
    assert len(preds) == 2
    assert np.isfinite(preds).all()


def test_api_endpoints(monkeypatch, tmp_path):
    from src import api

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "California": {
                    "best_model": "XGBoost",
                    "best_mape": 5.0,
                    "best_rmse": 10.0,
                    "comparison": {"XGBoost": {"mape": 5.0, "rmse": 10.0}},
                    "val_dates": ["2024-01-07"],
                    "val_actuals": [100.0],
                    "val_predictions": [98.0],
                    "model_path": "dummy.pkl",
                }
            }
        )
    )

    monkeypatch.setattr(api, "REGISTRY_JSON", registry_path)
    monkeypatch.setattr(
        api,
        "_cached_forecast",
        lambda state, weeks: {
            "state": state,
            "model": "XGBoost",
            "n_periods": weeks,
            "forecast": [{"date": "2024-01-14", "predicted_sales": 101.0}],
        },
    )

    client = TestClient(api.app)
    assert client.get("/states").status_code == 200
    assert client.get("/dashboard").status_code == 200
    assert client.get("/forecast/California").status_code == 200
    assert client.get("/models/California").status_code == 200
    bulk = client.request(
        "GET",
        "/forecast/bulk",
        json={"states": ["California"], "weeks": 1},
    )
    assert bulk.status_code == 200
    post_bulk = client.post(
        "/forecast/bulk",
        json={"states": ["California"], "weeks": 1},
    )
    assert post_bulk.status_code == 200
