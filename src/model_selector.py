"""
model_selector.py
-----------------
Orchestrates training of all 4 models per state,
compares MAPE on held-out validation, selects the best,
and persists the registry to disk.
"""

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.preprocessing import train_val_split, FORECAST_WEEKS
from src.models import sarima, prophet_model, xgboost_model, lstm_model

REGISTRY_DIR  = Path(__file__).parent.parent / "model_registry"
REGISTRY_JSON = REGISTRY_DIR / "registry.json"

REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# Model definitions: name → module
MODEL_MODULES = {
    "SARIMA":   sarima,
    "Prophet":  prophet_model,
    "XGBoost":  xgboost_model,
    "LSTM":     lstm_model,
}


# ---------------------------------------------------------------------------
# Single-state training
# ---------------------------------------------------------------------------

def train_one_state(state: str, state_df: pd.DataFrame, verbose: bool = True) -> dict:
    """Train all models for one state, evaluate, pick winner."""
    train_df, val_df = train_val_split(state_df)

    results = {}
    for name, module in MODEL_MODULES.items():
        if verbose:
            print(f"  [{state}] Training {name}...", flush=True)
        t0 = time.time()
        try:
            model_dict = module.train(train_df)
            metrics    = module.evaluate(model_dict, val_df)
            elapsed    = round(time.time() - t0, 1)
            metrics["train_time_sec"] = elapsed
            results[name] = {"model_dict": model_dict, "metrics": metrics}
            if verbose:
                print(f"    → MAPE={metrics['mape']:.2f}%  RMSE={metrics['rmse']:.0f}  ({elapsed}s)")
        except Exception as exc:
            if verbose:
                print(f"    ✗ {name} failed: {exc}")
            results[name] = None

    # Pick best by MAPE (lower is better)
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        raise RuntimeError(f"All models failed for state: {state}")

    best_name = min(valid, key=lambda k: valid[k]["metrics"]["mape"])
    best      = valid[best_name]

    # Save best model to disk
    model_path = REGISTRY_DIR / f"{state.replace(' ', '_')}_{best_name}.pkl"
    MODEL_MODULES[best_name].save(best["model_dict"], model_path)

    # Collect comparison table
    comparison = {
        name: {
            "mape": v["metrics"]["mape"],
            "rmse": v["metrics"]["rmse"],
            "train_time_sec": v["metrics"]["train_time_sec"],
        }
        for name, v in valid.items()
    }

    return {
        "state": state,
        "best_model": best_name,
        "best_mape": best["metrics"]["mape"],
        "best_rmse": best["metrics"]["rmse"],
        "model_path": str(model_path),
        "comparison": comparison,
        "val_predictions": best["metrics"]["predictions"],
        "val_actuals": val_df["Total"].tolist(),
        "val_dates": [str(d)[:10] for d in val_df["Date"].tolist()],
    }


# ---------------------------------------------------------------------------
# Full pipeline (all states, parallel)
# ---------------------------------------------------------------------------

def train_all_states(
    state_data: dict,
    n_jobs: int = 1,         # set > 1 to parallelise; LSTM needs 1 for GPU safety
    states_subset: Optional[list] = None,
) -> dict:
    """
    Train and evaluate models for all (or a subset of) states.
    Returns registry dict and saves to disk.
    """
    if states_subset:
        states = [state.strip().title() for state in states_subset]
    else:
        states = list(state_data.keys())
    print(f"\n=== Training {len(states)} states ===\n")

    if n_jobs == 1:
        registry = {}
        for state in states:
            try:
                result = train_one_state(state, state_data[state])
                registry[state] = result
                print(f"  ✓ {state} → best: {result['best_model']} (MAPE {result['best_mape']:.2f}%)\n")
            except Exception as exc:
                print(f"  ✗ {state} failed: {exc}\n")
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_one_state)(state, state_data[state], verbose=False)
            for state in states
        )
        registry = {r["state"]: r for r in results if r is not None}

    # Persist registry JSON (metrics and validation outputs, not model objects)
    registry_json = {
        state: info
        for state, info in registry.items()
    }
    with open(REGISTRY_JSON, "w") as f:
        json.dump(registry_json, f, indent=2)
    load_best_model.cache_clear()
    print(f"\nRegistry saved → {REGISTRY_JSON}")

    return registry


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def load_best_model(state: str) -> tuple:
    """
    Load the best model for a state from disk.
    Returns (model_dict, model_name).
    """
    with open(REGISTRY_JSON) as f:
        registry = json.load(f)

    if state not in registry:
        raise KeyError(f"State '{state}' not found in registry. Run training first.")

    entry      = registry[state]
    model_name = entry["best_model"]
    model_path = Path(entry["model_path"])
    module     = MODEL_MODULES[model_name]
    model_dict = module.load(model_path)
    return model_dict, model_name


def forecast(state: str, n_periods: int = FORECAST_WEEKS) -> dict:
    """Generate forecast for a state using its best model."""
    model_dict, model_name = load_best_model(state)
    module = MODEL_MODULES[model_name]
    preds  = module.predict(model_dict, n_periods=n_periods)

    last_date = pd.Timestamp(model_dict["last_train_date"])
    future_dates = [
        str((last_date + pd.Timedelta(weeks=i + 1)).date())
        for i in range(n_periods)
    ]

    return {
        "state": state,
        "model": model_name,
        "n_periods": n_periods,
        "forecast": [
            {"date": d, "predicted_sales": round(float(p), 2)}
            for d, p in zip(future_dates, preds)
        ],
    }


def get_registry_summary() -> list:
    """Return summary of all trained models."""
    with open(REGISTRY_JSON) as f:
        registry = json.load(f)
    return [
        {
            "state": state,
            "best_model": info["best_model"],
            "mape": info["best_mape"],
            "rmse": info["best_rmse"],
        }
        for state, info in registry.items()
    ]
