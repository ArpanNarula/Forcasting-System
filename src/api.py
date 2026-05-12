"""
api.py
------
FastAPI REST API exposing forecasts and model registry.

Run:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                          → health check
    GET  /states                    → list all trained states
    GET  /forecast/{state}          → 8-week forecast
    GET  /forecast/{state}?weeks=N  → N-week forecast
    GET  /models/{state}            → model comparison for a state
    GET  /models/summary            → best model for every state
    POST /retrain                   → trigger full retraining pipeline (async)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Sales Forecasting API",
    description="Time-series forecasting for US state-level weekly sales.",
    version="1.0.0",
)

REGISTRY_JSON = Path(__file__).parent.parent / "model_registry" / "registry.json"
FRONTEND_INDEX = Path(__file__).parent.parent / "frontend" / "index.html"
FORECAST_CACHE_TTL_SECONDS = 300
_FORECAST_CACHE: dict[tuple[str, int], tuple[float, dict]] = {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("forecasting_api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if not REGISTRY_JSON.exists():
        raise HTTPException(
            status_code=503,
            detail="Model registry not found. Run the training pipeline first.",
        )
    with open(REGISTRY_JSON) as f:
        return json.load(f)


def _normalize_state(state: str) -> str:
    """Case-insensitive + strip state lookup."""
    return state.strip().title()


def _available_states(registry: dict) -> list[str]:
    return sorted(registry.keys())


def _validate_state(state: str, registry: dict) -> str:
    normalized = _normalize_state(state)
    if normalized not in registry:
        states = _available_states(registry)
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"State '{normalized}' not found in trained registry.",
                "available_states": states,
            },
        )
    return normalized


def _cached_forecast(state: str, weeks: int) -> dict:
    cache_key = (state, weeks)
    now = time.time()
    cached = _FORECAST_CACHE.get(cache_key)
    if cached and now - cached[0] < FORECAST_CACHE_TTL_SECONDS:
        return cached[1]

    from src.model_selector import forecast

    result = forecast(state, n_periods=weeks)
    _FORECAST_CACHE[cache_key] = (now, result)
    return result


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "service": "Sales Forecasting API v1.0"}


@app.get("/dashboard", tags=["UI"], include_in_schema=False)
def dashboard():
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Dashboard file not found.")
    return FileResponse(FRONTEND_INDEX)


@app.get("/health/detailed", tags=["Health"])
def detailed_health():
    registry_exists = REGISTRY_JSON.exists()
    trained_states = 0
    last_modified = None

    if registry_exists:
        registry = _load_registry()
        trained_states = len(registry)
        last_modified = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(REGISTRY_JSON.stat().st_mtime),
        )

    return {
        "status": "ok" if registry_exists else "degraded",
        "service": "Sales Forecasting API v1.0",
        "registry_path": str(REGISTRY_JSON),
        "registry_exists": registry_exists,
        "trained_states": trained_states,
        "registry_last_modified_utc": last_modified,
        "forecast_cache_entries": len(_FORECAST_CACHE),
    }


@app.get("/scheduler/status", tags=["Admin"])
def get_scheduler_status():
    from src.scheduler import scheduler_status

    return scheduler_status()


@app.get("/states", tags=["Registry"])
def list_states():
    """Return all states with trained models."""
    registry = _load_registry()
    return {"states": sorted(registry.keys()), "count": len(registry)}


@app.get("/models/summary", tags=["Registry"])
def models_summary():
    """Return best model name + metrics for every state."""
    registry = _load_registry()
    summary = [
        {
            "state": state,
            "best_model": info["best_model"],
            "mape_pct": info["best_mape"],
            "rmse": info["best_rmse"],
        }
        for state, info in registry.items()
    ]
    summary.sort(key=lambda x: x["mape_pct"])
    return {"models": summary}


@app.get("/models/{state}", tags=["Registry"])
def model_detail(state: str):
    """Return model comparison table for a given state."""
    registry = _load_registry()
    state = _validate_state(state, registry)
    info = registry[state]
    return {
        "state": state,
        "best_model": info["best_model"],
        "comparison": info.get("comparison", {}),
        "val_dates": info.get("val_dates", []),
        "val_actuals": info.get("val_actuals", []),
    }


class BulkForecastRequest(BaseModel):
    states: list[str] = Field(..., min_length=1)
    weeks: int = Field(8, ge=1, le=52)


def _bulk_forecast_response(request: BulkForecastRequest) -> dict:
    registry = _load_registry()
    normalized_states = [_validate_state(state, registry) for state in request.states]

    forecasts = []
    errors = []
    for state in normalized_states:
        try:
            forecasts.append(_cached_forecast(state, request.weeks))
        except Exception as exc:
            errors.append({"state": state, "error": str(exc)})

    return {
        "n_states": len(normalized_states),
        "weeks": request.weeks,
        "forecasts": forecasts,
        "errors": errors,
    }


@app.get("/forecast/bulk", tags=["Forecast"])
def get_bulk_forecast(request: BulkForecastRequest = Body(...)):
    """
    Generate forecasts for multiple states in one call.

    Request body:
    {"states": ["California", "Texas"], "weeks": 8}
    """
    return _bulk_forecast_response(request)


@app.post("/forecast/bulk", tags=["Forecast"])
def post_bulk_forecast(request: BulkForecastRequest):
    """Browser-friendly alias for bulk forecast requests."""
    return _bulk_forecast_response(request)


@app.get("/forecast/{state}", tags=["Forecast"])
def get_forecast(state: str, weeks: int = 8):
    """
    Generate forecast for the specified state.

    - **state**: US state name (e.g. California)
    - **weeks**: number of weeks to forecast (default 8, max 52)
    """
    if weeks < 1 or weeks > 52:
        raise HTTPException(status_code=400, detail="weeks must be between 1 and 52.")

    registry = _load_registry()
    state = _validate_state(state, registry)

    try:
        result = _cached_forecast(state, weeks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

    return result


# ---------------------------------------------------------------------------
# Retraining endpoint (runs in background)
# ---------------------------------------------------------------------------

class RetrainRequest(BaseModel):
    states: Optional[list] = None   # None = retrain all


def _run_retrain(states: Optional[list]):
    """Background task: re-runs full pipeline."""
    from src.preprocessing import run_pipeline
    from src.model_selector import train_all_states

    state_data = run_pipeline()
    train_all_states(state_data, n_jobs=1, states_subset=states)
    _FORECAST_CACHE.clear()
    print("[retrain] Done.")


@app.post("/retrain", tags=["Admin"])
def trigger_retrain(request: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger retraining pipeline in the background.
    Optionally pass a list of state names to retrain only those.
    """
    background_tasks.add_task(_run_retrain, request.states)
    return {
        "status": "accepted",
        "message": "Retraining started in background.",
        "states": request.states or "all",
    }


# ---------------------------------------------------------------------------
# Entry point for direct run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
