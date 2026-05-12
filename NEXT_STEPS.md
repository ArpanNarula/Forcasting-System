# NEXT_STEPS.md — Completion Notes

## Context

This Sales Forecasting System has been completed against the continuation
requirements. The original remaining tasks are kept below for traceability.
The codebase is structured as follows:

```
forecasting_system/
├── data/Forecasting_Case-_Study.xlsx
├── src/
│   ├── preprocessing.py          ✅ Done
│   ├── model_selector.py         ✅ Done
│   ├── api.py                    ✅ Done (skeleton)
│   └── models/
│       ├── sarima.py             ✅ Done
│       ├── prophet_model.py      ✅ Done
│       ├── xgboost_model.py      ✅ Done
│       └── lstm_model.py         ✅ Done
├── notebooks/EDA.ipynb           ✅ Done
├── train.py                      ✅ Done
└── requirements.txt              ✅ Done
```

## Completed Work

- API hardening: state validation, bulk forecast, response cache, detailed health check, request logging middleware
- Evaluation/reporting: summary report, forecast-vs-actual plot, model comparison plot, Excel forecast export
- LSTM improvements: tuning flag, optional additive attention, multivariate engineered-feature recursive forecast
- Dockerisation: `Dockerfile` and `docker-compose.yml`
- Integration tests: preprocessing, feature engineering, split leakage, model interface smoke tests, API endpoint tests
- Optional scheduler: APScheduler weekly retraining module and `/scheduler/status`

## Original Remaining Tasks

---

### Task 1 — API Hardening & Error Handling

File: `src/api.py`

- Add input validation: ensure `state` names are validated against `registry.json` with a helpful error message listing available states
- Add a `GET /forecast/bulk` endpoint that accepts a list of states and returns forecasts for all of them in one call (request body: `{"states": ["California", "Texas"], "weeks": 8}`)
- Add response caching: use `functools.lru_cache` or a simple in-memory TTL dict so repeated calls to the same state don't re-load the model from disk every time
- Add a `GET /health/detailed` endpoint that returns: API status, number of trained states, registry last-modified timestamp
- Add request logging middleware using FastAPI's `middleware` decorator (log method, path, status code, response time)

---

### Task 2 — Evaluation & Reporting Module

Create file: `src/evaluation.py`

- Function `generate_report(registry: dict) -> pd.DataFrame`: creates a summary DataFrame with columns: State, Best_Model, MAPE, RMSE, Train_Time
- Function `plot_forecast_vs_actual(state: str, registry: dict, state_data: dict)`: plots actual vs predicted for validation period + the 8-week forecast horizon. Save as PNG to `outputs/{state}_forecast.png`
- Function `plot_model_comparison(registry: dict)`: horizontal bar chart showing MAPE by state, colored by winning model. Save as `outputs/model_comparison.png`
- Function `export_forecasts_to_excel(registry: dict, state_data: dict, output_path: str)`: runs inference for all states and writes an Excel file with one sheet per state containing forecast dates and values

---

### Task 3 — LSTM Improvements

File: `src/models/lstm_model.py`

- Add hyperparameter tuning: add a `tune=True` parameter to `train()` that tries 3 configurations (units=32/64/128, lookback=8/12/16) and picks the best by validation MSE
- Add attention mechanism: optionally wrap LSTM with a simple Bahdanau attention layer
- Fix the `last_sequence` update in `predict()` to correctly include all engineered features (currently only uses `Total` values)

---

### Task 4 — Dockerisation

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml` with:
- `api` service (the FastAPI app)
- `trainer` service that runs `python train.py` on container start with a volume mount for `model_registry/`
- Optional: `redis` service for caching (replace in-memory cache with Redis via `aioredis`)

---

### Task 5 — End-to-End Integration Test

Create file: `tests/test_pipeline.py`

- Test preprocessing: load data → resample → check no NaNs in key columns for any state
- Test feature engineering: verify lag_1 values match shifted Total values for a sample state
- Test train/val split: ensure no date leakage (val_dates > all train_dates)
- Test each model (SARIMA, Prophet, XGBoost, LSTM) on a small 2-year subset of California to ensure `train()` and `predict()` complete without error
- Test API endpoints using `fastapi.testclient.TestClient`: `/states`, `/forecast/California`, `/models/California`
- Add `pytest.ini` with test discovery config

---

### Task 6 — Model Retraining Scheduler (Optional / Bonus)

Create `src/scheduler.py`:
- Use `APScheduler` (`pip install apscheduler`) to trigger weekly retraining automatically
- Schedule: every Sunday at 2 AM, run `run_pipeline()` → `train_all_states()`
- Log retraining start/end/errors to `logs/retrain.log`
- Expose scheduler status via `GET /scheduler/status` in the API

---

## Important Notes for Next Agent

1. **Model registry path**: `model_registry/registry.json` — all model metadata lives here
2. **Model files**: saved as `model_registry/{State}_{ModelName}.pkl` (LSTM also saves `*_keras.keras`)
3. **State names**: title-cased (e.g. "New York", "North Carolina") — the API normalises via `.title()`
4. **Data frequency**: strictly weekly (Sunday-anchored `W` frequency) after resampling
5. **Forecast horizon**: default 8 weeks, configurable in `FORECAST_WEEKS` constant in `preprocessing.py`
6. **MAPE is the primary metric** for model selection (lower = better)
7. **All models implement the same interface**: `train(train_df)`, `predict(model_dict, n_periods)`, `evaluate(model_dict, val_df)`, `save(model_dict, path)`, `load(path)`
