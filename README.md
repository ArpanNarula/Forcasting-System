# 📦 Sales Forecasting System

End-to-end time series forecasting for US state-level weekly Beverages sales.

---

## Project Structure

```
forecasting_system/
├── data/
│   └── Forecasting_Case-_Study.xlsx      ← raw data
├── model_registry/
│   └── registry.json                     ← auto-generated after training
├── notebooks/
│   └── EDA.ipynb                         ← exploratory data analysis
├── src/
│   ├── preprocessing.py                  ← data loading, resampling, feature engineering
│   ├── model_selector.py                 ← orchestrates training + model selection
│   ├── api.py                            ← FastAPI REST service
│   └── models/
│       ├── sarima.py                     ← SARIMA (pmdarima auto_arima)
│       ├── prophet_model.py              ← Facebook Prophet
│       ├── xgboost_model.py              ← XGBoost with lag features
│       └── lstm_model.py                 ← LSTM (TensorFlow/Keras)
├── outputs/                              ← generated charts / Excel reports
├── tests/
│   └── test_pipeline.py                  ← integration tests
├── Dockerfile
├── docker-compose.yml
├── train.py                              ← training entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline

```bash
# Train all 43 states (takes ~30–60 min depending on hardware)
python train.py

# Train a subset during development
python train.py --states California Texas Florida "New York"
```

The pipeline will:
1. Load and resample the data to weekly frequency
2. Engineer lag, rolling, and calendar features
3. Train SARIMA, Prophet, XGBoost, and LSTM on each state
4. Evaluate each model on a held-out 8-week validation set
5. Select the best model per state by **MAPE**
6. Save models to `model_registry/`

### 3. Start the API

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs
Dashboard: http://localhost:8000/dashboard

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/states` | List all trained states |
| GET | `/forecast/{state}` | 8-week sales forecast |
| GET | `/forecast/{state}?weeks=N` | N-week sales forecast |
| GET | `/forecast/bulk` | Bulk forecasts, request body: `{"states": [...], "weeks": 8}` |
| POST | `/forecast/bulk` | Browser-friendly bulk forecast alias |
| GET | `/models/{state}` | Model comparison for state |
| GET | `/models/summary` | Best model for all states |
| GET | `/health/detailed` | Registry status, trained state count, cache size |
| GET | `/scheduler/status` | Optional weekly retraining scheduler status |
| POST | `/retrain` | Trigger retraining (background) |

### Example: Forecast California
```bash
curl http://localhost:8000/forecast/California
```

Response:
```json
{
  "state": "California",
  "model": "Prophet",
  "n_periods": 8,
  "forecast": [
    {"date": "2024-01-07", "predicted_sales": 498234567.89},
    ...
  ]
}
```

---

## Models

| Model | Key Config |
|-------|-----------|
| **SARIMA** | auto_arima, seasonal m=52, stepwise search |
| **Prophet** | multiplicative seasonality, US holidays, changepoint_prior=0.1 |
| **XGBoost** | 500 estimators, recursive multi-step forecast, lag features |
| **LSTM** | Multivariate sequence model, optional tuning, optional additive attention, MinMaxScaler, EarlyStopping |

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `lag_1` | Sales 1 week ago |
| `lag_4` | Sales 4 weeks ago (~1 month) |
| `lag_7` | Sales 7 weeks ago |
| `lag_30` | Sales 30 weeks ago |
| `lag_52` | Sales 52 weeks ago (~1 year) |
| `roll_mean_4/12` | Rolling mean over 4 / 12 weeks |
| `roll_std_4/12` | Rolling std over 4 / 12 weeks |
| `day_of_week` / `week_of_year` | Calendar day/week |
| `month` / `quarter` | Calendar month and quarter |
| `holiday_flag` | 1 if any US federal holiday in the week |

---

## Evaluation & Reporting

`src/evaluation.py` includes:

- `generate_report(registry)` for state/model/MAPE/RMSE/train-time summary
- `plot_forecast_vs_actual(...)` for validation and 8-week forecast PNGs
- `plot_model_comparison(registry)` for best-model MAPE comparison
- `export_forecasts_to_excel(...)` for one-sheet-per-state forecast exports

---

## Docker

```bash
docker compose up api
docker compose --profile training up trainer
```

The compose file mounts `model_registry/`, `outputs/`, and `logs/` so trained models and reports persist outside the container.

---

## Tests

```bash
python3 -m pytest
```

The tests cover preprocessing, feature engineering, time-series split leakage, all four model interfaces, and API endpoints.

---

## Notes

- SARIMA with m=52 is slow; `approximation=True` is set for speed
- LSTM saves the Keras model separately as `*_keras.keras` alongside the `.pkl`
- Retraining via POST `/retrain` runs in a background thread — monitor server logs
