# Sales Forecasting System

This project is my end-to-end time series forecasting system for the assignment. The goal was to forecast the next 8 weeks of sales for each state using historical sales data, compare multiple forecasting algorithms, automatically select the best model, and expose the predictions through a REST API.

I designed it like a backend forecasting service instead of only a notebook. The system includes data preprocessing, feature engineering, four model families, model comparison, model registry, API endpoints, reporting outputs, Docker support, tests, and a small dashboard for explanation.

## What I Built

The complete flow of the project is:

1. Load the Excel sales dataset.
2. Clean and resample the data into weekly state-level time series.
3. Handle missing dates and missing values.
4. Create time series features such as lags, rolling statistics, calendar features, and holiday flags.
5. Train four forecasting models for each state:
   - SARIMA
   - Prophet
   - XGBoost
   - LSTM
6. Evaluate every model using a time-based validation split.
7. Select the best model automatically using MAPE as the primary metric.
8. Save the selected model and metadata in a model registry.
9. Serve forecasts through FastAPI.
10. Show forecasts and model results through a dashboard.

## Folder Explanation

### `data/`

This folder contains the Excel dataset used for the case study.

The pipeline starts from this file. The data is loaded, sorted by state and date, converted into weekly frequency, and then used for model training.

### `src/`

This is the main backend source code folder.

It contains the complete forecasting pipeline:

- `preprocessing.py` handles data loading, weekly resampling, missing value handling, feature engineering, and train-validation split.
- `model_selector.py` trains all models, compares their metrics, selects the best model, and saves the registry.
- `api.py` exposes the trained forecasting system using FastAPI.
- `evaluation.py` generates reports, plots, and Excel forecast exports.
- `scheduler.py` contains optional weekly retraining logic.
- `models/` contains the individual forecasting model implementations.

### `src/models/`

This folder contains the four required models.

- `sarima.py`: statistical time series model for trend and seasonality.
- `prophet_model.py`: business forecasting model with trend, seasonality, and holidays.
- `xgboost_model.py`: machine learning model using lag, rolling, calendar, and holiday features.
- `lstm_model.py`: deep learning sequence model using engineered time series features.

All models follow the same structure: `train`, `predict`, `evaluate`, `save`, and `load`. This keeps the training and selection pipeline clean.

### `model_registry/`

This folder stores trained model artifacts and registry metadata.

The `registry.json` file stores information such as:

- state name
- selected best model
- MAPE
- RMSE
- model path
- validation actuals
- validation predictions

For the local demo, I trained California as a sample artifact. The same pipeline supports all 43 states when full training is run.

### `outputs/`

This folder contains generated reporting outputs.

It includes:

- forecast-vs-actual chart
- model comparison chart
- Excel forecast export

These outputs help explain model performance and forecast results outside the API.

### `frontend/`

This folder contains the dashboard.

The dashboard is served by FastAPI and uses the backend API directly. It shows:

- API health
- trained state count
- state selector
- forecast horizon controls
- selected model
- MAPE and RMSE
- forecast chart
- forecast table
- model leaderboard
- latest API JSON response

This makes the project easier to explain visually during the video.

### `tests/`

This folder contains integration tests.

The tests cover:

- preprocessing
- lag feature correctness
- time-series train-validation split
- all four model interfaces
- API endpoints

### Docker Files

The project also includes Docker support with:

- `Dockerfile`
- `docker-compose.yml`

These files make it possible to run the API and training service in a containerized way.

## Feature Engineering

I created the following features because time series models and machine learning models need historical and calendar-based signals:

- `lag_1`
- `lag_7`
- `lag_30`
- `lag_4`
- `lag_52`
- rolling mean
- rolling standard deviation
- day of week
- week of year
- month
- quarter
- year
- holiday flag

The lag features help the model understand previous sales behavior. Rolling features help capture recent trend and volatility. Calendar and holiday features help capture seasonality and event-based effects.

## Model Training and Selection

For every state, the system trains and evaluates four models:

### SARIMA

SARIMA is used as a classical statistical baseline. It is useful for time series data with trend and seasonality.

### Prophet

Prophet is used because it works well for business time series and can handle trend, seasonality, and holidays.

### XGBoost

XGBoost uses the engineered features such as lag values, rolling statistics, calendar variables, and holiday flags. It performs recursive forecasting by predicting one future week at a time and feeding the prediction back into the feature generation process.

### LSTM

LSTM is used as the deep learning model. It learns from sequences of historical data. I also added support for multivariate engineered features, optional tuning, and optional attention.

After training, every model is evaluated on a validation window. The system selects the model with the lowest MAPE. MAPE is used as the primary metric because it is easy to interpret as percentage error.

## API Layer

The API is built using FastAPI.

Important endpoints include:

- `/states`: returns trained states.
- `/forecast/{state}`: returns forecast for one state.
- `/forecast/bulk`: returns forecasts for multiple states.
- `/models/{state}`: returns model comparison for a state.
- `/models/summary`: returns selected model summary.
- `/health/detailed`: returns API and registry health.
- `/retrain`: starts retraining in the background.
- `/dashboard`: opens the frontend dashboard.

The API also includes input validation, request logging, response caching, and detailed health checks.

## Dashboard

The dashboard is included to make the system easier to understand.

Instead of only showing JSON, the dashboard visualizes:

- forecast values
- forecast trend
- selected best model
- model metrics
- model comparison
- service health
- live API response

This helps explain both the backend and the forecasting result clearly in a demo video.

## Evaluation and Reporting

The evaluation module can generate:

- summary report
- forecast-vs-actual plot
- model comparison plot
- Excel workbook with forecasts

This is useful for sharing results with non-technical users or reviewers.

## Current Demo State

For the local demo, I trained California and saved its best model artifact in the registry. The selected model for California is XGBoost.

The project is designed to support all 43 states. Full training can be run using the same pipeline, but it takes more time because each state trains SARIMA, Prophet, XGBoost, and LSTM.

## Assignment Coverage

This project covers the required points from the assignment:

- Multiple forecasting algorithms implemented.
- SARIMA / Prophet / XGBoost / LSTM included.
- Missing dates and values handled.
- Seasonality and trend handled through model design and features.
- Lag features created.
- Rolling mean and standard deviation created.
- Calendar and holiday features created.
- Time-series validation split used.
- Best model selected automatically using MAPE.
- Predictions exposed through REST API.
- Dashboard created for better understanding.
- Reports and Excel outputs generated.
- Integration tests added.
- Docker support added.

## Final Summary

This project is a complete backend-style forecasting system. It starts from raw Excel data, prepares weekly time series, creates features, trains multiple models, compares them, saves the best model, exposes forecasts through FastAPI, and presents the result through a dashboard.

The main focus of the project is not only forecasting accuracy, but also building the system in a structured and production-ready way.
