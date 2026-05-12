"""
models/lstm_model.py
--------------------
Multivariate LSTM forecaster with optional hyperparameter tuning and attention.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FORECAST_WEEKS = 8
LOOKBACK = 12
EPOCHS = 50
BATCH = 16

FEATURE_COLS = [
    "Total",
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


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("Date").dropna(subset=FEATURE_COLS).reset_index(drop=True)


def _build_sequences(features: np.ndarray, target: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback: i])
        y.append(target[i])
    return np.array(X), np.array(y)


def _build_model(lookback: int, n_features: int, units: int, use_attention: bool):
    import tensorflow as tf
    from tensorflow.keras.layers import (
        LSTM,
        AdditiveAttention,
        Concatenate,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        Input,
    )
    from tensorflow.keras.models import Model, Sequential

    tf.keras.utils.set_random_seed(42)

    if use_attention:
        inputs = Input(shape=(lookback, n_features))
        x = LSTM(units, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        context = AdditiveAttention()([x, x])
        x = Concatenate()([x, context])
        x = GlobalAveragePooling1D()(x)
        x = Dense(max(units // 2, 16), activation="relu")(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
    else:
        model = Sequential(
            [
                LSTM(units, return_sequences=True, input_shape=(lookback, n_features)),
                Dropout(0.2),
                LSTM(max(units // 2, 16)),
                Dropout(0.2),
                Dense(1),
            ]
        )

    model.compile(optimizer="adam", loss="mse")
    return model


def _fit_model(train_df: pd.DataFrame, lookback: int, units: int, use_attention: bool):
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.callbacks import EarlyStopping

    tf.get_logger().setLevel("ERROR")

    clean = _clean(train_df)
    if len(clean) <= lookback + 2:
        raise ValueError(f"Not enough clean rows to train LSTM with lookback={lookback}.")

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(clean[FEATURE_COLS].values)
    scaled_target = target_scaler.fit_transform(clean[["Total"]].values).flatten()

    X, y = _build_sequences(scaled_features, scaled_target, lookback)
    model = _build_model(lookback, len(FEATURE_COLS), units, use_attention)
    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=0)

    return {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": FEATURE_COLS,
        "lookback": lookback,
        "units": units,
        "use_attention": use_attention,
        "last_train_df": clean,
        "last_sequence": scaled_features[-lookback:],
        "last_train_date": clean["Date"].max(),
    }


def _score_config(train_df: pd.DataFrame, units: int, lookback: int, use_attention: bool) -> float:
    val_size = min(FORECAST_WEEKS, max(4, len(train_df) // 10))
    fit_df = train_df.iloc[:-val_size].copy()
    val_df = train_df.iloc[-val_size:].copy()
    model_dict = _fit_model(fit_df, lookback, units, use_attention)
    preds = predict(model_dict, n_periods=len(val_df))
    actuals = val_df["Total"].values
    return float(np.mean((actuals - preds) ** 2))


def train(train_df: pd.DataFrame, tune: bool = False, use_attention: bool = False) -> dict:
    """
    Fit an LSTM model.

    When tune=True, tries three lightweight configurations and retrains the best
    one on the full training window using validation MSE for selection.
    """
    try:
        import tensorflow  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        raise ImportError("Install tensorflow and scikit-learn to train the LSTM model.")

    configs = [(64, LOOKBACK)]
    if tune:
        configs = [(32, 8), (64, 12), (128, 16)]

    best = None
    for units, lookback in configs:
        try:
            mse = _score_config(train_df, units, lookback, use_attention) if tune else 0.0
            if best is None or mse < best["mse"]:
                best = {"units": units, "lookback": lookback, "mse": mse}
        except Exception:
            continue

    if best is None:
        best = {"units": 64, "lookback": min(LOOKBACK, max(4, len(train_df) // 4)), "mse": None}

    model_dict = _fit_model(train_df, best["lookback"], best["units"], use_attention)
    model_dict["tuning"] = {
        "enabled": tune,
        "best_units": best["units"],
        "best_lookback": best["lookback"],
        "best_validation_mse": best["mse"],
    }
    return model_dict


def _future_feature_row(history: pd.DataFrame, next_date: pd.Timestamp) -> pd.DataFrame:
    vals = history["Total"].values
    lag1 = vals[-1]
    lag4 = vals[-4] if len(vals) >= 4 else vals[0]
    lag7 = vals[-7] if len(vals) >= 7 else vals[0]
    lag30 = vals[-30] if len(vals) >= 30 else vals[0]
    lag52 = vals[-52] if len(vals) >= 52 else vals[0]

    roll4 = pd.Series(vals[-4:]).mean()
    roll12 = pd.Series(vals[-12:]).mean()
    std4 = pd.Series(vals[-4:]).std() if len(vals) >= 4 else 0.0
    std12 = pd.Series(vals[-12:]).std() if len(vals) >= 12 else 0.0

    return pd.DataFrame(
        {
            "Date": [next_date],
            "State": [history["State"].iloc[0] if "State" in history else "Unknown"],
            "Total": [np.nan],
            "lag_1": [lag1],
            "lag_4": [lag4],
            "lag_7": [lag7],
            "lag_30": [lag30],
            "lag_52": [lag52],
            "roll_mean_4": [roll4],
            "roll_mean_12": [roll12],
            "roll_std_4": [0.0 if pd.isna(std4) else std4],
            "roll_std_12": [0.0 if pd.isna(std12) else std12],
            "day_of_week": [next_date.dayofweek],
            "week_of_year": [next_date.isocalendar()[1]],
            "month": [next_date.month],
            "quarter": [(next_date.month - 1) // 3 + 1],
            "year": [next_date.year],
            "holiday_flag": [_is_holiday_week(next_date)],
        }
    )


def predict(model_dict: dict, n_periods: int = FORECAST_WEEKS) -> np.ndarray:
    model = model_dict["model"]
    feature_scaler = model_dict["feature_scaler"]
    target_scaler = model_dict["target_scaler"]
    feature_cols = model_dict.get("feature_cols", FEATURE_COLS)
    lookback = model_dict.get("lookback", LOOKBACK)

    history = model_dict["last_train_df"].copy().sort_values("Date").reset_index(drop=True)
    seq = model_dict["last_sequence"].copy()

    preds = []
    for _ in range(n_periods):
        inp = seq[-lookback:].reshape(1, lookback, len(feature_cols))
        pred_scaled = float(model.predict(inp, verbose=0)[0, 0])
        pred = float(target_scaler.inverse_transform([[pred_scaled]])[0, 0])
        pred = max(pred, 0.0)
        preds.append(pred)

        next_date = pd.Timestamp(history["Date"].iloc[-1]) + pd.Timedelta(weeks=1)
        next_row = _future_feature_row(history, next_date)
        next_row["Total"] = pred
        scaled_next = feature_scaler.transform(next_row[feature_cols].values)
        seq = np.vstack([seq, scaled_next])
        history = pd.concat([history, next_row], ignore_index=True)

    return np.array(preds)


def evaluate(model_dict: dict, val_df: pd.DataFrame) -> dict:
    actuals = val_df["Total"].values
    preds = predict(model_dict, n_periods=len(actuals))

    mape = np.mean(np.abs((actuals - preds) / np.where(actuals == 0, 1, actuals))) * 100
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))

    return {
        "model_name": "LSTM",
        "mape": round(float(mape), 4),
        "rmse": round(float(rmse), 4),
        "predictions": preds.tolist(),
    }


def save(model_dict: dict, path: Path):
    """Save LSTM separately: Keras model plus scaler/metadata pickle."""
    keras_path = str(path).replace(".pkl", "_keras.keras")
    model_dict["model"].save(keras_path)
    meta = {k: v for k, v in model_dict.items() if k != "model"}
    meta["keras_path"] = keras_path
    with open(path, "wb") as f:
        pickle.dump(meta, f)


def load(path: Path) -> dict:
    import tensorflow as tf

    with open(path, "rb") as f:
        meta = pickle.load(f)
    meta["model"] = tf.keras.models.load_model(meta["keras_path"])
    return meta
