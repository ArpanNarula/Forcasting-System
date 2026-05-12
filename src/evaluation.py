"""
evaluation.py
-------------
Reporting utilities for trained state-level forecasting models.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.model_selector import MODEL_MODULES, load_best_model

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_report(registry: dict) -> pd.DataFrame:
    """Create a per-state summary DataFrame from the model registry."""
    rows = []
    for state, info in registry.items():
        best_model = info.get("best_model")
        comparison = info.get("comparison", {})
        train_time = comparison.get(best_model, {}).get("train_time_sec")
        rows.append(
            {
                "State": state,
                "Best_Model": best_model,
                "MAPE": info.get("best_mape"),
                "RMSE": info.get("best_rmse"),
                "Train_Time": train_time,
            }
        )
    return pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True)


def _validation_predictions(state: str, registry: dict, state_df: pd.DataFrame) -> list[float]:
    info = registry[state]
    if info.get("val_predictions"):
        return info["val_predictions"]

    from src.preprocessing import train_val_split

    _, val_df = train_val_split(state_df)
    model_dict, model_name = load_best_model(state)
    module = MODEL_MODULES[model_name]
    return module.predict(model_dict, n_periods=len(val_df)).tolist()


def plot_forecast_vs_actual(state: str, registry: dict, state_data: dict):
    """
    Plot validation actuals vs validation predictions plus the 8-week future
    horizon. Saves to outputs/{state}_forecast.png.
    """
    if state not in registry:
        raise KeyError(f"State '{state}' is not present in registry.")
    if state not in state_data:
        raise KeyError(f"State '{state}' is not present in state_data.")

    from src.model_selector import forecast
    from src.preprocessing import train_val_split

    state_df = state_data[state]
    _, val_df = train_val_split(state_df)
    val_preds = _validation_predictions(state, registry, state_df)
    future = forecast(state, n_periods=8)

    plt.figure(figsize=(12, 6))
    plt.plot(val_df["Date"], val_df["Total"], marker="o", label="Validation actual")
    plt.plot(val_df["Date"], val_preds, marker="o", label="Validation forecast")
    future_dates = pd.to_datetime([row["date"] for row in future["forecast"]])
    future_values = [row["predicted_sales"] for row in future["forecast"]]
    plt.plot(future_dates, future_values, marker="o", linestyle="--", label="8-week forecast")
    plt.title(f"{state} sales forecast vs actual")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{state.replace(' ', '_')}_forecast.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_model_comparison(registry: dict):
    """Create a horizontal bar chart of MAPE by state, colored by winning model."""
    report = generate_report(registry)
    models = sorted(report["Best_Model"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    colors = {model: cmap(i % 10) for i, model in enumerate(models)}
    bar_colors = [colors.get(model, "gray") for model in report["Best_Model"]]

    plt.figure(figsize=(12, max(6, 0.35 * len(report))))
    plt.barh(report["State"], report["MAPE"], color=bar_colors)
    plt.title("Best model MAPE by state")
    plt.xlabel("MAPE (%)")
    plt.ylabel("State")
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=model)
        for model, color in colors.items()
    ]
    if handles:
        plt.legend(handles=handles, title="Best Model", loc="best")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def export_forecasts_to_excel(registry: dict, state_data: dict, output_path: str):
    """
    Run inference for all registry states and write one sheet per state.
    Each sheet contains validation actuals/predictions and future forecasts.
    """
    from src.model_selector import forecast
    from src.preprocessing import train_val_split

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        generate_report(registry).to_excel(writer, sheet_name="Summary", index=False)

        for state in sorted(registry.keys()):
            if state not in state_data:
                continue

            _, val_df = train_val_split(state_data[state])
            val_preds = _validation_predictions(state, registry, state_data[state])
            future = forecast(state, n_periods=8)

            validation_rows = pd.DataFrame(
                {
                    "date": val_df["Date"].dt.date.astype(str),
                    "actual_sales": val_df["Total"].values,
                    "predicted_sales": val_preds,
                    "period": "validation",
                }
            )
            future_rows = pd.DataFrame(future["forecast"])
            future_rows["actual_sales"] = None
            future_rows["period"] = "forecast"
            future_rows = future_rows[["date", "actual_sales", "predicted_sales", "period"]]

            sheet = state[:31]
            pd.concat([validation_rows, future_rows], ignore_index=True).to_excel(
                writer,
                sheet_name=sheet,
                index=False,
            )

    return output
