import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data.load import load_features_and_target
from src.models.pipelines import build_xgboost_pipeline
from src.utils.paths import FIGURES_DIR, METRICS_DIR


FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def train():
    """
    Train and evaluate the XGBoost model.
    """

    X, y = load_features_and_target()

    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    boolean_cols = X.select_dtypes(include=["bool"]).columns.tolist()

    print("Column overview:")
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Boolean columns: {len(boolean_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=123,
    )

    model = build_xgboost_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        boolean_cols=boolean_cols,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Convert to NumPy arrays for plotting
    actual = np.asarray(y_test)
    predicted = np.asarray(y_pred)
    residuals = actual - predicted

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    metrics = {
        "model": "xgboost",
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

    print("\nXGBoost test metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    metrics_path = METRICS_DIR / "xgboost_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved metrics to: {metrics_path}")

    # 1. Full actual vs predicted plot

    plt.figure(figsize=(7, 5))
    plt.scatter(actual, predicted, alpha=0.5)

    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("XGBoost: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgboost_actual_vs_predicted.png")
    plt.close()

    # 2. Capped actual vs predicted plot

    price_cutoff = np.quantile(np.concatenate([actual, predicted]), 0.99)

    mask = (actual <= price_cutoff) & (predicted <= price_cutoff)

    actual_capped = actual[mask]
    predicted_capped = predicted[mask]

    plt.figure(figsize=(7, 5))
    plt.scatter(actual_capped, predicted_capped, alpha=0.5)

    min_value = min(actual_capped.min(), predicted_capped.min())
    max_value = max(actual_capped.max(), predicted_capped.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("XGBoost: Actual vs Predicted, capped at 99th percentile")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgboost_actual_vs_predicted_capped.png")
    plt.close()

    # 3. Capped residual plot

    residual_cutoff = np.quantile(np.abs(residuals), 0.99)

    mask = np.abs(residuals) <= residual_cutoff

    plt.figure(figsize=(7, 5))
    plt.scatter(predicted[mask], residuals[mask], alpha=0.5)
    plt.axhline(0, linestyle="--")

    plt.xlabel("Predicted price")
    plt.ylabel("Residual")
    plt.title("XGBoost: Residual Plot, capped at 99th percentile")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgboost_residuals_capped.png")
    plt.close()

    # 4. Log-scaled actual vs predicted plot

    # Clip only for plotting -> because log(1 + x) is not defined for x <= -1.
    predicted_for_log = np.clip(predicted, a_min=0, a_max=None)

    actual_log = np.log1p(actual)
    predicted_log = np.log1p(predicted_for_log)

    plt.figure(figsize=(7, 5))
    plt.scatter(actual_log, predicted_log, alpha=0.5)

    min_value = min(actual_log.min(), predicted_log.min())
    max_value = max(actual_log.max(), predicted_log.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    plt.xlabel("log(1 + actual price)")
    plt.ylabel("log(1 + predicted price)")
    plt.title("XGBoost: Actual vs Predicted on log scale")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgboost_actual_vs_predicted_log.png")
    plt.close()

    print("\nSaved plots:")
    print(FIGURES_DIR / "xgboost_actual_vs_predicted.png")
    print(FIGURES_DIR / "xgboost_actual_vs_predicted_capped.png")
    print(FIGURES_DIR / "xgboost_residuals_capped.png")
    print(FIGURES_DIR / "xgboost_actual_vs_predicted_log.png")

    return model, metrics


if __name__ == "__main__":
    train()
