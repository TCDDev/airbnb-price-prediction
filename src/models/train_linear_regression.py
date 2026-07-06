from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px # use either this or plt, whichever one you prefer
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error
)

from src.data.load import load_features_and_target
from src.models.pipelines import build_linear_pipeline
from src.utils.paths import FIGURES_DIR, METRICS_DIR # save metrics and figures to these, i.e. FIGURES_DIR / "linreg_actual_vs_predicted"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def train():
    

    X, y = load_features_and_target()

    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    boolean_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=123
    )
    
    # build pipeline
    linear_pipeline = build_linear_pipeline(numerical_cols, categorical_cols, boolean_cols)
    
    # fit model 
    linear_pipeline.fit(X_train, y_train)
    
    # generate predictions
    y_pred = linear_pipeline.predict(X_test)
    
    # evaluate metrics 
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # save figures and metrics to appropriate directories
    
    # METRICS
    metrics_path = METRICS_DIR / "linear_regression_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== Linear Regression Evaluation ===\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"R²:   {r2:.4f}\n")

    # PLOT
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="teal", label="Vorhergesagte Preise")
    
    ideal_line = [y_test.min(), y_test.max()]
    plt.plot(ideal_line, ideal_line, color="red", linestyle="--", linewidth=2, label="Perfekte Vorhersage")
    
    plt.xlabel("True Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Linear Regression (Ridge): Actual vs. Predicted Prices")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    figure_path = FIGURES_DIR / "linreg_actual_vs_predicted.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close() 

if __name__ == "__main__":
    train()