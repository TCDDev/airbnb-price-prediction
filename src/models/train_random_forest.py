from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px # use either this or plt, whichever one you prefer
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error
)

from src.data.load import load_features_and_target
from src.models.pipelines import build_random_forest_pipeline
from src.utils.paths import FIGURES_DIR, METRICS_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def train():
    X, y = load_features_and_target()

    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    boolean_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=123
    )

    # build pipeline
    regrTree_pipeline  = build_random_forest_pipeline(numerical_cols, categorical_cols, boolean_cols)
    
    # fit model 
    regrTree_pipeline.fit(X_train, y_train)
    
    # generate predictions
    y_pred = regrTree_pipeline.predict(X_test)
    
    # FEATURE IMPORTANCE
    preprocessor = regrTree_pipeline.named_steps["preprocessor"]
    feature_names = []
    feature_names.extend(numerical_cols)
    
    cat_transformer = preprocessor.named_transformers_["cat"]
    if "onehot" in cat_transformer.named_steps:
        onehot_cols = cat_transformer.named_steps["onehot"].get_feature_names_out()
        feature_names.extend(onehot_cols)
    else:
        feature_names.extend(categorical_cols)
        
    feature_names.extend(boolean_cols)

    importances = regrTree_pipeline.named_steps["model"].feature_importances_
    named_importances = pd.Series(importances, index=feature_names)
    top_5 = named_importances.sort_values(ascending=False).head(5)
    
    print("\nTop 5 Feature Importances:")
    print(top_5)
    
    # evaluate metrics 
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # save figures and metrics to appropriate directories
    
    # PLOTS - Scatterplot of actual vs predicted values
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal', label="Random Forest Predictions")
    
    ideal_line = [y_test.min(), y_test.max()]
    plt.plot(ideal_line, ideal_line, color='red', linestyle='--', linewidth=2, label="Ideal Prediction")
    
    plt.title("Random Forest Regression: Actual vs. Predicted")
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    figure_path = FIGURES_DIR / "rf_actual_vs_predicted.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # PLOTS - feature importance bar chart
    plt.figure(figsize=(10, 6))
    
    top_features = named_importances.sort_values(ascending=False).head(10)
    top_features.sort_values(ascending=True).plot(kind='barh', color='teal')
    
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Relative Importance")
    plt.ylabel("Features")
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    feat_img_path = FIGURES_DIR / "rf_feature_importances.png"
    plt.savefig(feat_img_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # METRICS
    metrics_path = METRICS_DIR / "random_forest_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== Random Forest Evaluation ===\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"R²:   {r2:.4f}\n")
        f.write(f"MSE:  {mse:.4f}\n")

if __name__ == "__main__":
    train()