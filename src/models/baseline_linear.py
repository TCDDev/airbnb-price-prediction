"""
Module: baseline_linear.py
Description: Linear Regression Baseline for the Airbnb Singapore project.
Serves as the classical statistical benchmark (OLS) against ML methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from src.utils.paths import PROCESSED_DATA_DIR

def load_data():
    """
    Loads the cleaned dataset, applies the core consumer market filter,
    and handles kategorial dummy variables and numeric median imputation.
    """
    print("Loading cleaned dataset for Linear Regression Baseline...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "listings_clean.csv")
    
    # MARKET DEFINTION based on descriptive analysis
    df = df[df["price"] <= 800] 
    
    y = df["price"]
    X_raw = df.drop(columns=["price"], errors="ignore")
    
    # CATEGORICAL FEATURES: One-Hot Encoding
    categorical_cols = ["room_type", "neighbourhood_group_cleansed"]
    categorical_cols = [c for c in categorical_cols if c in X_raw.columns]
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
    
    # NUMERICAL FEATURES: Median Imputations for NaNs
    X_numeric = X_encoded.select_dtypes(include=[np.number]).fillna(
        X_encoded.select_dtypes(include=[np.number]).median()
    )
    return X_numeric, y

def main():
    X, y = load_data()
    
    # TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    print(f"Training OLS Linear Regression on {X_train.shape[0]} rows...")
    
    # Instantiate and fit the OLS model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictons
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    # Metrics
    r2_train = r2_score(y_train, preds_train)
    r2_test = r2_score(y_test, preds_test)
    mae = mean_absolute_error(y_test, preds_test)
    rmse = root_mean_squared_error(y_test, preds_test)
    gap = r2_train - r2_test
    
    # OUTPUT
    print("\n" + "="*40)
    print("  LINEAR REGRESSION BASELINE (OLS)")
    print("="*40)
    print(f"MAE (Test)             : {mae:.2f} SGD")
    print(f"RMSE (Test)            : {rmse:.2f} SGD")
    print("-" * 40)
    print(f"Training R²    : {r2_train:.4f}")
    print(f"Test R²          : {r2_test:.4f}")
    print(f"Overfitting Gap: {gap:.4f}")
    print("="*40)
    
    # COEFFICIENTS FOR PRESENTATION
    # Extracts explicit beta weights representing ceteris paribus marginal changes
    print("\nTop 5 Strongest Positive Price Drivers (Coefficients):")
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    print(coefficients.sort_values(by='Coefficient', ascending=False).head(5).to_string(index=False))
    
    print("\nTop 5 Strongest Negative Price Drivers (Coefficients):")
    print(coefficients.sort_values(by='Coefficient', ascending=True).head(5).to_string(index=False))

    # ACTUAL VS PREDICTED PLOT
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds_test, alpha=0.4, color='coral') 
    plt.plot([0, 800], [0, 800], color='darkred', linestyle='--', linewidth=2)
    plt.title(f"Linear Regression: Actual vs. Predicted Prices (Test R²: {r2_test:.3f})")
    plt.xlabel("Actual Prices (SGD)")
    plt.ylabel("Predicted Prices (SGD)")
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    plt.grid(True, alpha=0.3)
    
    plot_path = "linear_baseline_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()