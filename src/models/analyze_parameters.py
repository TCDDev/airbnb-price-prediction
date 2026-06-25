"""
Module: analyze_parameters.py
Description: Sensitivity Analysis for Random Forest Hyperparameters.
Generates Cross-Validated Validation Curves to visually map the overfitting gap and optimnize regularizers.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from src.utils.paths import PROCESSED_DATA_DIR

def load_data():
    """
    Loads the cleaned dataset, restricts the target variable to the core market,
    and returns a clean, imputed matrix alongside the target vector.
    """
    df = pd.read_csv(PROCESSED_DATA_DIR / "listings_clean.csv")
    # Market Definition: exclude luxury outliers
    df = df[df["price"] <= 800] 
    
    y = df["price"]
    X_raw = df.drop(columns=["price"], errors="ignore")
    
    # CATEGORICAL FEATURES: One-Hot Encoding
    categorical_cols = ["room_type", "neighbourhood_group_cleansed"]
    categorical_cols = [c for c in categorical_cols if c in X_raw.columns]
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
    X_numeric = X_encoded.select_dtypes(include=[np.number]).fillna(X_encoded.select_dtypes(include=[np.number]).median())
    
    return X_numeric, y

def plot_validation_curve(param_name, param_values, base_estimator, X, y):
    """TEvaluates a specific hyperparameter across an array of values, running 
    a 5-Fold Cross Validation loop to save training vs. validation profiles."""
    
    train_scores_mean = []
    val_scores_mean = []
    
    for value in param_values:
        # Creates a clean reference instance and inject the target hyperparameter
        model = base_estimator
        model.set_params(**{param_name: value})
        
        # Execute 5-Fold Cross Validation returning both training and validation metrics
        cv_results = cross_validate(model, X, y, cv=5, scoring='r2', return_train_score=True, n_jobs=-1)
        
        # Aggregate out-of-sample and in-sample metrics across all folds
        train_scores_mean.append(np.mean(cv_results['train_score']))
        val_scores_mean.append(np.mean(cv_results['test_score']))
    
    # PLOTTING
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, train_scores_mean, label='Training R²', color='darkred', marker='o', linewidth=2)
    plt.plot(param_values, val_scores_mean, label='Validation R²', color='teal', marker='o', linewidth=2)
    
    # Visually shade the overfitting gap between training and validation curves
    plt.fill_between(param_values, val_scores_mean, train_scores_mean, alpha=0.15, color='red', label='Overfitting-Kluft')
    
    plt.title(f"Validation Curve: Impact of '{param_name}' on Overfitting")
    plt.xlabel(param_name)
    plt.ylabel("R² Score")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    filename = f"curve_{param_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def main():
    X,y = load_data()
   # Isolate training space to guarantee clean parameter analysis (80% holdout split)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # MAX_DEPTH
    # Demonstrates how higher depths trigger massive overfitting as trees memorize noise.
    base_rf = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=123, n_jobs=-1)
    plot_validation_curve('max_depth', [5, 10, 15, 20, 25, 30], base_rf, X_train, y_train)
    
    # N_ESTIMATORS
    # Proves that expanding tree count stabilizes variance and does not trigger overfitting.
    base_rf = RandomForestRegressor(max_depth=15, max_features='sqrt', random_state=123, n_jobs=-1)
    plot_validation_curve('n_estimators', [10, 50, 100, 200, 300], base_rf, X_train, y_train)
    
    # MIN_SAMPLES_LEAF
    # Demonstrates regularization: Larger values restrict growth, forcing trees to generalize.
    base_rf = RandomForestRegressor(n_estimators=100, max_depth=25, max_features='sqrt', random_state=123, n_jobs=-1)
    plot_validation_curve('min_samples_leaf', [1, 2, 4, 8, 16], base_rf, X_train, y_train)
if __name__ == "__main__":
    main()