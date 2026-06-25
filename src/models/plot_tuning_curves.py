"""
Module: plot_tuning_curves.py
Description: Generates validation curves for Random Forest hyperparameters. 
The curves visually illustrate the trade-off between model complexity and generalization, aiding in the selection of optimal hyperparameter values. 
Plots the training vs. validation scores, highlights the overfitting gap.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from src.utils.paths import PROCESSED_DATA_DIR

def load_data():
    # Loads the cleaned dataset, applies the core consumer market abstraction,
    # and isolates variables into an encoded feature matrix and target vector.
    print("Loading cleaned dataset for curve plotting...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "listings_clean.csv")
    
    # Market Definition: exclude luxury outliers
    df = df[df["price"] <= 800] 
    
    y = df["price"]
    X_raw = df.drop(columns=["price"], errors="ignore")
    
    # CATEGORICAL: One-Hot Encoding
    categorical_cols = ["room_type", "neighbourhood_group_cleansed"]
    categorical_cols = [c for c in categorical_cols if c in X_raw.columns]
    
    # NUMERICAL: Median Imputation
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
    X_numeric = X_encoded.select_dtypes(include=[np.number]).fillna(
        X_encoded.select_dtypes(include=[np.number]).median()
    )
    return X_numeric, y

def generate_and_save_curve(param_name, param_values, base_estimator, X, y, chosen_value):
    """
    Computes cross-validated training and validation scores across an array 
    of hyperparameter settings and exports a stylized evaluation plot.
    """
    print(f"\nEvaluating parameter: '{param_name}' over values {param_values}...")
    
    train_scores_mean = []
    val_scores_mean = []
    
    for value in param_values:
        model = base_estimator
        model.set_params(**{param_name: value})
        
        # 5-folg cross validation 
        cv_results = cross_validate(model, X, y, cv=5, scoring='r2', return_train_score=True, n_jobs=-1)
        
        train_scores_mean.append(np.mean(cv_results['train_score']))
        val_scores_mean.append(np.mean(cv_results['test_score']))
    
    # PLOTTING
    plt.figure(figsize=(8, 5.5))
    

    plt.plot(param_values, train_scores_mean, label='Training R²', color='darkred', marker='o', linewidth=2)
    plt.plot(param_values, val_scores_mean, label='Validation R²', color='teal', marker='o', linewidth=2)
    plt.fill_between(param_values, val_scores_mean, train_scores_mean, alpha=0.15, color='red', label='Overfitting Gap')
    
    # Highlight the chosen design
    if chosen_value in param_values:
        chosen_idx = param_values.index(chosen_value)
        chosen_val_score = val_scores_mean[chosen_idx]
        
        plt.axvline(x=chosen_value, color='#d97706', linestyle='--', linewidth=1.5, label=f'Chosen Value ({chosen_value})')
        plt.scatter([chosen_value], [chosen_val_score], color='#d97706', s=120, zorder=5, edgecolors='black')
    
    plt.title(f"Validation Curve: Impact of '{param_name}' on Overfitting", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel(param_name, fontsize=11)
    plt.ylabel("R² Score", fontsize=11)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle=':')

    filename = f"presentation_curve_{param_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def main():
    X, y = load_data()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=123)
    
    print(f"Starting curve generation on {X_train.shape[0]} samples...")
    
    # max_depth
    rf_depth = RandomForestRegressor(n_estimators=50, max_features='sqrt', random_state=123, n_jobs=-1)
    generate_and_save_curve('max_depth', [3, 5, 10, 15, 20, 25, 30], rf_depth, X_train, y_train, chosen_value=15)
    
    # n_estimator
    rf_estimators = RandomForestRegressor(max_depth=15, max_features='sqrt', random_state=123, n_jobs=-1)
    generate_and_save_curve('n_estimators', [10, 30, 50, 100, 200, 300], rf_estimators, X_train, y_train, chosen_value=50)
    
    # min_sample_leaf
    rf_leaf = RandomForestRegressor(n_estimators=50, max_depth=15, max_features='sqrt', random_state=123, n_jobs=-1)
    generate_and_save_curve('min_samples_leaf', [1, 2, 4, 8, 16, 32], rf_leaf, X_train, y_train, chosen_value=2)

if __name__ == "__main__":
    main()