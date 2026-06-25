"""
Module: hyperparameter_tuning.py
Description: Hyperparameter Tuning for Random Forest with Regularization.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from src.utils.paths import PROCESSED_DATA_DIR

def load_and_prepare_data():
    """Loads the cleaned dataset, applies the core consumer market filter, 
    and isolates variables into a clear numerical feature matrix and target vector."""
    print("Loading cleaned dataset for advanced regularization...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "listings_clean.csv")
    
    # Market Definition: exclude luxury outliers
    df = df[df["price"] <= 800]
    
    y = df["price"]
    X_raw = df.drop(columns=["price"], errors="ignore")
    
    # CATEGORICAL ENCODING:
    categorical_cols = ["room_type", "neighbourhood_group_cleansed"]
    categorical_cols = [c for c in categorical_cols if c in X_raw.columns]
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
    
    # NUMERICAL IMPUTATION
    X_numeric = X_encoded.select_dtypes(include=[np.number]).fillna(
        X_encoded.select_dtypes(include=[np.number]).median()
    )
    return X_numeric, y

def tune_hyperparameters_regularized():
    """
    Executes an intensive cross-validated grid exploration over specialized growth brakes
    to find the mathematically ideal configuration balancing bias and variance.
    """
    X, y = load_and_prepare_data()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=123)
    
    rf = RandomForestRegressor(random_state=123, n_jobs=-1)
    
    
    # We utilize a 10-Fold CV to evaluate anti-overfitting parameters:
    # min_samples_leaf: Prevents leaf nodes from becoming overly specific to single listings.
    # min_samples_split: Forces branches to halt expansion unless supported by substantial data density.
    param_grid = {
        'n_estimators': [300],                     
        'max_depth': [15, 20, 25],                
        'max_features': ['sqrt'],                 
        'min_samples_split': [2, 5, 10],           
        'min_samples_leaf': [1, 2, 4]            
    }
    
    # 270 fits -> 3 depths * 3 splits * 3 leaves * 10 folds
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=10, 
        scoring='r2', 
        n_jobs=-1, 
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    # RESULTS
    best_index = grid_search.best_index_
    train_score = grid_search.cv_results_['mean_train_score'][best_index]
    cv_score = grid_search.cv_results_['mean_test_score'][best_index]
    overfitting_gap = train_score - cv_score
    
    print("\n" + "="*40)
    print("  OPTIMAL REGULARIZED PARAMETERS:")
    print("="*40)
    for param, value in grid_search.best_params_.items():
        print(f"{param:<20}: {value}")
    print("="*40)
    
    print("\n" + "="*40)
    print(" POST REGULARIZATION VARIANCE EVALUATION")
    print("="*40)
    print(f"Clean Training R²   : {train_score:.4f}")
    print(f"Cross-Validation R²  : {cv_score:.4f}")
    print(f"Overfitting Gap: {overfitting_gap:.4f}")
    print("="*40)
    
    if overfitting_gap < 0.10:
        print("Success: The variance gap has dropped below 10%. The model is highly robust")
    else:
        print(f"Warning: The variance gap is still high at {overfitting_gap*100:.1f}%. Consider adjusting min_samples.")

if __name__ == "__main__":
    tune_hyperparameters_regularized()