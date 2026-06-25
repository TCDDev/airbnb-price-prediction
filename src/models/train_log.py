"""
Module: train_log.py
Description: Model training (Random Forest) with Log-Transformed Airbnb Prices.
This approach stabilizes variance and reduces the influence of extreme outliers, potenially enhancing model performance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.paths import PROCESSED_DATA_DIR

def load_and_prepare_data():
    """
    Loads the dataset, enforces market boundaries, handles log-transformations 
    of the target variable, and engineers structural and text features.
    """
    df = pd.read_csv(PROCESSED_DATA_DIR / "listings_clean.csv")
    
    # Market Definition based on descriptive analysis
    df = df[df["price"] <= 800] 
    
    # LOGARITHMIC TRANSFORMATION
    # log(x+1) to handle right-skewed price distribution
    y_log = np.log1p(df["price"])
    
    # Isolating features from target variable
    X_raw = df.drop(columns=["price"], errors="ignore")
    
    # CATEGORICAL FEATURES: One-Hot Encoding
    categorical_cols = ["room_type", "neighbourhood_group_cleansed"]
    categorical_cols = [c for c in categorical_cols if c in X_raw.columns]
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
    
    # NUMERICAL FEATURES
    X_numeric = X_encoded.select_dtypes(include=[np.number]).fillna(
        X_encoded.select_dtypes(include=[np.number]).median()
    )
    
    # TEXT FEATURE ENGINEERING
    text_cols = ["name", "description", "neighborhood_overview", "space", "summary"]
    text_cols = [c for c in text_cols if c in X_raw.columns]
    
    text_features_list = []
    if text_cols:
        for col in text_cols:
            # Imputation of NaNs with empty strings
            text_series = X_raw[col].fillna("").astype(str)
            # Extracts top 20 structural tokes while ignoring common stop words
            tfidf = TfidfVectorizer(max_features=20, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(text_series)
            # maps feature names uniquely to prevent overlap during concatenation
            feature_names = [f"text_{col}_{name}" for name in tfidf.get_feature_names_out()]
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=X_raw.index)
            text_features_list.append(tfidf_df)
    
    # FEATURE CONCATENTATION: Merging all features 
    if text_features_list:
        X_all = pd.concat([X_numeric] + text_features_list, axis=1)
    else:
        X_all = X_numeric
        
    return X_all, y_log, df["price"] 

def train_log_model():
    """
    Trains the Random Forest model on a logarithmic scale, projects predictions 
    back to fiat currency (SGD), and runs a comparative metric evaluation.
    """
    X, y_log, y_true = load_and_prepare_data()
    
    # Train-Test Split
    X_train, X_test, y_train_log, y_test_log, _, y_test_true = train_test_split(
        X, y_log, y_true, test_size=0.2, random_state=123
    )
    
    print(f"Training Log-Model on {X_train.shape[0]} rows...")
    
    # HYPERPARAMETER CONFIGURATION
    model = RandomForestRegressor(
        n_estimators=50, 
        max_depth=15, 
        min_samples_leaf=2, 
        max_features='sqrt', 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train_log)
    
    # Predictions on log-transformed scale
    preds_train_log = model.predict(X_train)
    preds_test_log = model.predict(X_test)
    
    # Inverse transformation to original scale (SGD)
    preds_test_sgd = np.expm1(preds_test_log)      
    preds_train_sgd = np.expm1(preds_train_log)
    y_train_true = np.expm1(y_train_log)
    
    # EVALUATION METRICS
    mae = mean_absolute_error(y_test_true, preds_test_sgd)
    rmse = root_mean_squared_error(y_test_true, preds_test_sgd)
    r2_test = r2_score(y_test_true, preds_test_sgd)
    r2_train = r2_score(y_train_true, preds_train_sgd)
    overfitting_gap = r2_train - r2_test
    
    # OUTPUT
    print("\n" + "="*40)
    print("  LOG-TRANSFORMED MODEL EVALUATION")
    print("="*40)
    print(f"MAE (Test)             : {mae:.2f} SGD")
    print(f"RMSE (Test)            : {rmse:.2f} SGD")
    print("-" * 40)
    print(f"TrainingR²    : {r2_train:.4f}")
    print(f"Test R²          : {r2_test:.4f}")
    print(f"Overfitting Gap: {overfitting_gap:.4f}")
    print("="*40)

    # Diagnostic Performance Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_true, preds_test_sgd, alpha=0.4, color='purple') 
    plt.plot([0, 800], [0, 800], color='darkred', linestyle='--', linewidth=2)
    plt.title(f"Log-Model: Actual vs. Predicted Prices (Test R²: {r2_test:.3f})")
    plt.xlabel("Actual Prices (SGD)")
    plt.ylabel("Predicted Prices (SGD)")
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    plt.grid(True, alpha=0.3)
    
    plot_path = "log_model_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    train_log_model()