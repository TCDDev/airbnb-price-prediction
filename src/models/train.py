"""
Module: train.py
Description: Optimal Random Forest Regressor for Airbnb price prediction in Singapore.
Includes geo-spatial feature engineering (Haversine distance) and text feature engineering.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.paths import PROCESSED_DATA_DIR

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance in kilometres between two geographic coordinates on 
    a sphere using the trigonometric Haversine formula.
    
    Economic Rationale: Replaces the rigid, flawed assumption of linear price trends 
    with a true, radial distance metric relative to the economic epicenter.
    """
    # Mean Earth radius in kilometers
    R = 6371.0
    
    # Convert decimal degree coordinates to radians for NumPy trigonometric functions
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Coordinate Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Multiplying by the Earth's radius yields the exact distance in kilometers
    return R * c

def load_and_prepare_data():
    """Loads the dataset and prepares ALL variables (including Text & Geo-Features)."""
    print("Loading cleaned dataset and processing ALL variables...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "listings_clean.csv")
    
    # MARKET DEFINITION
    # exclusion of luxury outliers 
    df = df[df["price"] <= 800]
    
    # Coordinates of Singapore's economic epicenter (Marina Bay/Downtown Core)
    CENTER_LAT = 1.2823
    CENTER_LON = 103.8584
    
    if "latitude" in df.columns and "longitude" in df.columns:
        print("Calculating Haversine distance to Singapore Downtown Core...")
        # generating a new feature: distance to the economic center
        df["distance_to_center"] = haversine_distance(
            df["latitude"], df["longitude"], CENTER_LAT, CENTER_LON
        )
    else:
        print("Warning: Latitude/Longitude not found. Distance calculation skipped.")
    
    # separating target variable (y) and feature matrix (X)
    y = df["price"]
    X_raw = df.drop(columns=["price"], errors="ignore")
    
    # CATEGORICAL FEATURES: One-Hot Encoding
    # drop_first = True avoids perfect multicollinearity 
    categorical_cols = ["room_type", "neighbourhood_group_cleansed"]
    categorical_cols = [c for c in categorical_cols if c in X_raw.columns]
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
    
    # NUMERICAL FEATURES: Isolate and fill NaNs with mean values
    X_numeric = X_encoded.select_dtypes(include=[np.number]).fillna(
        X_encoded.select_dtypes(include=[np.number]).median()
    )
    
    # TEXT FEATURE ENGINEERING
    text_cols = ["name", "description", "neighborhood_overview", "space", "summary"]
    text_cols = [c for c in text_cols if c in X_raw.columns]
    
    text_features_list = []
    if text_cols:
        print(f"Processing text columns via TF-IDF: {text_cols}")
        for col in text_cols:
            # Replace NaNs with empty strings to prevent type conversion errors
            text_series = X_raw[col].fillna("").astype(str)
            # computes relative word frequeny adjusted for stop words
            tfidf = TfidfVectorizer(max_features=20, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(text_series)
            # uniquely naming the new features for subsequent model interpreation
            feature_names = [f"text_{col}_{name}" for name in tfidf.get_feature_names_out()]
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=X_raw.index)
            text_features_list.append(tfidf_df)
    
    # FEATURE CONCATENATION: Merging features       
    if text_features_list:
        X_all = pd.concat([X_numeric] + text_features_list, axis=1)
    else:
        X_all = X_numeric
        
    return X_all, y

def train_baseline():
    """Trains the Random Forest Model using restrictive hyperparameters
    and performs a detailed evaluation of its generalization capabilities."""
    X, y = load_and_prepare_data()
    
    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    
    print(f"Training on {X_train.shape[0]} rows with {X_train.shape[1]} features...")
    
    # OPTIMIZED, REGULARIZED RANDOM FOREST CONFIGURATION
    model = RandomForestRegressor(
        n_estimators=50,          
        max_depth=15,             
        min_samples_leaf=2,       
        max_features='sqrt', 
        random_state=123, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions for both training and test sets
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    # PERFORMANCE METRICS - TEST DATASET
    mae = mean_absolute_error(y_test, preds_test)
    rmse = root_mean_squared_error(y_test, preds_test)
    r2_test = r2_score(y_test, preds_test)
    
    # OVERFITTING DIAGNOSIS - TRAINING DATASET
    r2_train = r2_score(y_train, preds_train)
    overfitting_gap = r2_train - r2_test
    
    # OUTPUT
    print("\n" + "="*40)
    print("  BALANCED MODEL EVALUATION (WITH GEO-FEATURES)")
    print("="*40)
    print(f"MAE (Test)             : {mae:.2f} SGD")
    print(f"RMSE (Test)            : {rmse:.2f} SGD")
    print("-" * 40)
    print(f"Training R²    : {r2_train:.4f}")
    print(f"Test R²          : {r2_test:.4f}")
    print(f"Overfitting Gap: {overfitting_gap:.4f}")
    print("="*40)
    
    if overfitting_gap < 0.12:
        print("[NOTICE] Perfect! Regularization is working. The gap is stable.")
    else:
        print("[NOTICE] Analyze the metrics in comparison to the previous run.")

    # Generate and save plots
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds_test, alpha=0.4, color='teal')
    plt.plot([0, 800], [0, 800], color='darkred', linestyle='--', linewidth=2)
    plt.title(f"Actual vs. Predicted Prices (Test R²: {r2_test:.3f})")
    plt.xlabel("Actual Prices (SGD)")
    plt.ylabel("Predicted Prices (SGD)")
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    plt.grid(True, alpha=0.3)
    
    plot_path = "final_model_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    train_baseline()