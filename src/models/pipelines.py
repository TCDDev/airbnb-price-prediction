"""
Shared model pipelines

Currently:
- Median imputation only

TODO:
- Add ColumnTransformer + OneHotEncoder
- Nothing else (for now 👁️)
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.data.load import load_features_and_target

X, y = load_features_and_target()

def build_preprocessor(numerical_cols : list[str],
                       categorical_cols : list[str],
                       boolean_cols: list[str]) -> ColumnTransformer:

    """
    Shared Preprocessing Pipeline

    TODO:
    - Add numeric and categorical feature selection
    - Median-impute numerical features
    - One-hot encode categorical features

    Model preprocessing only, so...

    - No TF-IDF 🙂
    - No Haversine 🙂
    - No feature generation 🙂

    Those were all already handled upstream in src.data.preprocess
    """

    return "passthrough" # placeholder

def build_linear_pipeline() -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(...)), 
                     ("model", 
                      LinearRegression())])

def build_random_forest_pipeline() -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(...)), 
                     ("model", RandomForestRegressor(n_estimators=100, 
                                                     random_state=123, 
                                                     n_jobs=1))])
                                                     
