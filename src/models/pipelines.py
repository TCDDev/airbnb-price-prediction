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

# Convenience copy for schema inspection / debugging
# Training pipeline should receive X and y inside the training script
# The correct thing to include from src.data.load in this case would be
# from src.data.load import load_features_and_target

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

    Hint: Start by using the pandas DataFrame's columns.tolist() method
    in the outer scope to first create the required lists, then parse
    them into the function. You can also create a helper function which
    returns the required lists in a list or tuple, and then parse that
    in using the unpacking operator (*).
    """

    return "passthrough" # placeholder, finished function should return a ColumnTransformer object

# Functions below use a placeholder for the build_preprocessor variables
# Will not run as is

def build_linear_pipeline() -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(...)), 
                     ("model", 
                      LinearRegression())])

def build_random_forest_pipeline() -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(...)), 
                     ("model", RandomForestRegressor(n_estimators=100, 
                                                     random_state=123, 
                                                     n_jobs=-1))])
                                                     

if __name__ == "__main__":
    from src.data.load import load_processed_listings
    df = load_processed_listings().copy()

    # Use this for quick testing/debugging. Remove the pass statement when you actually write code into here
    print(df.dtypes)