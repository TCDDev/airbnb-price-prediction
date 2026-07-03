"""
Shared model pipelines

Currently:
- Working pipeline for LinReg and Random Forest models

TODO:
- Add pipeline for XGBoost
- Nothing else (for now 👁️)
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor



def build_preprocessor(numerical_cols : list[str],
                       categorical_cols : list[str],
                       boolean_cols : list[str]) -> ColumnTransformer:

    """
    Shared Preprocessing Pipeline

    Used jointly by all ML models within the project

    Responsibilities:
    - Median-impute numerical features
    - One-hot encode categorial features
    - Pass boolean features through unchanged

    Feature engineering and dataset cleaning are intentionally excluded because they're handled upstream
    """

    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    # bool_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols), 
        ('cat', cat_transformer, categorical_cols),
        ('bool', "passthrough", boolean_cols) # Boolean features appear to already be clean after upstream preprocessing, pass them through unchanged 
    ],
    remainder='drop' 
)
    return preprocessor 


def build_linear_pipeline(numerical_cols: list[str], categorical_cols: list[str], boolean_cols: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(numerical_cols, categorical_cols, boolean_cols)), 
        ("model", LinearRegression())
    ])

def build_random_forest_pipeline(numerical_cols: list[str], categorical_cols: list[str], boolean_cols: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(numerical_cols, categorical_cols, boolean_cols)), 
        ("model", RandomForestRegressor(n_estimators=100, 
                                        random_state=123, 
                                        n_jobs=-1))
    ])

def build_xgboost_pipeline() -> Pipeline:
    """Temporary Placeholder"""
    pass
                                                     

if __name__ == "__main__":
    # Convenience copy for schema inspection / debugging
    # Training pipeline should receive X and y inside the training script
    # The correct thing to include from src.data.load in this case would be
    # from src.data.load import load_features_and_target
    
    from src.data.load import load_processed_listings
    df = load_processed_listings().copy()
    
    X = df.drop(columns=["price"], errors="ignore")
    
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    boolean_cols = X.select_dtypes(include=['bool']).columns.tolist()

    print(f"Numerical ({len(numerical_cols)}):", numerical_cols)
    print(f"Categorical ({len(categorical_cols)}):", categorical_cols)
    print(f"Boolean ({len(boolean_cols)}):", boolean_cols)

    rf_pipeline = build_random_forest_pipeline(numerical_cols, categorical_cols, boolean_cols)
    
    print(df.dtypes)