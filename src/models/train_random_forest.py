from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px # use either this or plt, whichever one you prefer
import pandas as pd

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

    """
    TODO:
    - Build pipeline
    - Fit model
    - Generate predictions
    - Evaluate metrics
    - Save figures and metrics to appropriate directories
    """

if __name__ == "__main__":
    train()