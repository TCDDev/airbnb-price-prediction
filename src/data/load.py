"""
Data loading utilities for the Airbnb Singapore project.

This module centralizes dataset loading operations and provides
convenience functions for reading raw tabular and spatial datasets.

Functions
---------
load_raw_listings()
    Load the raw Airbnb listings dataset.

load_neighbourhoods()
    Load the Singapore neighborhood GeoJSON dataset.

load_rail_stations()
    Load the external rail exit dataset from the external GeoJSON

load_processed_listings()
    Load the processed listing dataset

load_features_and_target()
    Convenience function that loads both the feature and target variables

Notes
-----
Dataset paths are managed through src.utils.paths to ensure
consistent file access across notebooks and scripts.
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR

def load_raw_listings(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DATA_DIR / "listings.csv"
    return pd.read_csv(path)

def load_neighbourhoods(path: Path | None = None) -> gpd.GeoDataFrame:
    path = path or RAW_DATA_DIR / "neighbourhoods.geojson"
    return gpd.read_file(path)

def load_rail_stations(path: Path | None = None) -> gpd.GeoDataFrame:
    path = path or EXTERNAL_DATA_DIR / "LTAMRTStationExitGEOJSON.geojson"
    return gpd.read_file(path)

def load_processed_listings(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_DATA_DIR / "listings_clean.csv"
    return pd.read_csv(path)

def load_features_and_target(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    path = path or PROCESSED_DATA_DIR / "listings_clean.csv"
    df = pd.read_csv(path)
    y = df["price"]
    X = df.drop(columns=["price"])

    return X, y