"""
Dataset loading utilities.

TODO:
- Load listings
- Load reviews
- Load calendar data
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_raw_listings(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DATA_DIR / "listings.csv"
    return pd.read_csv(path)

def load_neighborhoods(path: Path | None = None) -> gpd.DataFrame:
    path = path or RAW_DATA_DIR / "neighbourhoods.geojson"
    return gpd.read_file(path)

def load_processed_listings(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_DATA_DIR / "listings_clean.csv"
    return pd.read_csv(path)