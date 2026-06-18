"""
Spatial feature engineering.

TODO:
- Distance to city center
- Neighborhood density
- Spatial clustering features
"""

# 👀 OpenMP candidate

import pandas as pd
import geopandas as gpd
from haversine import haversine
from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

CBD = (1.283, 103.851) # Approximate center of Singapore's Central Business District
CHANGI_AIRPORT = (1.3644, 103.9915)

def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    pass

def add_neighborhood_area(listings: pd.DataFrame, neighborhoods: gpd.GeoDataFrame) -> pd.DataFrame:
    pass