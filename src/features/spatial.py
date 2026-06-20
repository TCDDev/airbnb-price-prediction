"""
Spatial feature engineering for the Airbnb Singapore listings dataset.

This module generates location-based features used by the machine learning
pipeline. These features are derived from external geospatial datasets and
are appended to the cleaned listings dataset during preprocessing

Implemented features:
    - Neighbourhood density
        Number of Airbnb listings per square kilometre within each
        neighbourhood

    - Distance to nearest rail station
        Great-circle distance (km) from a listing to the nearest
        MRT/LRT station

    - Distance to CBD
        Great-circle distance (km) from a listing to Singapore's
        Central Business District

    - Distance to Changi Airport
        Great-circle distance (km) from a listing to Changi Airport

Notes:
    - Area calculations are performed in EPSG:3414 (SVY21), Singapore's
      projected coordinate reference system
    - Distance calculations use the Haversine formula on WGS84
      latitude/longitude coordinates (EPSG:4326)
    - MRT/LRT stations are represented by a single centroid point
      computed from all station exits
"""

# 👀 OpenMP candidate

import pandas as pd
import geopandas as gpd
from haversine import haversine
from src.data.load import load_processed_listings, load_neighbourhoods, load_rail_stations

CBD = (1.283, 103.851) # Approximate center of Singapore's Central Business District
CHANGI_AIRPORT = (1.3644, 103.9915) # Approximate centre point of Singapore Changi Airport.

def add_neighbourhood_density(df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Add AirBnB listing density per neighborhood

    Density is calculated with the following formula:
    No. of listings in the neighborhood / Neighborhood's area in km²
    """
    df = df.copy()
    gdf = gdf.copy()

    gdf_metric = gdf.to_crs("EPSG:3414")

    gdf_metric["area_km2"] = (gdf_metric.geometry.area / 1_000_000)

    listing_counts = (df["neighbourhood_cleansed"].value_counts())

    gdf_metric["listing_counts"] = (gdf_metric["neighbourhood"].map(listing_counts).fillna(0))

    gdf_metric["neighbourhood_density"] = (gdf_metric["listing_counts"] / gdf_metric["area_km2"])

    density_map = (gdf_metric.set_index("neighbourhood")["neighbourhood_density"])
    
    df["neighbourhood_density"] = (df["neighbourhood_cleansed"].map(density_map))

    return df


def build_station_points(rail_exits: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Collapse MRT/LRT exists into one representative point per station"""

    rail_metric = rail_exits.to_crs("EPSG:3141")

    station_points = rail_metric.dissolve(by="STATION_NA")
    station_points["geometry"] = station_points.geometry.centroid

    station_points = (station_points.to_crs("EPSG:4326").reset_index()[["STATION_NA", "geometry"]])

    return station_points

def add_distance_to_nearest_rail(df: pd.DataFrame, rail_exits: gpd.GeoDataFrame) -> pd.DataFrame:
    """Add distance to the nearest MRT/LRT station in km"""

    df = df.copy()

    station_points = build_station_points(rail_exits)

    stations = [(point.y, point.x) for point in station_points.geometry]

    def nearest_station_distance(row: pd.Series) -> float:
        listing_point = (row["latitude"], row["longitude"])

        return min(haversine(listing_point, station) for station in stations)
    
    df["distance_to_nearest_rail"] = df.apply(nearest_station_distance, axis=1)

    return df

def add_distance_to_point(df: pd.DataFrame, point: tuple[float, float], column_name: str) -> pd.DataFrame:
    """Adds distance to a pre-defined point"""
    
    df = df.copy()

    df[column_name] = df.apply(lambda row: haversine((row["latitude"], row["longitude"]), point), axis=1)

    return df

if __name__ == "__main__":
    """Testing implementation"""
    listings = load_processed_listings()
    neighbourhoods = load_neighbourhoods()
    rail_stations = load_rail_stations()

    result = add_neighbourhood_density(
        listings,
        neighbourhoods
    )

    rail_result = add_distance_to_nearest_rail(listings, rail_stations)

    point_result = add_distance_to_point(listings, CBD, "distance_to_cbd")

    # print(result[["neighbourhood_cleansed", "neighbourhood_density"]].head())
    # print(rail_result["distance_to_nearest_rail"].describe())
    # print("isna:", rail_result["distance_to_nearest_rail"].describe().isna().sum())
    # print(point_result["distance_to_cbd"].describe())
    # print("isna: ", point_result["distance_to_cbd"].isna().sum())
    