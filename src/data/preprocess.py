"""
Data preprocessing utilities.

TODO:
- Missing value handling
- Data cleaning
- Feature preparation
"""

import pandas as pd

from src.data.load import load_raw_listings
from src.utils.paths import PROCESSED_DATA_DIR

DROP_COLUMNS = [
    # ID / URLs
    "id",
    "listing_url",
    "scrape_id",
    "picture_url",
    "host_id",
    "host_url",
    "host_name",
    "host_thumbnail_url",
    "host_picture_url",

    # Text / NLP data
    "name",
    "description",
    "neighborhood_overview",
    "host_about",
    "amenities",

    # Leakage
    "estimated_occupancy_l365d",
    "estimated_revenue_l365d",

    # Scrape Metadata
    "last_scraped",
    "calendar_last_scraped",
    "calendar_updated",
    "source",
    
    # Further feature selection
    "host_location",
    "host_neighbourhood",
    "host_verifications",
    "license",
    "neighbourhood",
    "bathrooms_text",
]

BOOLEAN_COLUMNS = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "has_availability",
    "instant_bookable",
]

PERCENTAGE_COLUMNS = [
    "host_response_rate",
    "host_acceptance_rate",
]


def clean_price(df:pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["price"]).copy() # Avoids SettingWithCopyWarning

    df["price"] = (
        df["price"].str.replace("$", "", regex=False) \
        .str.replace(",", "", regex=False) \
        .astype(float)       
    )

    return df

def drop_unused_columns(df:pd.DataFrame) -> pd.DataFrame:
    existing = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=existing)


def convert_booleans(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"t": True, "f": False}

    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(False)

    return df

def convert_host_response_time(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        # 0-3 scale, with a lower number equalling a faster response time by the host
        "within an hour" : 0,
        "within a few hours" : 1,
        "within a day" : 2,
        "a few days or more" : 3 
    }

    df["host_response_time"] = (
        df["host_response_time"].map(mapping)
    )

    return df

def clean_percentages(df: pd.DataFrame) -> pd.DataFrame:
    for col in PERCENTAGE_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col].str.replace("%", "", regex=False) \
                .astype(float) \
                / 100
            )
    
    return df

def preprocess_listings(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price(df)
    df = drop_unused_columns(df)
    df = convert_booleans(df)
    df = convert_host_response_time(df)
    df = clean_percentages(df)
    return df

def main() -> None:
    df = load_raw_listings()
    cleaned = preprocess_listings(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DATA_DIR / "listings_clean.csv", index=False)

if __name__ == "__main__":
    main()
