"""
Tabular feature engineering for Airbnb price prediction.

This module adds non-spatial features based on host information,
review activity, availability, and listing capacity.
"""

import numpy as np
import pandas as pd


# Added fixed dataset snapshot date.
# -> based on max(last_scraped) from data/raw/listings.csv
REFERENCE_DATE = pd.Timestamp("2025-09-28")


def add_host_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Currently no additional host activity features here
    """
    return df


def add_review_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add review count, review frequency, and review recency features
    """
    df = df.copy()

    if "number_of_reviews" in df.columns:
        df["log_number_of_reviews"] = np.log1p(df["number_of_reviews"])
        df["has_reviews"] = (df["number_of_reviews"].fillna(0) > 0).astype(int)

    if "reviews_per_month" in df.columns:
        df["log_reviews_per_month"] = np.log1p(df["reviews_per_month"].fillna(0))

    if "last_review" in df.columns:
        last_review = pd.to_datetime(df["last_review"], errors="coerce")

        df["days_since_last_review"] = (REFERENCE_DATE - last_review).dt.days
        df["has_recent_review"] = (df["days_since_last_review"] <= 180).astype(int)
        df.loc[df["days_since_last_review"].isna(), "has_recent_review"] = 0

    return df


def add_availability_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standardized availability ratios and availability indicators
    """
    df = df.copy()

    if "availability_30" in df.columns:
        df["availability_ratio_30"] = df["availability_30"] / 30

    if "availability_60" in df.columns:
        df["availability_ratio_60"] = df["availability_60"] / 60

    if "availability_90" in df.columns:
        df["availability_ratio_90"] = df["availability_90"] / 90

    if "availability_365" in df.columns:
        df["availability_ratio_365"] = df["availability_365"] / 365
        df["high_availability"] = (df["availability_365"] > 180).astype(int)
        df["low_availability"] = (df["availability_365"] < 30).astype(int)

    return df


def add_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add capacity-normalized listing size features
    """
    df = df.copy()

    if {"beds", "accommodates"}.issubset(df.columns):
        df["beds_per_guest"] = df["beds"] / df["accommodates"].replace(0, np.nan)

    if {"bedrooms", "accommodates"}.issubset(df.columns):
        df["bedrooms_per_guest"] = (
            df["bedrooms"] / df["accommodates"].replace(0, np.nan)
        )

    if {"bathrooms", "accommodates"}.issubset(df.columns):
        df["bathrooms_per_guest"] = (
            df["bathrooms"] / df["accommodates"].replace(0, np.nan)
        )

    return df


def add_room_type_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary indicators for common Airbnb room types
    """
    df = df.copy()

    if "room_type" in df.columns:
        df["is_entire_home"] = (df["room_type"] == "Entire home/apt").astype(int)
        df["is_private_room"] = (df["room_type"] == "Private room").astype(int)
        df["is_shared_room"] = (df["room_type"] == "Shared room").astype(int)

    return df


def broaden_property_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group detailed property types into broader categories and one-hot encode them
    """
    df = df.copy()

    if "property_type" in df.columns:
        property_type = df["property_type"].astype(str).str.lower()

        grouped_property_type = pd.Series("other", index=df.index)

        grouped_property_type.loc[
            property_type.str.contains("apartment|rental unit", na=False)
        ] = "apartment"

        grouped_property_type.loc[
            property_type.str.contains("condo|condominium", na=False)
        ] = "condo"

        grouped_property_type.loc[
            property_type.str.contains("house|home", na=False)
        ] = "house"

        grouped_property_type.loc[
            property_type.str.contains("hotel|hostel", na=False)
        ] = "hotel"

        grouped_property_type.loc[
            property_type.str.contains("serviced", na=False)
        ] = "serviced_apartment"

        property_type_dummies = pd.get_dummies(
            grouped_property_type,
            prefix="property_type",
            dtype=int,
        )

        expected_dummy_columns = [
            "property_type_apartment",
            "property_type_condo",
            "property_type_hotel",
            "property_type_house",
            "property_type_other",
            "property_type_serviced_apartment",
        ]

        property_type_dummies = property_type_dummies.reindex(
            columns=expected_dummy_columns,
            fill_value=0,
        )

        df = pd.concat([df, property_type_dummies], axis=1)

    return df


def add_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all tabular features to the cleaned Airbnb listings dataset

    Parameters
    ----------
    df : pd.DataFrame
        Listings dataframe after basic preprocessing

    Returns
    -------
    pd.DataFrame
        Dataframe with additional engineered tabular features
    """
    df = df.copy()

    df = add_host_tenure_features(df)
    df = add_host_activity_features(df)
    df = add_review_activity_features(df)
    df = add_availability_ratio(df)
    df = add_capacity(df)
    df = add_room_type_indicators(df)
    df = broaden_property_types(df)

    return df


if __name__ == "__main__":
    pass
