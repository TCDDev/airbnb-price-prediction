"""
Tabular feature engineering for Airbnb price prediction.

This module adds non-spatial features based on host information,
review activity, availability, and listing capacity.
"""

# TODO: Refactor file into a more atomic pipeline

# Hint: Since every single function will presumably reference it, making the reference_date a global variable
# is most likely going to be a slightly nicer and more readable design pattern. Python doesn't support global
# variables by default, but conventionally, these would be defined at the top of the file in all caps,
# i.e. "REFERENCE_DATE". The functions can simply reference it from the outer scope after that.


import numpy as np # Numpy remains unused, remove if you don't need it
import pandas as pd

def add_host_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    """
    Code Here
    """
    return df

def add_host_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    """
    Code Here
    """
    return df

def add_review_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    """
    Code Here
    """
    return df

def add_availability_ratio(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    """
    Code Here
    """
    return df

def add_capacity(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    """
    Code Here
    """
    return df

def add_room_type_indicators(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    """
    Code Here
    """
    return df

def broaden_property_types(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    """
    Code Here
    """
    return df

def add_tabular_features(
    df: pd.DataFrame,
    # reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Add tabular features to the cleaned Airbnb listings dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Listings dataframe after basic preprocessing.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional engineered tabular features.
    """
    # df = df.copy()
    # reference_date = pd.Timestamp(reference_date).normalize()

    # # Host tenure features
    # if "host_since" in df.columns:
    #     host_since = pd.to_datetime(df["host_since"], errors="coerce")
    #     host_tenure_days = (reference_date - host_since).dt.days

    #     df["host_tenure_years"] = host_tenure_days / 365.25

    # # Review activity features
    # if "number_of_reviews" in df.columns:
    #     df["log_number_of_reviews"] = np.log1p(df["number_of_reviews"])
    #     df["has_reviews"] = (df["number_of_reviews"].fillna(0) > 0).astype(int)

    # if "reviews_per_month" in df.columns:
    #     df["log_reviews_per_month"] = np.log1p(df["reviews_per_month"].fillna(0))

    # if "last_review" in df.columns:
    #     last_review = pd.to_datetime(df["last_review"], errors="coerce")

    #     df["days_since_last_review"] = (reference_date - last_review).dt.days
    #     df["has_recent_review"] = (df["days_since_last_review"] <= 180).astype(int)
    #     df.loc[df["days_since_last_review"].isna(), "has_recent_review"] = 0

    # # Availability features (ratios)
    # if "availability_30" in df.columns:
    #     df["availability_ratio_30"] = df["availability_30"] / 30

    # if "availability_60" in df.columns:
    #     df["availability_ratio_60"] = df["availability_60"] / 60

    # if "availability_90" in df.columns:
    #     df["availability_ratio_90"] = df["availability_90"] / 90

    # if "availability_365" in df.columns:
    #     df["availability_ratio_365"] = df["availability_365"] / 365
    #     df["high_availability"] = (df["availability_365"] > 180).astype(int)
    #     df["low_availability"] = (df["availability_365"] < 30).astype(int)

    # # Capacity / listing-size features
    # if {"beds", "accommodates"}.issubset(df.columns):
    #     df["beds_per_guest"] = df["beds"] / df["accommodates"].replace(0, np.nan)

    # if {"bedrooms", "accommodates"}.issubset(df.columns):
    #     df["bedrooms_per_guest"] = df["bedrooms"] / df["accommodates"].replace(0, np.nan)

    # if {"bathrooms", "accommodates"}.issubset(df.columns):
    #     df["bathrooms_per_guest"] = df["bathrooms"] / df["accommodates"].replace(0, np.nan)

    # # Room type indicators
    # if "room_type" in df.columns:
    #     df["is_entire_home"] = (df["room_type"] == "Entire home/apt").astype(int)
    #     df["is_private_room"] = (df["room_type"] == "Private room").astype(int)
    #     df["is_shared_room"] = (df["room_type"] == "Shared room").astype(int)

    # # Group rare property types into broader categories
    # if "property_type" in df.columns:
    #     property_type = df["property_type"].astype(str).str.lower()

    #     grouped_property_type = pd.Series("other", index=df.index)

    #     grouped_property_type.loc[
    #         property_type.str.contains("apartment|rental unit", na=False)
    #     ] = "apartment"

    #     grouped_property_type.loc[
    #         property_type.str.contains("condo|condominium", na=False)
    #     ] = "condo"

    #     grouped_property_type.loc[
    #         property_type.str.contains("house|home", na=False)
    #     ] = "house"

    #     grouped_property_type.loc[
    #         property_type.str.contains("hotel|hostel", na=False)
    #     ] = "hotel"

    #     grouped_property_type.loc[
    #         property_type.str.contains("serviced", na=False)
    #     ] = "serviced_apartment"

    #     property_type_dummies = pd.get_dummies(
    #         grouped_property_type,
    #         prefix="property_type",
    #         dtype=int,
    #     )

    #     df = pd.concat([df, property_type_dummies], axis=1)

    return df

if __name__ == "__main__":
# Use this for quick testing/debugging. Remove the pass statement when you actually write code into here
    pass