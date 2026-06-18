"""
Shared project paths.

ALlows for ergonomic importing of file paths.

RAW_DATA_DIR -> Location of the raw AirBnB data
PROCESSED_DATA_DIR -> Location of the preprocessed datasets
EXTERNAL_DATA_DIR -> Location of any external (i.e. non-AirBnB) data
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

# Import the ones below for your needs

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"