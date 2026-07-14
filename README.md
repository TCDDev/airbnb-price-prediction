# UGOE ML Course

**Liam Lang · Louisa Woop · Tahir Can Dermanlı**

Machine Learning course project focused on predicting Airbnb listing prices in Singapore using tabular and geospatial data.

---

# Results

| Model | MAE | RMSE | R² |
|------|-----:|------:|----:|
| Ridge Regression | 267.33 | 496.96 | 0.62 |
| Random Forest | **106.19** | **329.97** | **0.83** |
| XGBoost | 112.41 | 356.87 | 0.80 |

### Summary

- Random Forest achieved the best predictive performance after hyperparameter tuning.
- XGBoost performed similarly, with only a small decrease in predictive accuracy.
- Both tree-based models substantially outperformed the linear baseline.

---

# Repository Structure

```
src/
├── data/          # Loading and preprocessing
├── features/      # Feature engineering
├── models/        # Model pipelines and training
├── evaluation/    # Evaluation utilities
└── utils/         # Shared utilities

data/
├── raw/
├── processed/
└── external/

results/
├── figures/
└── metrics/

cpp/
└── Optional pybind11 extension
```

---

# Project Pipeline

```
Raw Airbnb Dataset
        │
        ▼
Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Processed Dataset
        │
        ▼
Shared Scikit-Learn Pipeline
        │
  ┌─────┼──────────┐
  ▼     ▼          ▼
Ridge  Random Forest  XGBoost
        │
        ▼
Evaluation
```

---

# Dataset Setup

Download the required Airbnb dataset files:

- `listings.csv.gz` (extract before use)
- `neighbourhoods.geojson`

Place both files inside:

```
data/raw/
```

Download the MRT/LRT station exit dataset from the Singapore Open Data Portal and place it inside:

```
data/external/
```

---

# Generating the Processed Dataset

Run:

```bash
python -m src.data.preprocess
```

The preprocessing pipeline:

- removes missing price values
- converts price columns to numeric format
- removes leakage-related features
- converts Airbnb boolean fields
- encodes host response time
- converts percentage features
- appends engineered spatial features

The resulting dataset is written to:

```
data/processed/listings_clean.csv
```

---

# Training Models

Each model can be trained independently.

```bash
python -m src.models.linreg_train
```

```bash
python -m src.models.random_forest_train
```

```bash
python -m src.models.xgboost_train
```

Evaluation metrics and figures are written to:

```
results/
├── figures/
└── metrics/
```

---

# Optional C++ Extension

The repository contains an optional `pybind11` extension inside the `cpp/` directory.

It was used to evaluate native implementations of selected spatial computations.

To build:

```bash
cd cpp
python -m pip install -e .
```

The project functions correctly without compiling this extension.

---

# Notes

- Raw and processed datasets are intentionally excluded from version control.
- Generated figures and evaluation metrics are written to the `results/` directory.
- The reported model results were generated using the repository's shared preprocessing pipeline to ensure consistent evaluation across all models.