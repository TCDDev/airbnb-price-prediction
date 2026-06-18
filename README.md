# UGOE ML Course

*Liam Lang · Louisa Woop · Tahir Can Dermanlı*

Machine Learning course project focused on Airbnb price prediction.

The goal is to build reproducible machine learning pipelines using multiple data modalities (tabular, text, image, or spatial data) and compare different predictive approaches.

---

## Setup

### 1. Install prerequisites

* Python (3.11+ recommended)
* VSCode
* VSCode extensions:

  * Python (Microsoft)
  * Jupyter (Microsoft)
* Git

---

### 2. Clone the repository

```bash
git clone https://github.com/TCDDev/UGOE-ML-Course.git
cd UGOE-ML-Course
```

---

### 3. Create a virtual environment

```bash
python -3.14 -m venv .venv
```

---

### 4. Activate the environment

#### Windows (PowerShell)

```powershell
& .\.venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
& .\.venv\Scripts\Activate.ps1
```

---

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

### 6. Open in VSCode

* Open the folder in VSCode
* Select the interpreter:

  * `Ctrl + Shift + P` → **Python: Select Interpreter**
  * Choose `.venv`

---

### 7. Run notebooks

* Open a notebook in `notebooks/`
* Select the `.venv` kernel (top right)
* Run cells normally

---

## Git Workflow

### Basic rule

* Create a branch for any non-trivial work
* Avoid committing directly to `main` when working in parallel

---

### Create a branch

Before starting work:

```bash
git checkout main
git pull
git checkout -b feature/your-name-task
```

---

### Work and commit

After making changes:

```bash
git add .
git commit -m "Describe your changes"
```

---

### Push your branch

```bash
git push -u origin feature/your-name-task
```

After the first push, you can simply use:

```bash
git push
```

---

### Merging back into main

Do not merge directly into `main`.

When your work is finished:

1. Push your branch
2. Open a Pull Request on GitHub
3. Wait for review/approval
4. Merge only after approval

So:

```bash
git checkout main
git pull
git checkout -b feature/your-task 
git push
```

Then:

```bash
git add .
git commit -m "Describe your changes"
git push -u origin feature/your-task
```

`main` is protected. Direct commits to `main` are not allowed.

---

### Team guidelines

* Use branches for assignments and features
* Always `git pull` before starting work
* **Avoid editing the same notebook at the same time**
* Use clear commit messages
* Use English for variable and function names

---

## Notebook Guidelines

* **Avoid editing the same notebook at the same time**
* Always run: **Restart Kernel → Run All** before committing
* Jupyter notebooks can cause merge conflicts. Prefer one person editing at a time

---

## Repository Structure

### Source Code

* src/data/ → data loading and preprocessing
* src/features/ → feature engineering
* src/models/ → model training
* src/evaluation/ → evaluation utilities
* src/utils/ → shared helper functions

### Notebooks

* notebooks/exploration/ → exploratory analysis
* notebooks/reports/ → presentation-ready notebooks

### Data

* data/raw/ → original downloaded datasets (not tracked)
* data/processed/ → cleaned datasets (not tracked)

### Results

* results/figures/ → generated plots
* results/metrics/ → model evaluation results

### Optional Components

* cpp/ → C++ code exposed via pybind11, callable from Python (⚠️ advanced / optional)

---

## Development Guidelines

* Reusable code belongs in `src/`
* Exploratory work belongs in `notebooks/`
* Large datasets should not be committed to Git
* Use English for variable and function names
* Profile before optimizing

---

## Data

Datasets are stored locally under:

```text
data/raw/
data/processed/
```

These folders are intentionally excluded from Git.

Each team member should download the required datasets locally.

---

## Notes

* Jupyter notebooks use Plotly for interactive plots.
* These plots may not render correctly in GitHub’s preview.
* For full visualization, open notebooks locally in VSCode or Jupyter.

---

## Dataset Setup

Download the Airbnb dataset files provided for the project and extract:

- `listings.csv`
- `neighbourhoods.geojson`

Place both files in:

data/raw/

Expected directory structure:

data/
├── raw/
│   ├── listings.csv
│   └── neighbourhoods.geojson
├── processed/
└── ...
Generating the Cleaned Dataset

Run:

`python -m src.data.preprocess`

This preprocessing pipeline:

- Removes listings with missing prices
- Converts price values to numeric format
- Removes unused and leakage-related features
- Converts Airbnb boolean values (`t` / `NaN`) to Python booleans
- Encodes host response time categories
- Converts percentage features to decimal values

The cleaned dataset will be generated at:

data/processed/listings_clean.csv

## Loading the Cleaned Dataset

Example:

```python
import pandas as pd
from src.utils.paths import PROCESSED_DATA_DIR

df = pd.read_csv(
    PROCESSED_DATA_DIR / "listings_clean.csv"
)
```
#### Notes
- `host_since` is retained for potential future feature engineering but is not currently transformed.
- Review-related missing values are preserved because they correspond to listings with no reviews.
- Numerical missing values are preserved for downstream handling during model development.

## Optional C++ Extension

The `cpp/` directory contains an optional C++ extension built with `pybind11`, callable from Python as `fast_module`.

If you want to use any functions from `fast_module`, the extension must be built locally.

### Requirements (Windows)

* Visual Studio Build Tools
* CMake

### Install

1. Install Visual Studio Build Tools:
   https://aka.ms/buildtools

   During installation, select:

   * **Desktop development with C++**

2. Install CMake:
   https://cmake.org/download/

---

### Build the extension

```bash
cd cpp
python -m pip install -e .
```

---

### Notes

* If you are not using any functions from `fast_module`, you can ignore the `cpp/` directory.
* If you are using `fast_module`, the C++ toolchain is required.
* VSCode may show pybind11 include/package warnings. These can be ignored if the build succeeds.

---

## Exposing C++ Functions to Python with pybind11

### Workflow for adding a new C++ function

1. **Write the function** in `cpp/fast_module.cpp`

```cpp
double power(double base, double exponent) {
    return std::pow(base, exponent);
}
```

2. **Declare it** in `cpp/bindings.cpp`

```cpp
double power(double base, double exponent);
```

3. **Expose it to Python** in `cpp/bindings.cpp`

```cpp
m.def("power", &power, "Raise a number to a power");
```

4. **Rebuild the extension**

```bash
cd cpp
python -m pip install -e .
```

5. **Restart the notebook kernel** and import it from Python

```python
import fast_module
print(fast_module.power(2.0, 3.0))
```

---

### Summary

```text
write function → declare in bindings → m.def(...) → rebuild → restart kernel
```
