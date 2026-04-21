# UGOE ML Course

## Structure
- src/ → reusable code
- notebooks/ → experiments
- data/ → datasets (not tracked)

## Notes

* Jupyter notebooks use Plotly for interactive plots.
* These plots may not render correctly in GitHub’s preview.
* For full visualization, open notebooks locally in VSCode or Jupyter.

## Setup

### 1. Install prerequisites

* Python (3.11+ recommended)
* VSCode
* VSCode extensions:
* Git

  * Python (Microsoft)
  * Jupyter (Microsoft)

---

### 2. Clone the repository

```bash
git clone https://github.com/TCDDev/UGOE-ML-Course.git
cd UGOE-ML-Course
```

---

### 3. Create a virtual environment

```bash
py -3.14 -m venv .venv
```

---

### 4. Activate the environment

**Windows (PowerShell):**

```bash
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

### Notes

* Plotly plots may not render correctly on GitHub — open notebooks locally for full output
* Always run:

  * **Restart Kernel → Run All** before committing notebooks
