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

## Git Workflow

### Basic rule

* Create a branch for any non-trivial work
* Avoid committing directly to `main` when working in parallel

---

### Daily workflow

Before starting work:

```bash
git pull
```

Create a branch:

```bash
git checkout -b feature/your-name-task
```

After making changes:

```bash
git add .
git commit -m "Describe your changes"
git push -u origin feature/your-name-task
```

---

### Merging back into main

When your work is finished:

```bash
git checkout main
git pull
git merge feature/your-name-task
git push
```

---

### Team guidelines

* Use branches for assignments and features
* Always `git pull` before starting work
* **Avoid editing the same notebook at the same time**
* Use clear commit messages

---

### Notes on notebooks

Jupyter notebooks can cause merge conflicts.

* Prefer one person editing a notebook at a time
* Restart kernel and run all cells before committing


### Notes

* Plotly plots may not render correctly on GitHub — open notebooks locally for full output
* Always run:

  * **Restart Kernel → Run All** before committing notebooks
