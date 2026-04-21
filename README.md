# UGOE ML Course

## Structure
- cpp/ → C++ code exposed via pybind11, callable from Python (⚠️ advanced / optional)
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
  * Python (Microsoft)
  * Jupyter (Microsoft)
* Git (Obviously)
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
* Use English for variable and function names

---

## Exposing C++ functions to Python with pybind11

The `cpp/` directory contains a minimal C++ extension built with `pybind11`, callable from Python as `fast_module`.

### Workflow for adding a new C++ function

1. **Write the function** in `cpp/fast_module.cpp`

```cpp id="3b9vkn"
double power(double base, double exponent) {
    return std::pow(base, exponent);
}
```

2. **Declare it** in `cpp/bindings.cpp`

```cpp id="ivc0sw"
double power(double base, double exponent);
```

3. **Expose it to Python** in `cpp/bindings.cpp`

```cpp id="6fgu0v"
m.def("power", &power, "Raise a number to a power");
```

4. **Rebuild the extension**

```bash id="ztm748"
cd cpp
python -m pip install -e .
```

5. **Restart the notebook kernel** and import it from Python

```python id="gq98qu"
import fast_module
print(fast_module.power(2.0, 3.0))
```

### Notes

* If the build succeeds, VSCode include/package warnings in `cpp/` can usually be ignored.
* After changing C++ code, always rebuild and restart the notebook kernel.
* The basic pattern is:

```text id="4rnif0"
write function → declare in bindings → m.def(...) → rebuild → restart kernel
```


### Notes

* Plotly plots may not render correctly on GitHub — open notebooks locally for full output
* Always run:
* Jupyter notebooks can cause merge conflicts. Prefer one person editing a notebook at a time
* Restart kernel and run all cells before committing

  * **Restart Kernel → Run All** before committing notebooks
