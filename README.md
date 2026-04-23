# UGOE ML Course

_Liam Lang · Louisa Woop · Tahir Can Dermanlı_ 

## Structure

* cpp/ → C++ code exposed via pybind11, callable from Python (⚠️ advanced / optional)
* src/ → reusable code
* notebooks/ → experiments
* data/ → datasets (not tracked)

---

## Notes

* Jupyter notebooks use Plotly for interactive plots.
* These plots may not render correctly in GitHub’s preview.
* For full visualization, open notebooks locally in VSCode or Jupyter.

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
py -3.14 -m venv .venv
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

## Notebook Guidelines

* **Avoid editing the same notebook at the same time**
* Always run: **Restart Kernel → Run All** before committing
* Jupyter notebooks can cause merge conflicts — prefer one person editing at a time

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

### Merge back into main

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


## C++ Extension (optional)

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
* VSCode may show pybind11 include/package warnings — these can be ignored if the build succeeds.

---

## Exposing C++ functions to Python with pybind11

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
