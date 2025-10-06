# Thermodynamic Systems: Modeling in Python

A compact collection of aerospace models **rockets, engines, and flame physics**; implemented as reproducible Jupyter notebooks with **plots**. This repo is ideal for students and practitioners who want clear, minimal models they can run and extend.

> **Scope cutoff:** Analyses summarized here reflect work available as of **2025-09-24**.

---

## 📦 Contents

Contents Include:
```

├── Rocket Ariane 6 Staged Mass Loss.ipynb
├── Engine GE90 Residence Time.ipynb
├── Rocket Falcon Heavy Optimal Throttle Schedules.ipynb
├── Rocket Staged and Throttled.ipynb
├── Engine Optimizing for Pressure Ratio.ipynb
├── Mechanism1.yaml (n-dodecane)
├── C7H16.yaml
└── README.md
```

---

## 🚀 Quickstart

```bash
# (optional) create an isolated environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

# install core packages
pip install -r requirements.txt

# launch Jupyter
jupyter lab  # or: jupyter notebook
```

> Some notebooks read mechanism files (`Mechanism1.yaml`, `C7H16.yaml`). If you move them, adjust paths in the notebooks.

---

## 📥 Getting the repository

Whether you prefer to work locally, in Colab, or on another hosted service, you
have a few options to bring the project onto your machine:

### Option A — Clone with Git (recommended)

```bash
git clone https://github.com/<your-user>/Thermodynamic_Models.git
cd Thermodynamic_Models
```

Cloning keeps the Git history so you can pull updates, push changes, and create
branches or pull requests. Replace `<your-user>` with the owner of your fork if
you are working from a personal copy.

### Option B — Download a ZIP snapshot

1. Navigate to the repository page in your browser.
2. Click **Code ▾ → Download ZIP**.
3. Extract the archive locally (macOS Finder, Windows Explorer, or `unzip` on
   the command line).
4. Open a terminal in the extracted folder and follow the [Quickstart](#-quickstart)
   steps to create an environment and install dependencies.

### Option C — Pull directly inside Google Colab

In a new Colab notebook, run:

```python
%cd /content
!git clone https://github.com/<your-user>/Thermodynamic_Models.git
%cd Thermodynamic_Models
```

This mirrors the workflow documented in the
[Google Colab section](#-running-the-full-workflow-in-google-colab) and is
useful when you want to keep everything in the cloud.

> **Tip:** if you are handed the files outside of Git, simply drop the folder in
> your working directory and run the same commands from the Quickstart. The
> models rely only on the Python source and notebooks present in the repository.

---

## 📚 Model Catalog (at a glance)

> Replace any placeholders below with specifics if your notebooks print/annotate them.

### `Rocket Ariane 6 Staged Mass Loss.ipynb`
- **System:** Ariane 6 Rocket with Separation
- **Dimensions:** 0D 
- **Combustion:** Yes (Hydrogen)
- **Outputs:** Plot (total mass vs. time)
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/083b488a-6d74-46c6-b134-311d4991bdad" />


### `Engine GE90 Residence Time.ipynb`
- **System:** Boeing 777-300ER Engine
- **Dimensions:** 0D 
- **Combustion:** Yes (Heptane)
- **Outputs:** Plot (Temperature/Fuel Mass Fraction vs time), residence time
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/d21ff469-4dc9-4136-99a3-e3a79639efa1" />


### `Rocket Falcon Heavy Optimal Throttle Schedules.ipynb`
- **System:** Falcon Heavy
- **Dimensions:** 1D 
- **Combustion:** No
- **Outputs:** Plots (Thrust vs time, Propellant mass vs time, Velocity vs time, Altitude vs time
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/f46a74b9-c598-4b1c-a753-295c15c4db07" />
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/62b85b4b-0d95-4795-bb03-15398cd81bcd" />
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/e2021b44-3ce0-4b5f-ab19-0beef5657c93" />
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/12e05f52-5e04-4502-be8c-59fea0893d03" />


### `Rocket Staged and Throttled.ipynb`
- **System:** Falcon Heavy 
- **Dimensions:** 1D 
- **Combustion:** No
- **Outputs:** Plots (altitude vs time, dynamic pressure vs altitude, total mass vs time)
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/2f60f00e-2776-4b17-b8c5-331c6ca98a62" />
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/0434de96-f2e6-43b2-867f-b42342e8befc" />
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/ecfed296-c4fd-452a-b05f-e4fe01884b3c" />

### `Engine Optimizing for Pressure Ratio.ipynb`
- **System:** PW4000-94
- **Dimensions:** 0D 
- **Combustion:** yes (n-dodecane)
- **Outputs:** Table of optimized outputs
**Ideal cycle + Cantera combustor**

| Metric                       | Value                 |
|-----------------------------|-----------------------|
| Optimization finished in    | 0.16 s                |
| Iterations                  | 10                    |
| Best overall pressure ratio | 14.76                 |
| Specific thrust *F*<sub>s</sub> | 1074.2 N·s/`kg_air`   |
| TSFC                        | 0.00002 kg/N/s        |
| Equivalence ratio φ         | 0.376                 |
| Fuel mass flow              | 0.022 kg/s            |
| Mass flow (compressor)      | 54.042 kg/s           |


---

## 🔁 Reproducibility notes

- Pin versions using `requirements.txt` (or add a conda `environment.yml`).
- For deterministic runs, set RNG seeds where applicable.
- Mechanism files are included to avoid external downloads.

---

## 🧭 Running the full workflow in Google Colab

The repository includes three entry points you will usually execute together:

1. The **Brayton-cycle script** (`python brayton.py`) – exercises the
   modular gas-turbine solver and prints a summary of the current operating
   point.
2. The **unit test suite** (`pytest -q`) – validates the solver, regression
   data, and constant-property flag.
3. Any **notebook or ad-hoc study** you want to run (e.g., the notebooks in
   `Therm_Models/notebooks/`).

Below is a minimal Colab notebook skeleton that wires all three together. You
can paste these cells into a fresh Colab runtime and run them top-to-bottom.

```python
# --- 1. Runtime preparation -------------------------------------------------
!pip install --quiet "pip>=23"  # modern pip keeps resolver errors informative
!pip install --quiet -r https://raw.githubusercontent.com/<your-user>/Thermodynamic_Models/main/requirements.txt
!pip install --quiet papermill

# Optional: clone your fork to edit notebooks or push results
%cd /content
!git clone https://github.com/<your-user>/Thermodynamic_Models.git
%cd Thermodynamic_Models

# --- 2. Continuous-integration style smoke test ----------------------------
import pathlib

# Run the Brayton example script (mirrors `python brayton.py`)
!python brayton.py

# Execute the regression/unit tests
!pytest -q

# --- 3. Launch notebooks or additional studies -----------------------------
# Colab already ships with Jupyter; use nbformat/nbconvert if you need to
# batch-run notebooks.  Example: parameter sweep via papermill.
import papermill as pm

notebook_dir = pathlib.Path("Therm_Models/notebooks")
pm.execute_notebook(
    notebook_dir / "Engine Optimizing for Pressure Ratio.ipynb",
    notebook_dir / "Engine Optimizing for Pressure Ratio - output.ipynb",
)

# View results directly in Colab's file browser or download artifacts.
```

**Tips for scaling this workflow**

- Use Colab "runtime type" > GPU only if a notebook explicitly benefits; the
  thermodynamic scripts are CPU-oriented.
- For multi-point design studies, wrap the papermill call in a loop over
  ambient schedules, writing each output notebook to a unique name.
- Store large data files on Google Drive, then `drive.mount("/content/drive")`
  before running the setup cell so notebooks can access shared inputs.

---

## 🧪 Tested with

- Python 3.11
- `cantera`, `netCDF4`, `cftime`, `ruamel.yaml`, `matplotlib`, `jupyter`

---

## 🙌 Contributing

PRs are welcome! If you add a new model, please:
1. Follow the **Plot Style** defaults.
2. Add a short section to **Model Catalog**.
3. Include a small test cell or validation check.
