# Aerospace System Models

A compact collection of aerospace models—rockets, engines, and flame physics—implemented as reproducible Jupyter notebooks with **uniform, publication-ready plots** (Consolas font, consistent sizing). This repo is ideal for students and practitioners who want clear, minimal models they can run and extend.

> **Scope cutoff:** Analyses summarized here reflect work available as of **2025-09-24**.

---

## 📦 Contents

- Notebooks: `Script A.ipynb`, `Script B.ipynb`, `Script C.ipynb`, `Script D.ipynb`
- External Mechanisms: `Mechanism1.yaml`, `C7H16.yaml`
- This README and a minimal `requirements.txt`

Suggested structure:
```
.
├── Script A.ipynb
├── Script B.ipynb
├── Script C.ipynb
├── Script D.ipynb
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

## 🎨 Plot Style (consistent across all notebooks)

All figures share a single, consistent style for easy side‑by‑side comparison:

- **Font:** Consolas (fallbacks used if not available)
- **Figure size:** 8 × 5 in
- **Line width:** 2.0
- **Grid:** major+minor; alpha 0.3
- **DPI:** 150

Paste this at the top of each notebook (or keep in a common `style.py`):

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "Consolas",
    "figure.figsize": (8, 5),
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.grid.which": "both",
    "savefig.dpi": 150,
})
```

---

## 📚 Model Catalog (at a glance)

> Replace any placeholders below with specifics if your notebooks print/annotate them.

### `Script A.ipynb`
- **System:** Ariane 6 Rocket with Separation
- **Dimensions:** 0D 
- **Combustion:** Yes 
- **Outputs:** Plot (total mass vs. time)
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/083b488a-6d74-46c6-b134-311d4991bdad" />


### `Script B.ipynb`
- **System:** Boeing 777-300ER Engine
- **Dimensions:** 0D 
- **Combustion:** Yes 
- **Outputs:** Plot (Temperature/Fuel Mass Fraction vs time), residence time
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/d21ff469-4dc9-4136-99a3-e3a79639efa1" />


### `Script C.ipynb`
- **System:** Falcon Heavy Boosters and Throttling
- **Dimensions:** 1D 
- **Combustion:** No
- **Outputs:** Plots (altitude vs time, dynamic pressure vs altitude, total mass vs time)
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/2f60f00e-2776-4b17-b8c5-331c6ca98a62" />
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/0434de96-f2e6-43b2-867f-b42342e8befc" />
<img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/ecfed296-c4fd-452a-b05f-e4fe01884b3c" />

### `Script D.ipynb`
- **System:** Optimizing Pressure Ratio
- **Dimensions:** 0D 
- **Combustion:** yes
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

## 🧪 Tested with

- Python 3.11
- `cantera`, `netCDF4`, `cftime`, `ruamel.yaml`, `matplotlib`, `jupyter`

---

## 🙌 Contributing

PRs are welcome! If you add a new model, please:
1. Follow the **Plot Style** defaults.
2. Add a short section to **Model Catalog**.
3. Include a small test cell or validation check.
