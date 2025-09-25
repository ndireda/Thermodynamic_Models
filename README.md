# Aerospace System Models

A compact collection of aerospace models—rockets, engines, and flame physics—implemented as reproducible Jupyter notebooks with **uniform, publication-ready plots** (Consolas font, consistent sizing). This repo is ideal for students and practitioners who want clear, minimal models they can run and extend.

> **Scope cutoff:** Analyses summarized here reflect work available as of **2025-09-24**.

---

## 📦 Contents

- Notebooks: `Script A.ipynb`, `Script B.ipynb`, `Script C.ipynb`, `Script D.ipynb`
- Mechanisms: `Mechanism1.yaml`, `C7H16.yaml`
- This README and a minimal `requirements.txt`

Suggested structure:
```
.
├── Script A.ipynb
├── Script B.ipynb
├── Script C.ipynb
├── Script D.ipynb
├── Mechanism1.yaml
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
- **System:** _{YourCompany} {SystemName}_
- **Dimensions:** 0D / 1D (choose as appropriate)
- **Combustion:** yes / no
- **Outputs:** list the variables, metrics, and plots the notebook produces

### `Script B.ipynb`
- **System:** _{YourCompany} {SystemName}_
- **Dimensions:** 0D / 1D (choose as appropriate)
- **Combustion:** yes / no
- **Outputs:** list the variables, metrics, and plots the notebook produces

### `Script C.ipynb`
- **System:** _{YourCompany} {SystemName}_
- **Dimensions:** 0D / 1D (choose as appropriate)
- **Combustion:** yes / no
- **Outputs:** list the variables, metrics, and plots the notebook produces

### `Script D.ipynb`
- **System:** _{YourCompany} {SystemName}_
- **Dimensions:** 0D / 1D (choose as appropriate)
- **Combustion:** yes / no
- **Outputs:** list the variables, metrics, and plots the notebook produces

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

## 📄 License

Add your preferred license (e.g., MIT, BSD-3-Clause).

---

## 🙌 Contributing

PRs are welcome! If you add a new model, please:
1. Follow the **Plot Style** defaults.
2. Add a short section to **Model Catalog**.
3. Include a small test cell or validation check.