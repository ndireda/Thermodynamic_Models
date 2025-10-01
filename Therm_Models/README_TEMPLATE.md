# Thermodynamic Models

A collection of thermodynamic modeling notebooks and supporting data.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project layout
```
notebooks/          # Analysis & modeling notebooks
data/mechanisms/    # Chemical mechanism YAML files
assets/geometry/    # 3D assets (STL) and similar
src/thermo_models/  # (Optional) reusable python modules
tests/              # Tests
docs/               # Additional docs
.github/workflows/  # CI
```

## Conventions
- Outputs in notebooks should be lightweight; prefer saving large results to `data/` (gitignored).
- Use `nbqa` (Black, isort) to keep notebooks consistent.
- Consider pairing notebooks with scripts via jupytext if you need code review on logic.

## CI
A minimal CI checks formatting for notebooks using nbQA. Edit `.github/workflows/ci.yml` to add execution tests if your environment supports it.

## Data
- Mechanism files are under `data/mechanisms/`.
- Large/binary assets should use Git LFS (see `.gitattributes`).

## License
TBD