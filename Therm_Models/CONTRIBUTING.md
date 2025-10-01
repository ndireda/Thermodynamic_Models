# Contributing

## Environment
- Python 3.11+ recommended
- `pip install -r requirements.txt`
- Optional tooling: `pip install pre-commit nbqa black isort flake8`

Enable pre-commit hooks:
```bash
pre-commit install
```

## Structure
- Notebooks live in `notebooks/`
- Mechanism YAML files in `data/mechanisms/`
- STL or other CAD assets in `assets/geometry/`
- (Optional) library code in `src/thermo_models/`
- Tests in `tests/`

## PR checklist
- Clear title, small focused changes
- Notebooks stripped of large outputs; keep figures light
- Update README if user-facing behavior changes