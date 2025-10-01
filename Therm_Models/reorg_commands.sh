#!/usr/bin/env bash
set -euo pipefail

# Run from your repo root
mkdir -p notebooks/
mkdir -p data/mechanisms/
mkdir -p assets/geometry/
mkdir -p docs/
mkdir -p src/thermo_models/
mkdir -p tests/
mkdir -p .github/workflows/
git mv 'Engine GE90 Residence Time.ipynb' 'notebooks/Engine GE90 Residence Time.ipynb' || mv 'Engine GE90 Residence Time.ipynb' 'notebooks/Engine GE90 Residence Time.ipynb'
git mv 'Engine Optimizing for Pressure Ratio.ipynb' 'notebooks/Engine Optimizing for Pressure Ratio.ipynb' || mv 'Engine Optimizing for Pressure Ratio.ipynb' 'notebooks/Engine Optimizing for Pressure Ratio.ipynb'
git mv 'Rocket Ariane 6 Staged Mass Loss.ipynb' 'notebooks/Rocket Ariane 6 Staged Mass Loss.ipynb' || mv 'Rocket Ariane 6 Staged Mass Loss.ipynb' 'notebooks/Rocket Ariane 6 Staged Mass Loss.ipynb'
git mv 'Rocket Falcon Heavy Optimal Throttle Schedules.ipynb' 'notebooks/Rocket Falcon Heavy Optimal Throttle Schedules.ipynb' || mv 'Rocket Falcon Heavy Optimal Throttle Schedules.ipynb' 'notebooks/Rocket Falcon Heavy Optimal Throttle Schedules.ipynb'
git mv 'Rocket Staged and Throttled.ipynb' 'notebooks/Rocket Staged and Throttled.ipynb' || mv 'Rocket Staged and Throttled.ipynb' 'notebooks/Rocket Staged and Throttled.ipynb'
git mv 'Two_Stage_Optimal_Mass.ipynb' 'notebooks/Two_Stage_Optimal_Mass.ipynb' || mv 'Two_Stage_Optimal_Mass.ipynb' 'notebooks/Two_Stage_Optimal_Mass.ipynb'
git mv 'C7H16.yaml' 'data/mechanisms/C7H16.yaml' || mv 'C7H16.yaml' 'data/mechanisms/C7H16.yaml'
git mv 'Mechanism1.yaml' 'data/mechanisms/Mechanism1.yaml' || mv 'Mechanism1.yaml' 'data/mechanisms/Mechanism1.yaml'
git mv 'TPMS_solid_G_2mm.stl' 'assets/geometry/TPMS_solid_G_2mm.stl' || mv 'TPMS_solid_G_2mm.stl' 'assets/geometry/TPMS_solid_G_2mm.stl'