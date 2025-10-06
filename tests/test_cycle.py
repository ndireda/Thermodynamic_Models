"""Regression tests for the Brayton cycle model."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import brayton


def test_inner_solver_converges():
    model = brayton._build_default_cycle(variable_properties=True)
    params = brayton.CycleParameters(
        compressor_speed=1.0,
        turbine_speed=1.0,
        load_watts=7_900_000.0,
        nozzle_area=0.1,
        target_specific_work=320_000.0,
        ambient_Tt=288.15,
        ambient_Pt=101_325.0,
        burner_exit_Tt=1_600.0,
        variable_properties=True,
    )

    x0 = [24.0, 12.0, 8.0, 1.0, 1.0]
    bounds = [
        (5.0, 80.0),
        (1.5, 35.0),
        (1.5, 25.0),
        (0.5, 2.0),
        (0.5, 1.5),
    ]

    result = model.solve_inner(params, x0, bounds)
    assert result.success
    assert np.linalg.norm(result.fun) < 1e-2


def test_run_example_returns_summary():
    info = brayton.run_example()
    assert "inner_solution" in info
    assert "optimization_solution" in info
    assert len(info["inner_solution"]) == 5


def test_constant_property_flag():
    model = brayton._build_default_cycle(variable_properties=False)
    params = brayton.CycleParameters(
        compressor_speed=1.0,
        turbine_speed=1.0,
        load_watts=7_900_000.0,
        nozzle_area=0.1,
        target_specific_work=320_000.0,
        ambient_Tt=288.15,
        ambient_Pt=101_325.0,
        burner_exit_Tt=1_600.0,
        variable_properties=False,
    )
    x0 = [24.0, 12.0, 8.0, 1.0, 1.0]
    bounds = [
        (5.0, 80.0),
        (1.5, 35.0),
        (1.5, 25.0),
        (0.5, 2.0),
        (0.5, 1.5),
    ]
    result = model.solve_inner(params, x0, bounds)
    assert result.success
