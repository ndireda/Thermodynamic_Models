"""

The module restructures the earlier scalar Brayton script into a modular
framework that mirrors modern propulsion analysis practices:

* **Stations** encapsulate total thermodynamic states, mass-flow, and fuel
  content.  They provide accessors for temperature-dependent properties so the
  rest of the model can operate on enriched flow objects rather than loose
  tuples.
* **Components** (compressor, burner, turbine, duct, and nozzle) expose
  consistent call signatures and map-based performance predictions.  Their
  implementations use the same ideal-gas relations as the original script while
  allowing variable specific heats and mixture rules.
* **Inner solves** leverage :func:`scipy.optimize.least_squares` to balance the
  shaft power, satisfy component maps, and enforce nozzle boundary conditions.
  The unknown vector can include corrected flow, pressure ratios, throat area,
  or spool speed, making it straightforward to extend with additional
  constraints.
* **Design versus off-design** support arrives via map scaling factors that are
  established at a chosen design point and then frozen when evaluating
  off-design ambient conditions or speed schedules.
* **Outer optimization** wraps the inner nonlinear solve in an SLSQP
  formulation so users can perform multi-point trade studies that respect
  physical and operability constraints (thermal limits, surge margin, specific
  work targets, etc.).

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, MutableMapping, Sequence

import numpy as np
from scipy import optimize


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Property package
# ---------------------------------------------------------------------------


@dataclass
class GasProperties:
    """Simple calorically imperfect gas model with mixture rules.

    Parameters
    ----------
    variable : bool, optional
        When ``True`` temperature-dependent cp/gamma expressions are used.  When
        ``False`` the properties collapse to constant values corresponding to
        the original script.
    """

    variable: bool = False
    t_ref: float = 288.15  # [K]
    h_ref: float = 0.0

    # Polynomial coefficients for cp(T) = a + b*T + c*T^2 + d*T^3
    # Values loosely based on NASA curves for air (valid over ~200-2000 K).
    cp_air_coeffs: Sequence[float] = (1_003.0, 0.1, -3.0e-4, 1.0e-7)
    cp_fuel_coeffs: Sequence[float] = (1_850.0, 0.2, -4.5e-4, 1.5e-7)

    r_air: float = 287.05  # J/(kg*K)
    r_fuel: float = 50.0   # surrogate for liquid hydrocarbon vapor mix

    def _cp_poly(self, coeffs: Sequence[float], T: float) -> float:
        a, b, c, d = coeffs
        return a + b * T + c * T * T + d * T * T * T

    def _h_poly(self, coeffs: Sequence[float], T: float) -> float:
        a, b, c, d = coeffs
        return (
            a * (T - self.t_ref)
            + 0.5 * b * (T * T - self.t_ref * self.t_ref)
            + (1.0 / 3.0) * c * (T**3 - self.t_ref**3)
            + 0.25 * d * (T**4 - self.t_ref**4)
        )

    def cp_air(self, T: float) -> float:
        return 1_005.0 if not self.variable else self._cp_poly(self.cp_air_coeffs, T)

    def cp_fuel(self, T: float) -> float:
        return 2_000.0 if not self.variable else self._cp_poly(self.cp_fuel_coeffs, T)

    def cp(self, T: float, f: float) -> float:
        cp_air = self.cp_air(T)
        cp_fuel = self.cp_fuel(T)
        return (cp_air + f * cp_fuel) / (1.0 + f)

    def R(self, f: float) -> float:
        return (self.r_air + f * self.r_fuel) / (1.0 + f)

    def gamma(self, T: float, f: float) -> float:
        cp_val = self.cp(T, f)
        r_val = self.R(f)
        return cp_val / (cp_val - r_val)

    def h(self, T: float, f: float) -> float:
        if not self.variable:
            return self.cp(T, f) * (T - self.t_ref)
        w_air = 1.0 / (1.0 + f)
        w_fuel = f / (1.0 + f)
        return self._h_poly(self.cp_air_coeffs, T) * w_air + self._h_poly(
            self.cp_fuel_coeffs, T
        ) * w_fuel

    def solve_temperature(self, h_target: float, f: float, T_guess: float) -> float:
        """Invert enthalpy->temperature with a Newton iteration."""

        T = max(50.0, T_guess)
        for _ in range(20):
            h_val = self.h(T, f)
            resid = h_val - h_target
            if abs(resid) < 1e-4:
                return T
            cp_val = max(10.0, self.cp(T, f))
            T -= resid / cp_val
        raise RuntimeError("Temperature solve did not converge")


# ---------------------------------------------------------------------------
# Stations (total conditions)
# ---------------------------------------------------------------------------


@dataclass
class Station:
    """Total-state container used for component interfaces."""

    Tt: float
    Pt: float
    mdot: float
    f: float = 0.0
    gas: GasProperties = field(default_factory=GasProperties)

    def copy(self, **kwargs: float) -> "Station":
        data = {
            "Tt": self.Tt,
            "Pt": self.Pt,
            "mdot": self.mdot,
            "f": self.f,
            "gas": self.gas,
        }
        data.update(kwargs)
        return Station(**data)

    @property
    def cp(self) -> float:
        return self.gas.cp(self.Tt, self.f)

    @property
    def gamma(self) -> float:
        return self.gas.gamma(self.Tt, self.f)

    @property
    def h(self) -> float:
        return self.gas.h(self.Tt, self.f)

    @property
    def R(self) -> float:
        return self.gas.R(self.f)


# ---------------------------------------------------------------------------
# Component map helpers
# ---------------------------------------------------------------------------


def polyval2d(dphi: float, dn: float, coeffs: np.ndarray) -> float:
    total = 0.0
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            total += coeffs[i, j] * (dphi**i) * (dn**j)
    return total


@dataclass
class MapSurface:
    """Quadratic surface for compressor/turbine map quantities."""

    coeffs: np.ndarray
    flow_ref: float
    speed_ref: float
    scale: float = 1.0

    def evaluate(self, corrected_flow: float, corrected_speed: float) -> float:
        dphi = (corrected_flow - self.flow_ref) / self.flow_ref
        dn = (corrected_speed - self.speed_ref) / self.speed_ref
        return self.scale * polyval2d(dphi, dn, self.coeffs)


# ---------------------------------------------------------------------------
# Component implementations
# ---------------------------------------------------------------------------


@dataclass
class Compressor:
    """Compressor model driven by map surfaces for PR and efficiency."""

    pr_map: MapSurface
    eta_map: MapSurface
    mech_efficiency: float = 1.0

    def __call__(self, station_in: Station, mdot: float, speed: float, pr_target: float | None = None) -> tuple[Station, Dict[str, float]]:
        theta = station_in.Tt / 288.15
        delta = station_in.Pt / 101_325.0
        corrected_flow = mdot * np.sqrt(theta) / delta
        corrected_speed = speed / np.sqrt(theta)

        pr_map = max(1.0001, self.pr_map.evaluate(corrected_flow, corrected_speed))
        pr = max(1.0001, pr_target if pr_target is not None else pr_map)
        eta = max(1e-3, min(0.999, self.eta_map.evaluate(corrected_flow, corrected_speed)))

        gamma_in = station_in.gamma
        tau_isentropic = pr ** ((gamma_in - 1.0) / gamma_in)
        Tt_out_s = station_in.Tt * tau_isentropic
        h_in = station_in.h
        h_s = station_in.gas.h(Tt_out_s, station_in.f)
        delta_h = (h_s - h_in) / eta
        h_out = h_in + delta_h
        Tt_out = station_in.gas.solve_temperature(h_out, station_in.f, Tt_out_s)
        Pt_out = station_in.Pt * pr

        work_specific = (h_out - h_in) / self.mech_efficiency

        station_out = station_in.copy(Tt=Tt_out, Pt=Pt_out, mdot=mdot)
        return station_out, {
            "corrected_flow": corrected_flow,
            "corrected_speed": corrected_speed,
            "pr": pr_map,
            "eta": eta,
            "w_specific": work_specific,
        }


@dataclass
class Burner:
    """Burner model with pressure loss and fuel addition."""

    dp_frac: float = 0.05
    eta_burner: float = 0.99
    lhv: float = 43_000_000.0

    def __call__(self, station_in: Station, Tt_out_target: float) -> tuple[Station, Dict[str, float]]:
        Pt_out = station_in.Pt * (1.0 - self.dp_frac)
        h_in = station_in.h

        def energy_balance(fuel_to_air: float) -> float:
            f_out = station_in.f + fuel_to_air
            h_out = station_in.gas.h(Tt_out_target, f_out)
            return h_out - h_in - fuel_to_air * self.eta_burner * self.lhv

        fuel = 1e-6
        for _ in range(20):
            res = energy_balance(fuel)
            if abs(res) < 1e-3:
                break
            f_out = station_in.f + fuel
            cp_out = station_in.gas.cp(Tt_out_target, f_out)
            dres_dfuel = cp_out * 0.0 - self.eta_burner * self.lhv  # approx derivative
            if dres_dfuel == 0:
                break
            fuel -= res / dres_dfuel
            fuel = max(1e-8, fuel)
        f_out = station_in.f + fuel
        station_out = station_in.copy(Tt=Tt_out_target, Pt=Pt_out, f=f_out)
        return station_out, {"f_added": fuel}


@dataclass
class Turbine:
    """Turbine model mirroring the compressor structure."""

    pr_map: MapSurface
    eta_map: MapSurface
    mech_efficiency: float = 1.0

    def __call__(self, station_in: Station, mdot: float, speed: float, pr_target: float | None = None) -> tuple[Station, Dict[str, float]]:
        theta = station_in.Tt / 288.15
        delta = station_in.Pt / 101_325.0
        corrected_flow = mdot * np.sqrt(theta) / delta
        corrected_speed = speed / np.sqrt(theta)

        if pr_target is None:
            pr = max(1.0001, self.pr_map.evaluate(corrected_flow, corrected_speed))
        else:
            pr = max(1.0001, pr_target)
        eta = max(1e-3, min(0.999, self.eta_map.evaluate(corrected_flow, corrected_speed)))

        gamma_in = station_in.gamma
        tau_isentropic = pr ** ((gamma_in - 1.0) / gamma_in)
        Tt_out_s = station_in.Tt / tau_isentropic
        h_in = station_in.h
        h_s = station_in.gas.h(Tt_out_s, station_in.f)
        delta_h_is = h_in - h_s
        delta_h = eta * delta_h_is
        h_out = h_in - delta_h
        Tt_out = station_in.gas.solve_temperature(h_out, station_in.f, Tt_out_s)
        Pt_out = station_in.Pt / pr

        work_specific = (h_in - h_out) * self.mech_efficiency
        station_out = station_in.copy(Tt=Tt_out, Pt=Pt_out, mdot=mdot)
        return station_out, {
            "corrected_flow": corrected_flow,
            "corrected_speed": corrected_speed,
            "pr": pr,
            "eta": eta,
            "w_specific": work_specific,
        }


@dataclass
class Duct:
    """Total-pressure loss duct (e.g., inlet or diffuser)."""

    loss_frac: float = 0.02

    def __call__(self, station_in: Station) -> Station:
        Pt_out = station_in.Pt * (1.0 - self.loss_frac)
        return station_in.copy(Pt=Pt_out)


@dataclass
class Nozzle:
    """Convergent nozzle with optional choking."""

    efficiency: float = 0.98

    def mass_flow(self, station_in: Station, area: float, P_ambient: float) -> tuple[float, Dict[str, float]]:
        gamma = station_in.gamma
        R = station_in.R
        Pt = station_in.Pt
        Tt = station_in.Tt

        critical_pressure_ratio = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
        P_crit = Pt * critical_pressure_ratio

        cp_total = station_in.gas.cp(station_in.Tt, station_in.f)

        if P_ambient <= P_crit:  # Choked
            T_star = Tt * 2.0 / (gamma + 1.0)
            a_star = np.sqrt(gamma * R * T_star)
            rho_star = Pt / (R * Tt) * (2.0 / (gamma + 1.0)) ** (1.0 / (gamma - 1.0))
            mdot = rho_star * a_star * area
            Pe = P_crit
            choked = True
            Te = T_star
            Ve = np.sqrt(2.0 * self.efficiency * cp_total * (Tt - Te))
        else:
            def pressure_from_mach(M: float) -> float:
                return Pt * (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (-gamma / (gamma - 1.0))

            lower, upper = 1e-6, 5.0
            for _ in range(60):
                mid = 0.5 * (lower + upper)
                p_mid = pressure_from_mach(mid)
                if abs(p_mid - P_ambient) < 1e-6:
                    Me = mid
                    break
                if p_mid > P_ambient:
                    lower = mid
                else:
                    upper = mid
            else:
                Me = mid
            Te = Tt / (1.0 + 0.5 * (gamma - 1.0) * Me * Me)
            Pe = P_ambient
            rho_e = Pe / (R * Te)
            Ve = Me * np.sqrt(gamma * R * Te)
            mdot = rho_e * Ve * area
            choked = False

        thrust = mdot * Ve + (Pe - P_ambient) * area
        return mdot, {"Pe": Pe, "choked": choked, "thrust": thrust, "Ve": Ve, "Te": Te}


# ---------------------------------------------------------------------------
# Cycle assembly and residual system
# ---------------------------------------------------------------------------


@dataclass
class CycleParameters:
    compressor_speed: float
    turbine_speed: float
    load_watts: float
    nozzle_area: float
    target_specific_work: float
    ambient_Tt: float
    ambient_Pt: float
    burner_exit_Tt: float
    variable_properties: bool = False


@dataclass
class DesignPoint:
    corrected_flow: float
    pressure_ratio: float
    efficiency: float
    mass_flow: float
    speed: float


@dataclass
class CycleModel:
    gas: GasProperties
    compressor: Compressor
    burner: Burner
    turbine: Turbine
    inlet: Duct
    nozzle: Nozzle

    design_scaling: MutableMapping[str, float] = field(default_factory=dict)

    def set_design_point(self, compressor_design: DesignPoint, turbine_design: DesignPoint) -> None:
        """Determine map scaling factors that hit the provided design point."""

        self.compressor.pr_map.scale = compressor_design.pressure_ratio / self.compressor.pr_map.evaluate(
            compressor_design.corrected_flow, compressor_design.speed
        )
        self.compressor.eta_map.scale = compressor_design.efficiency / self.compressor.eta_map.evaluate(
            compressor_design.corrected_flow, compressor_design.speed
        )
        self.turbine.pr_map.scale = turbine_design.pressure_ratio / self.turbine.pr_map.evaluate(
            turbine_design.corrected_flow, turbine_design.speed
        )
        self.turbine.eta_map.scale = turbine_design.efficiency / self.turbine.eta_map.evaluate(
            turbine_design.corrected_flow, turbine_design.speed
        )

        self.design_scaling.update(
            {
                "compressor_flow": compressor_design.corrected_flow,
                "turbine_flow": turbine_design.corrected_flow,
            }
        )

    def residuals(self, x: np.ndarray, params: CycleParameters) -> np.ndarray:
        mdot, pr_c, turbine_pr, nozzle_area_scale, speed_scale = x

        gas = GasProperties(variable=params.variable_properties)
        ambient = Station(Tt=params.ambient_Tt, Pt=params.ambient_Pt, mdot=mdot, gas=gas)
        ambient = ambient.copy(f=0.0)

        inlet = self.inlet(ambient)

        comp_speed = params.compressor_speed * speed_scale
        turb_speed = params.turbine_speed * speed_scale

        comp_out, comp_data = self.compressor(inlet, mdot, comp_speed, pr_target=pr_c)

        burner_out, burner_data = self.burner(comp_out, params.burner_exit_Tt)

        turbine_in = burner_out.copy(mdot=mdot * (1.0 + burner_out.f))
        turbine_out, turbine_data = self.turbine(
            turbine_in,
            turbine_in.mdot,
            turb_speed,
            pr_target=turbine_pr,
        )

        nozzle_area = params.nozzle_area * nozzle_area_scale
        mdot_nozzle, nozzle_data = self.nozzle.mass_flow(turbine_out, nozzle_area, params.ambient_Pt)

        # Residuals --------------------------------------------------------
        residuals = np.zeros(5)

        # 1. Shaft power balance (turbine vs compressor + load)
        w_comp = comp_data["w_specific"]
        w_turb = turbine_data["w_specific"]
        turbine_power = turbine_in.mdot * w_turb
        compressor_power = mdot * w_comp
        shaft_balance = turbine_power - compressor_power - params.load_watts
        residuals[0] = shaft_balance

        # 2. Compressor pressure ratio map consistency
        residuals[1] = comp_data["pr"] - pr_c

        # 3. Turbine pressure ratio map consistency
        residuals[2] = turbine_data["pr"] - turbine_pr

        # 4. Nozzle mass conservation
        residuals[3] = mdot_nozzle - turbine_out.mdot

        # 5. Target specific work (per kg of air)
        w_net = (turbine_power - compressor_power) / mdot if mdot else 0.0
        residuals[4] = w_net - params.target_specific_work

        scales = np.array([
            max(1.0, abs(params.load_watts)),
            max(0.1, pr_c),
            max(0.1, turbine_pr),
            max(0.1, turbine_out.mdot),
            max(1.0, params.target_specific_work),
        ])

        diagnostics = {
            "w_comp": w_comp,
            "w_turb": w_turb,
            "turbine_power": turbine_power,
            "compressor_power": compressor_power,
            "f_total": burner_out.f,
            "nozzle": nozzle_data,
        }
        LOG.debug("Residual diagnostics: %s", diagnostics)

        return residuals / scales

    def solve_inner(self, params: CycleParameters, x0: Sequence[float], bounds: Sequence[tuple[float, float]]) -> optimize.OptimizeResult:
        """Solve the nonlinear system that balances the Brayton cycle."""

        def fun(x: np.ndarray) -> np.ndarray:
            return self.residuals(x, params)

        lower, upper = zip(*bounds)
        result = optimize.least_squares(
            fun,
            x0=x0,
            bounds=(lower, upper),
            xtol=1e-6,
            ftol=1e-2,
            gtol=1e-2,
            max_nfev=400,
        )

        if not result.success:
            LOG.error("Inner solve failed: %s", result.message)
            raise RuntimeError("Inner solve did not converge")

        return result

    def optimize(self, params: CycleParameters, x0: Sequence[float], bounds: Sequence[tuple[float, float]], objective: Callable[[np.ndarray], float], constraints: Sequence[Mapping[str, object]] = ()) -> optimize.OptimizeResult:
        """Outer optimization wrapper using SLSQP."""

        bounds_arr = optimize.Bounds(*zip(*bounds))

        def wrapped_objective(x: np.ndarray) -> float:
            inner = self.solve_inner(params, x, bounds)
            return objective(inner.x)

        def make_constraint(spec: Mapping[str, object]):
            kind = spec.get("type", "eq")

            def fun(x: np.ndarray) -> float:
                inner = self.solve_inner(params, x, bounds)
                return spec["fun"](inner.x)

            return {"type": kind, "fun": fun}

        scipy_constraints = [make_constraint(c) for c in constraints]

        result = optimize.minimize(
            wrapped_objective,
            x0=np.asarray(x0, dtype=float),
            method="SLSQP",
            bounds=bounds_arr,
            constraints=scipy_constraints,
            options={"maxiter": 25, "ftol": 1e-6},
        )

        if not result.success:
            LOG.error("Outer optimization failed: %s", result.message)
            raise RuntimeError("Outer optimization did not converge")

        return result


# ---------------------------------------------------------------------------
# Example usage (root finding + optimization)
# ---------------------------------------------------------------------------


def _build_default_cycle(variable_properties: bool = False) -> CycleModel:
    gas = GasProperties(variable=variable_properties)

    coeffs_pr = np.array(
        [
            [12.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    coeffs_eta = np.array(
        [
            [0.88, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    compressor = Compressor(
        pr_map=MapSurface(coeffs=coeffs_pr, flow_ref=22.0, speed_ref=1.0),
        eta_map=MapSurface(coeffs=coeffs_eta, flow_ref=22.0, speed_ref=1.0),
        mech_efficiency=0.99,
    )

    turbine_coeffs_pr = np.array(
        [
            [8.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    turbine_coeffs_eta = np.array(
        [
            [0.90, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    turbine = Turbine(
        pr_map=MapSurface(coeffs=turbine_coeffs_pr, flow_ref=20.0, speed_ref=1.0),
        eta_map=MapSurface(coeffs=turbine_coeffs_eta, flow_ref=20.0, speed_ref=1.0),
        mech_efficiency=0.99,
    )

    burner = Burner()
    inlet = Duct(loss_frac=0.03)
    nozzle = Nozzle(efficiency=0.98)

    model = CycleModel(gas=gas, compressor=compressor, burner=burner, turbine=turbine, inlet=inlet, nozzle=nozzle)

    compressor_design = DesignPoint(
        corrected_flow=22.0,
        pressure_ratio=12.0,
        efficiency=0.88,
        mass_flow=25.0,
        speed=1.0,
    )
    turbine_design = DesignPoint(
        corrected_flow=20.0,
        pressure_ratio=8.0,
        efficiency=0.90,
        mass_flow=27.0,
        speed=1.0,
    )

    model.set_design_point(compressor_design, turbine_design)
    return model


def run_example() -> Dict[str, float]:
    model = _build_default_cycle(variable_properties=True)

    params = CycleParameters(
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

    x0 = [25.0, 10.0, 6.0, 1.0, 1.0]
    bounds = [
        (5.0, 80.0),   # mdot
        (1.5, 35.0),   # compressor PR
        (1.5, 25.0),   # turbine PR
        (0.5, 2.0),    # nozzle area scale
        (0.5, 1.5),    # speed scale
    ]

    inner_result = model.solve_inner(params, x0, bounds)

    def objective(x: np.ndarray) -> float:
        mdot = x[0]
        return mdot  # minimize mass flow for efficiency proxy

    constraints = [
        {"type": "ineq", "fun": lambda x: 0.2 - abs(x[1] - 12.0)},  # maintain surge margin
        {"type": "ineq", "fun": lambda x: x[2] - 4.0},
    ]

    optimization_result = model.optimize(params, inner_result.x, bounds, objective, constraints)

    return {
        "inner_solution": inner_result.x.tolist(),
        "inner_cost": float(inner_result.cost),
        "optimization_solution": optimization_result.x.tolist(),
        "optimization_cost": float(optimization_result.fun),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    info = run_example()
    print("Inner solution:", np.array2string(np.asarray(info["inner_solution"]), precision=3))
    print("Optimization solution:", np.array2string(np.asarray(info["optimization_solution"]), precision=3))
