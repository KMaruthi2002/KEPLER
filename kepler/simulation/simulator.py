"""
Physics simulator module.

Takes candidate equations from Muse Spark and:
1. Converts them into runnable simulators
2. Fits physical constants to observed trajectory data
3. Produces predicted trajectories for comparison
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import sympy as sp
from dataclasses import dataclass

from kepler.tracking.tracker import Trajectory


@dataclass
class FitResult:
    """Result of fitting an equation to observed data."""
    equation_name: str
    equation_str: str
    constants: dict[str, float]
    residual: float  # total fit error
    predicted_trajectory: dict[str, np.ndarray]
    complexity: int


class PhysicsSimulator:
    """Fit and simulate physical equations against observed trajectories."""

    def __init__(self):
        self.t_sym = sp.Symbol("t")

    def fit_equation(
        self,
        candidate: dict,
        trajectory: Trajectory,
        pixel_to_meter: float = 0.01,
    ) -> FitResult:
        """
        Fit a candidate equation's constants to observed trajectory data.

        Args:
            candidate: Equation dict from Muse Spark with 'equation', 'constants', etc.
            trajectory: Observed trajectory from tracking.
            pixel_to_meter: Approximate scale factor (can be refined).

        Returns:
            FitResult with fitted constants and predicted trajectory.
        """
        obs = trajectory.to_numpy()
        t_obs = obs["t"]
        # Use y for vertical motion, x for horizontal
        y_obs = obs["y"] * pixel_to_meter

        # Parse the sympy equation
        eq_str = candidate["equation"]
        constants_info = candidate.get("constants", {})

        # Create sympy symbols for constants
        const_symbols = {}
        initial_guesses = []
        const_names = []

        for name, info in constants_info.items():
            const_symbols[name] = sp.Symbol(name)
            initial_guesses.append(info.get("initial_guess", 1.0))
            const_names.append(name)

        # Parse the equation
        all_symbols = {"t": self.t_sym, **const_symbols}
        try:
            expr = sp.sympify(eq_str, locals=all_symbols)
        except (sp.SympifyError, SyntaxError) as e:
            raise ValueError(f"Cannot parse equation '{eq_str}': {e}")

        # Lambdify for fast evaluation
        func = sp.lambdify([self.t_sym] + list(const_symbols.values()), expr, modules="numpy")

        def residual_fn(params):
            try:
                y_pred = func(t_obs, *params)
                return np.sum((y_pred - y_obs) ** 2)
            except (ValueError, OverflowError, ZeroDivisionError):
                return 1e12

        # Optimize
        result = minimize(
            residual_fn,
            x0=initial_guesses,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
        )

        fitted_constants = dict(zip(const_names, result.x))

        # Generate predicted trajectory
        t_pred = np.linspace(t_obs[0], t_obs[-1] * 1.5, len(t_obs) * 2)
        try:
            y_pred = func(t_pred, *result.x)
        except (ValueError, OverflowError):
            y_pred = np.full_like(t_pred, np.nan)

        return FitResult(
            equation_name=candidate.get("name", "unknown"),
            equation_str=eq_str,
            constants=fitted_constants,
            residual=float(result.fun),
            predicted_trajectory={"t": t_pred, "y": y_pred},
            complexity=candidate.get("complexity", 1),
        )

    def rank_candidates(
        self,
        fit_results: list[FitResult],
        simplicity_weight: float = 0.3,
    ) -> list[FitResult]:
        """
        Rank fitted equations by fit quality + simplicity (Occam's razor).

        Score = (1 - simplicity_weight) * normalized_fit + simplicity_weight * normalized_complexity
        Lower is better.
        """
        if not fit_results:
            return []

        residuals = np.array([r.residual for r in fit_results])
        complexities = np.array([r.complexity for r in fit_results])

        # Normalize to [0, 1]
        r_range = residuals.max() - residuals.min()
        if r_range > 0:
            norm_residuals = (residuals - residuals.min()) / r_range
        else:
            norm_residuals = np.zeros_like(residuals)

        c_range = complexities.max() - complexities.min()
        if c_range > 0:
            norm_complexity = (complexities - complexities.min()) / c_range
        else:
            norm_complexity = np.zeros_like(complexities)

        scores = (1 - simplicity_weight) * norm_residuals + simplicity_weight * norm_complexity

        ranked_indices = np.argsort(scores)
        return [fit_results[i] for i in ranked_indices]

    def compute_error(
        self,
        fit_result: FitResult,
        trajectory: Trajectory,
        pixel_to_meter: float = 0.01,
    ) -> dict:
        """Compute error metrics between simulation and observation."""
        obs = trajectory.to_numpy()
        t_obs = obs["t"]
        y_obs = obs["y"] * pixel_to_meter

        # Re-evaluate the fitted equation at observed times
        const_info = fit_result.constants
        const_symbols = {name: sp.Symbol(name) for name in const_info}
        all_symbols = {"t": self.t_sym, **const_symbols}

        expr = sp.sympify(fit_result.equation_str, locals=all_symbols)
        func = sp.lambdify(
            [self.t_sym] + list(const_symbols.values()), expr, modules="numpy"
        )

        try:
            y_pred = func(t_obs, *const_info.values())
        except (ValueError, OverflowError):
            return {"mae": float("inf"), "max_error": float("inf"), "trend": "diverging"}

        errors = np.abs(y_pred - y_obs)
        mae = float(np.mean(errors))
        max_error = float(np.max(errors))

        # Determine error trend
        if len(errors) > 10:
            first_half = np.mean(errors[: len(errors) // 2])
            second_half = np.mean(errors[len(errors) // 2 :])
            if second_half > first_half * 1.5:
                trend = "growing"
            elif second_half < first_half * 0.5:
                trend = "shrinking"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {"mae": mae, "max_error": max_error, "trend": trend}
