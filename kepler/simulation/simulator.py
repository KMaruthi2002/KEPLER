"""
Physics simulator module.

Takes candidate equations from Muse Spark and:
1. Converts them into runnable simulators
2. Fits physical constants to observed trajectory data (x and y independently)
3. Produces predicted trajectories for comparison
4. Falls back to generic equation templates when physics candidates fail
5. Reports confidence tiers for fit quality
"""

import logging
import numpy as np
from scipy.optimize import minimize
import sympy as sp
from dataclasses import dataclass, field
from enum import Enum

from kepler.tracking.tracker import Trajectory
from kepler.simulation.fallback_bank import get_fallback_candidates

logger = logging.getLogger(__name__)


class ConfidenceTier(str, Enum):
    """How much confidence KEPLER has in the discovered equation."""

    DISCOVERED_LAW = "discovered_law"
    """Low residual from a physics-based equation. High confidence."""

    APPROXIMATE_FIT = "approximate_fit"
    """A generic/fallback equation matched reasonably. Medium confidence."""

    TRACKING_ONLY = "tracking_only"
    """No equation fit well. Only raw trajectory data is reliable."""


# Default residual thresholds (per data point, in meters^2).
# These are applied to the mean-squared error per point.
RESIDUAL_GOOD = 0.05       # below this -> discovered_law
RESIDUAL_FALLBACK = 0.5    # below this -> approximate_fit, above -> tracking_only


@dataclass
class AxisFitResult:
    """Result of fitting an equation to one axis (x or y)."""

    equation_name: str
    equation_str: str
    constants: dict[str, float]
    residual: float
    predicted: np.ndarray  # predicted values at prediction times
    predicted_t: np.ndarray
    complexity: int
    from_fallback: bool = False


@dataclass
class FitResult:
    """Result of fitting equations to observed data.

    Backwards-compatible: the y_fit field matches the old single-axis interface,
    and .equation_str / .constants / .residual / .predicted_trajectory still
    work for code that reads only the y-axis result.
    """

    equation_name: str
    equation_str: str
    constants: dict[str, float]
    residual: float  # total fit error (y-axis for backwards compat)
    predicted_trajectory: dict[str, np.ndarray]
    complexity: int

    # New multi-axis fields
    y_fit: AxisFitResult | None = None
    x_fit: AxisFitResult | None = None
    confidence: ConfidenceTier = ConfidenceTier.TRACKING_ONLY
    from_fallback: bool = False


class PhysicsSimulator:
    """Fit and simulate physical equations against observed trajectories."""

    def __init__(self):
        self.t_sym = sp.Symbol("t")

    # ------------------------------------------------------------------
    # Single-axis fitting (the core engine)
    # ------------------------------------------------------------------

    def _fit_single_axis(
        self,
        candidate: dict,
        t_obs: np.ndarray,
        val_obs: np.ndarray,
    ) -> AxisFitResult:
        """Fit a candidate equation against a single axis of data."""
        eq_str = candidate["equation"]
        constants_info = candidate.get("constants", {})

        const_symbols = {}
        initial_guesses = []
        const_names = []

        for name, info in constants_info.items():
            const_symbols[name] = sp.Symbol(name)
            initial_guesses.append(info.get("initial_guess", 1.0))
            const_names.append(name)

        all_symbols = {"t": self.t_sym, **const_symbols}
        try:
            expr = sp.sympify(eq_str, locals=all_symbols)
        except (sp.SympifyError, SyntaxError) as e:
            raise ValueError(f"Cannot parse equation '{eq_str}': {e}")

        func = sp.lambdify(
            [self.t_sym] + list(const_symbols.values()), expr, modules="numpy"
        )

        def residual_fn(params):
            try:
                pred = func(t_obs, *params)
                return np.sum((pred - val_obs) ** 2)
            except (ValueError, OverflowError, ZeroDivisionError):
                return 1e12

        result = minimize(
            residual_fn,
            x0=initial_guesses,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
        )

        fitted_constants = dict(zip(const_names, result.x))

        t_pred = np.linspace(t_obs[0], t_obs[-1] * 1.5, len(t_obs) * 2)
        try:
            val_pred = func(t_pred, *result.x)
        except (ValueError, OverflowError):
            val_pred = np.full_like(t_pred, np.nan)

        return AxisFitResult(
            equation_name=candidate.get("name", "unknown"),
            equation_str=eq_str,
            constants=fitted_constants,
            residual=float(result.fun),
            predicted=val_pred,
            predicted_t=t_pred,
            complexity=candidate.get("complexity", 1),
            from_fallback=candidate.get("_from_fallback", False),
        )

    # ------------------------------------------------------------------
    # Public API: fit_equation (backwards compatible, y-axis only)
    # ------------------------------------------------------------------

    def fit_equation(
        self,
        candidate: dict,
        trajectory: Trajectory,
        pixel_to_meter: float = 0.01,
    ) -> FitResult:
        """
        Fit a candidate equation's constants to the y-axis of observed data.

        This is the original single-axis interface kept for backwards
        compatibility.  Use ``fit_equation_xy`` for independent x+y fitting.
        """
        obs = trajectory.to_numpy()
        t_obs = obs["t"]
        y_obs = obs["y"] * pixel_to_meter

        axis_result = self._fit_single_axis(candidate, t_obs, y_obs)

        return FitResult(
            equation_name=axis_result.equation_name,
            equation_str=axis_result.equation_str,
            constants=axis_result.constants,
            residual=axis_result.residual,
            predicted_trajectory={
                "t": axis_result.predicted_t,
                "y": axis_result.predicted,
            },
            complexity=axis_result.complexity,
            y_fit=axis_result,
            confidence=self._classify_confidence(axis_result, len(t_obs)),
            from_fallback=axis_result.from_fallback,
        )

    # ------------------------------------------------------------------
    # New API: fit both x and y independently
    # ------------------------------------------------------------------

    def fit_equation_xy(
        self,
        candidate_y: dict,
        candidate_x: dict | None,
        trajectory: Trajectory,
        pixel_to_meter: float = 0.01,
    ) -> FitResult:
        """
        Fit equations to both x(t) and y(t) independently.

        Args:
            candidate_y: Equation hypothesis for the y-axis.
            candidate_x: Equation hypothesis for the x-axis (optional).
            trajectory: Observed trajectory from tracking.
            pixel_to_meter: Scale factor for pixel-to-meter conversion.

        Returns:
            FitResult with both axis fits and a combined confidence tier.
        """
        obs = trajectory.to_numpy()
        t_obs = obs["t"]
        y_obs = obs["y"] * pixel_to_meter
        x_obs = obs["x"] * pixel_to_meter

        y_fit = self._fit_single_axis(candidate_y, t_obs, y_obs)

        x_fit = None
        if candidate_x is not None:
            x_fit = self._fit_single_axis(candidate_x, t_obs, x_obs)

        # Combined residual is sum of both axes
        combined_residual = y_fit.residual
        if x_fit is not None:
            combined_residual += x_fit.residual

        # Build predicted trajectory dict
        pred_traj: dict[str, np.ndarray] = {
            "t": y_fit.predicted_t,
            "y": y_fit.predicted,
        }
        if x_fit is not None:
            pred_traj["x"] = x_fit.predicted

        # Confidence is the worst of the two axes
        y_conf = self._classify_confidence(y_fit, len(t_obs))
        if x_fit is not None:
            x_conf = self._classify_confidence(x_fit, len(t_obs))
            confidence = self._worst_confidence(y_conf, x_conf)
        else:
            confidence = y_conf

        return FitResult(
            equation_name=y_fit.equation_name,
            equation_str=y_fit.equation_str,
            constants=y_fit.constants,
            residual=combined_residual,
            predicted_trajectory=pred_traj,
            complexity=y_fit.complexity,
            y_fit=y_fit,
            x_fit=x_fit,
            confidence=confidence,
            from_fallback=y_fit.from_fallback or (
                x_fit.from_fallback if x_fit else False
            ),
        )

    # ------------------------------------------------------------------
    # Fallback: try generic equations when candidates fail
    # ------------------------------------------------------------------

    def fit_with_fallback(
        self,
        candidates: list[dict],
        trajectory: Trajectory,
        pixel_to_meter: float = 0.01,
        residual_threshold: float = RESIDUAL_FALLBACK,
        axis: str = "y",
    ) -> list[FitResult]:
        """
        Fit candidates; if all are above threshold, try the fallback bank.

        Args:
            candidates: Primary equation hypotheses (from Muse Spark).
            trajectory: Observed trajectory.
            pixel_to_meter: Scale factor.
            residual_threshold: Per-point MSE above which fallbacks fire.
            axis: Which axis to fit ("y", "x", or "both").

        Returns:
            List of FitResults, sorted best-first.  Results from the fallback
            bank have ``from_fallback=True``.
        """
        obs = trajectory.to_numpy()
        n_points = len(obs["t"])

        # --- Phase 1: fit primary candidates ---
        primary_results: list[FitResult] = []
        for cand in candidates:
            try:
                if axis == "both":
                    # Use y candidate for y, look for a matching x candidate
                    x_cand = cand.get("_x_equation")
                    fit = self.fit_equation_xy(cand, x_cand, trajectory, pixel_to_meter)
                else:
                    fit = self.fit_equation(cand, trajectory, pixel_to_meter)
                primary_results.append(fit)
            except (ValueError, Exception) as e:
                logger.warning("Could not fit '%s': %s", cand.get("name", "?"), e)

        # Check if any primary result is good enough
        per_point_threshold = residual_threshold * n_points
        good_enough = any(r.residual < per_point_threshold for r in primary_results)

        if good_enough or not primary_results:
            # If primary results exist and are good, or if there are none
            # (fallback won't help if we can't even parse), return them
            if primary_results:
                return self.rank_candidates(primary_results)

        # --- Phase 2: fallback bank ---
        logger.info(
            "All %d primary candidates have high residuals. "
            "Trying fallback equation bank...",
            len(primary_results),
        )

        fallback_candidates = get_fallback_candidates()
        # Mark them as fallback so we can track provenance
        for fb in fallback_candidates:
            fb["_from_fallback"] = True

        fallback_results: list[FitResult] = []
        for cand in fallback_candidates:
            try:
                fit = self.fit_equation(cand, trajectory, pixel_to_meter)
                fallback_results.append(fit)
            except (ValueError, Exception) as e:
                logger.debug("Fallback '%s' failed: %s", cand.get("name", "?"), e)

        # Merge and rank everything together
        all_results = primary_results + fallback_results
        return self.rank_candidates(all_results)

    # ------------------------------------------------------------------
    # Confidence classification
    # ------------------------------------------------------------------

    def _classify_confidence(
        self,
        axis_fit: AxisFitResult,
        n_points: int,
    ) -> ConfidenceTier:
        """Classify a single-axis fit into a confidence tier."""
        if n_points == 0:
            return ConfidenceTier.TRACKING_ONLY

        mse_per_point = axis_fit.residual / n_points

        if mse_per_point < RESIDUAL_GOOD and not axis_fit.from_fallback:
            return ConfidenceTier.DISCOVERED_LAW
        elif mse_per_point < RESIDUAL_FALLBACK:
            return ConfidenceTier.APPROXIMATE_FIT
        else:
            return ConfidenceTier.TRACKING_ONLY

    @staticmethod
    def _worst_confidence(a: ConfidenceTier, b: ConfidenceTier) -> ConfidenceTier:
        """Return the less confident of two tiers."""
        order = [
            ConfidenceTier.TRACKING_ONLY,
            ConfidenceTier.APPROXIMATE_FIT,
            ConfidenceTier.DISCOVERED_LAW,
        ]
        return order[min(order.index(a), order.index(b))]

    # ------------------------------------------------------------------
    # Ranking (unchanged from original)
    # ------------------------------------------------------------------

    def rank_candidates(
        self,
        fit_results: list[FitResult],
        simplicity_weight: float = 0.3,
    ) -> list[FitResult]:
        """
        Rank fitted equations by fit quality + simplicity (Occam's razor).

        Score = (1 - w) * normalized_residual + w * normalized_complexity
        Lower is better.
        """
        if not fit_results:
            return []

        residuals = np.array([r.residual for r in fit_results])
        complexities = np.array([r.complexity for r in fit_results])

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

        scores = (
            (1 - simplicity_weight) * norm_residuals
            + simplicity_weight * norm_complexity
        )

        ranked_indices = np.argsort(scores)
        return [fit_results[i] for i in ranked_indices]

    # ------------------------------------------------------------------
    # Error computation (updated for multi-axis)
    # ------------------------------------------------------------------

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
            return {
                "mae": float("inf"),
                "max_error": float("inf"),
                "trend": "diverging",
                "confidence": fit_result.confidence.value,
            }

        errors = np.abs(y_pred - y_obs)
        mae = float(np.mean(errors))
        max_error = float(np.max(errors))

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

        return {
            "mae": mae,
            "max_error": max_error,
            "trend": trend,
            "confidence": fit_result.confidence.value,
        }
