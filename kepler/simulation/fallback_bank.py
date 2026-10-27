"""
Fallback equation bank for unknown physics.

When Muse Spark's hypothesized equations all produce high residuals,
KEPLER falls back to a bank of generic mathematical templates --
polynomials, sinusoids, exponentials, etc. -- that can approximate
a wide range of physical motions without requiring domain knowledge.
"""

from __future__ import annotations

# Each entry follows the same schema as a Muse Spark candidate:
#   name, equation, variables, constants (with initial_guess), complexity
# Complexity is set higher than typical physics equations so Occam's
# razor scoring naturally prefers a Muse Spark hit when the fit is close.

FALLBACK_EQUATIONS: list[dict] = [
    # --- Polynomials (capture parabolic, cubic, and higher-order motion) ---
    {
        "name": "linear",
        "equation": "a0 + a1*t",
        "variables": {"t": "time"},
        "constants": {
            "a0": {"description": "intercept", "initial_guess": 0.0},
            "a1": {"description": "slope", "initial_guess": 1.0},
        },
        "complexity": 2,
        "family": "polynomial",
    },
    {
        "name": "quadratic",
        "equation": "a0 + a1*t + a2*t**2",
        "variables": {"t": "time"},
        "constants": {
            "a0": {"description": "intercept", "initial_guess": 0.0},
            "a1": {"description": "linear coeff", "initial_guess": 1.0},
            "a2": {"description": "quadratic coeff", "initial_guess": -1.0},
        },
        "complexity": 3,
        "family": "polynomial",
    },
    {
        "name": "cubic",
        "equation": "a0 + a1*t + a2*t**2 + a3*t**3",
        "variables": {"t": "time"},
        "constants": {
            "a0": {"description": "intercept", "initial_guess": 0.0},
            "a1": {"description": "linear coeff", "initial_guess": 1.0},
            "a2": {"description": "quadratic coeff", "initial_guess": 0.0},
            "a3": {"description": "cubic coeff", "initial_guess": 0.0},
        },
        "complexity": 4,
        "family": "polynomial",
    },
    {
        "name": "poly5",
        "equation": "a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5",
        "variables": {"t": "time"},
        "constants": {
            "a0": {"description": "intercept", "initial_guess": 0.0},
            "a1": {"description": "degree 1", "initial_guess": 1.0},
            "a2": {"description": "degree 2", "initial_guess": 0.0},
            "a3": {"description": "degree 3", "initial_guess": 0.0},
            "a4": {"description": "degree 4", "initial_guess": 0.0},
            "a5": {"description": "degree 5", "initial_guess": 0.0},
        },
        "complexity": 6,
        "family": "polynomial",
    },
    # --- Sinusoidal (oscillatory motion) ---
    {
        "name": "sinusoidal",
        "equation": "c0 + A*sin(omega*t + phi)",
        "variables": {"t": "time"},
        "constants": {
            "c0": {"description": "vertical offset", "initial_guess": 0.0},
            "A": {"description": "amplitude", "initial_guess": 1.0},
            "omega": {"description": "angular frequency", "initial_guess": 3.0},
            "phi": {"description": "phase", "initial_guess": 0.0},
        },
        "complexity": 3,
        "family": "oscillatory",
    },
    # --- Damped oscillation ---
    {
        "name": "damped_oscillation",
        "equation": "c0 + A*exp(-gamma*t)*sin(omega*t + phi)",
        "variables": {"t": "time"},
        "constants": {
            "c0": {"description": "equilibrium position", "initial_guess": 0.0},
            "A": {"description": "initial amplitude", "initial_guess": 1.0},
            "gamma": {"description": "damping coefficient", "initial_guess": 0.1},
            "omega": {"description": "angular frequency", "initial_guess": 3.0},
            "phi": {"description": "phase", "initial_guess": 0.0},
        },
        "complexity": 4,
        "family": "oscillatory",
    },
    # --- Exponential decay ---
    {
        "name": "exponential_decay",
        "equation": "c0 + A*exp(-k*t)",
        "variables": {"t": "time"},
        "constants": {
            "c0": {"description": "asymptotic value", "initial_guess": 0.0},
            "A": {"description": "initial displacement", "initial_guess": 1.0},
            "k": {"description": "decay rate", "initial_guess": 0.5},
        },
        "complexity": 3,
        "family": "exponential",
    },
    # --- Power law ---
    {
        "name": "power_law",
        "equation": "A*(t + t0)**n + c0",
        "variables": {"t": "time"},
        "constants": {
            "A": {"description": "coefficient", "initial_guess": 1.0},
            "t0": {"description": "time offset", "initial_guess": 0.1},
            "n": {"description": "exponent", "initial_guess": 1.5},
            "c0": {"description": "vertical offset", "initial_guess": 0.0},
        },
        "complexity": 4,
        "family": "power",
    },
    # --- Logistic / saturation ---
    {
        "name": "logistic",
        "equation": "L / (1 + exp(-k*(t - t0)))",
        "variables": {"t": "time"},
        "constants": {
            "L": {"description": "saturation level", "initial_guess": 5.0},
            "k": {"description": "growth rate", "initial_guess": 1.0},
            "t0": {"description": "midpoint time", "initial_guess": 1.0},
        },
        "complexity": 4,
        "family": "growth",
    },
]


def get_fallback_candidates() -> list[dict]:
    """Return a copy of the full fallback equation bank."""
    return [dict(c) for c in FALLBACK_EQUATIONS]
