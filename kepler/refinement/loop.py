"""
Refinement loop module.

Implements the error-driven feedback cycle:
1. Compare simulation predictions to observed trajectories
2. Feed error data back to Muse Spark for equation revision
3. Repeat until convergence or max iterations
"""

import logging
from pathlib import Path

from kepler.tracking.tracker import Trajectory
from kepler.hypothesis.muse_spark import MuseSpark
from kepler.simulation.simulator import PhysicsSimulator, FitResult
from kepler.utils.config import KeplerConfig

logger = logging.getLogger(__name__)


class RefinementResult:
    """Stores the outcome of a full refinement run."""

    def __init__(self):
        self.iterations: list[dict] = []
        self.best_fit: FitResult | None = None
        self.converged: bool = False
        self.final_error: dict | None = None

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)


class RefinementLoop:
    """
    Error-driven refinement loop.

    Observe -> Hypothesize -> Simulate -> Compare -> Refine -> repeat.
    """

    def __init__(self, config: KeplerConfig | None = None):
        self.config = config or KeplerConfig()
        self.muse = MuseSpark(self.config)
        self.simulator = PhysicsSimulator()

    def run(
        self,
        frame_paths: list[str | Path],
        trajectory: Trajectory,
        pixel_to_meter: float = 0.01,
    ) -> RefinementResult:
        """
        Run the full refinement loop.

        Args:
            frame_paths: Paths to extracted video frames.
            trajectory: Observed trajectory from tracking.
            pixel_to_meter: Scale factor for pixel-to-meter conversion.

        Returns:
            RefinementResult with the best equation and fit history.
        """
        result = RefinementResult()
        max_iter = self.config.refinement.max_iterations
        error_threshold = self.config.refinement.error_threshold
        simplicity_weight = self.config.refinement.simplicity_weight

        traj_data = trajectory.to_dict()

        # Initial hypothesis
        logger.info("Requesting initial equation hypotheses from Muse Spark...")
        hypothesis = self.muse.propose_equations(frame_paths, traj_data)
        candidates = hypothesis.get("candidates", [])

        if not candidates:
            logger.warning("Muse Spark returned no candidates.")
            return result

        logger.info(
            "Received %d candidate(s). Phenomenon: %s",
            len(candidates),
            hypothesis.get("phenomenon", "unknown"),
        )

        for iteration in range(max_iter):
            logger.info("--- Refinement iteration %d/%d ---", iteration + 1, max_iter)

            # Fit all candidates
            fit_results = []
            for cand in candidates:
                try:
                    fit = self.simulator.fit_equation(cand, trajectory, pixel_to_meter)
                    fit_results.append(fit)
                    logger.info(
                        "  Fitted '%s': residual=%.6f",
                        fit.equation_name,
                        fit.residual,
                    )
                except (ValueError, Exception) as e:
                    logger.warning("  Could not fit '%s': %s", cand.get("name", "?"), e)

            if not fit_results:
                logger.warning("No candidates could be fitted. Stopping.")
                break

            # Rank by fit + simplicity
            ranked = self.simulator.rank_candidates(fit_results, simplicity_weight)
            best = ranked[0]
            error = self.simulator.compute_error(best, trajectory, pixel_to_meter)

            iteration_record = {
                "iteration": iteration + 1,
                "best_equation": best.equation_str,
                "best_name": best.equation_name,
                "constants": best.constants,
                "residual": best.residual,
                "error": error,
            }
            result.iterations.append(iteration_record)
            result.best_fit = best
            result.final_error = error

            logger.info(
                "  Best: '%s' | MAE=%.6f | trend=%s",
                best.equation_name,
                error["mae"],
                error["trend"],
            )

            # Check convergence
            if error["mae"] < error_threshold:
                logger.info("Converged! MAE below threshold (%.6f < %.6f).", error["mae"], error_threshold)
                result.converged = True
                break

            # Request refinement from Muse Spark
            if iteration < max_iter - 1:
                logger.info("Requesting refined hypotheses...")
                current_eq = {
                    "equation": best.equation_str,
                    "name": best.equation_name,
                    "constants": best.constants,
                }
                refined = self.muse.refine_equation(
                    frame_paths, traj_data, current_eq, error
                )
                candidates = refined.get("candidates", [])
                if not candidates:
                    logger.warning("Muse Spark returned no refined candidates. Stopping.")
                    break

        if not result.converged:
            logger.info(
                "Refinement did not converge after %d iterations. Best MAE=%.6f",
                result.num_iterations,
                result.final_error["mae"] if result.final_error else float("inf"),
            )

        return result
