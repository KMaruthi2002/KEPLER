"""
Prediction overlay module.

Takes the best-fit equation from the refinement loop and:
1. Generates future trajectory predictions (x and y independently)
2. Overlays predictions on live video frames
3. Shows confidence tier, equation annotations, and confidence bands
"""

import cv2
import numpy as np
import sympy as sp
from pathlib import Path

from kepler.simulation.simulator import FitResult, ConfidenceTier
from kepler.utils.config import KeplerConfig


# Colors for confidence tiers (BGR)
TIER_COLORS = {
    ConfidenceTier.DISCOVERED_LAW: (0, 220, 0),     # green
    ConfidenceTier.APPROXIMATE_FIT: (0, 180, 255),   # orange
    ConfidenceTier.TRACKING_ONLY: (0, 0, 220),       # red
}

TIER_LABELS = {
    ConfidenceTier.DISCOVERED_LAW: "DISCOVERED LAW",
    ConfidenceTier.APPROXIMATE_FIT: "APPROXIMATE FIT",
    ConfidenceTier.TRACKING_ONLY: "TRACKING ONLY",
}


class PredictionOverlay:
    """Overlay predicted trajectories onto video frames."""

    def __init__(self, config: KeplerConfig | None = None):
        self.config = config or KeplerConfig()
        self.t_sym = sp.Symbol("t")

    def _build_predictor(self, equation_str: str, constants: dict):
        """Build a callable from an equation string and fitted constants."""
        const_symbols = {name: sp.Symbol(name) for name in constants}
        all_symbols = {"t": self.t_sym, **const_symbols}

        expr = sp.sympify(equation_str, locals=all_symbols)
        func = sp.lambdify(
            [self.t_sym] + list(const_symbols.values()), expr, modules="numpy"
        )
        return func, list(constants.values())

    def predict_trajectory(
        self,
        fit_result: FitResult,
        t_start: float,
        t_end: float,
        num_points: int = 100,
    ) -> dict[str, np.ndarray]:
        """
        Generate predicted positions from t_start to t_end.

        Returns dict with 't', 'y', and optionally 'x' arrays.
        """
        t_pred = np.linspace(t_start, t_end, num_points)
        result = {"t": t_pred}

        # Y prediction (always available via the top-level equation)
        y_func, y_params = self._build_predictor(
            fit_result.equation_str, fit_result.constants
        )
        try:
            result["y"] = y_func(t_pred, *y_params)
        except (ValueError, OverflowError):
            result["y"] = np.full_like(t_pred, np.nan)

        # X prediction (only when x_fit is available)
        if fit_result.x_fit is not None:
            x_func, x_params = self._build_predictor(
                fit_result.x_fit.equation_str, fit_result.x_fit.constants
            )
            try:
                result["x"] = x_func(t_pred, *x_params)
            except (ValueError, OverflowError):
                result["x"] = np.full_like(t_pred, np.nan)

        return result

    def draw_overlay(
        self,
        frame: np.ndarray,
        observed_x: list[float],
        observed_y: list[float],
        predicted_y: np.ndarray,
        predicted_t: np.ndarray,
        fps: float,
        current_frame_idx: int,
        fit_result: FitResult,
        meter_to_pixel: float = 100.0,
        predicted_x: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Draw observed path, predicted path, equation, and confidence on a frame.

        Args:
            frame: The video frame (BGR).
            observed_x: Observed x positions in pixels.
            observed_y: Observed y positions in pixels.
            predicted_y: Predicted y values (in meters).
            predicted_t: Predicted time values (in seconds).
            fps: Video FPS for time-to-frame mapping.
            current_frame_idx: Current frame index in the video.
            fit_result: The equation fit result (for annotation).
            meter_to_pixel: Conversion factor back to pixels.
            predicted_x: Predicted x values (in meters), or None.

        Returns:
            Annotated frame.
        """
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # Draw observed trajectory (green dots)
        for i in range(len(observed_x)):
            px = int(observed_x[i])
            py = int(observed_y[i])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(overlay, (px, py), 3, (0, 255, 0), -1)

        # Draw observed trail (green line)
        trail_len = self.config.prediction.trail_length
        obs_points = list(zip(observed_x, observed_y))
        start_idx = max(0, len(obs_points) - trail_len)
        for i in range(start_idx, len(obs_points) - 1):
            pt1 = (int(obs_points[i][0]), int(obs_points[i][1]))
            pt2 = (int(obs_points[i + 1][0]), int(obs_points[i + 1][1]))
            cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)

        # Draw predicted trajectory (blue dots, or full x+y path)
        current_time = current_frame_idx / fps
        future_mask = predicted_t > current_time

        # Pick the confidence-tier color for predictions
        tier_color = TIER_COLORS.get(
            fit_result.confidence, (255, 100, 0)
        )

        if np.any(future_mask):
            future_t = predicted_t[future_mask]
            future_y = predicted_y[future_mask]

            if predicted_x is not None:
                # Full x+y predicted path
                future_x_m = predicted_x[future_mask]
                for i in range(len(future_t)):
                    px = int(future_x_m[i] * meter_to_pixel)
                    py = int(future_y[i] * meter_to_pixel)
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(overlay, (px, py), 2, tier_color, -1)
            else:
                # Y-only prediction: use last observed x as horizontal ref
                last_x = observed_x[-1] if observed_x else w / 2
                for i in range(len(future_t)):
                    py = int(future_y[i] * meter_to_pixel)
                    px = int(last_x)
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(overlay, (px, py), 2, tier_color, -1)

        # --- Confidence tier badge ---
        tier_label = TIER_LABELS.get(fit_result.confidence, "UNKNOWN")
        badge_color = TIER_COLORS.get(fit_result.confidence, (128, 128, 128))

        # Badge background
        (tw, th), _ = cv2.getTextSize(
            tier_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        badge_x, badge_y = 10, 10
        cv2.rectangle(
            overlay,
            (badge_x, badge_y),
            (badge_x + tw + 12, badge_y + th + 10),
            badge_color,
            -1,
        )
        cv2.putText(
            overlay,
            tier_label,
            (badge_x + 6, badge_y + th + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # --- Equation annotation ---
        eq_text = f"y(t) = {fit_result.equation_str}"
        const_text = ", ".join(
            f"{k}={v:.3f}" for k, v in fit_result.constants.items()
        )

        # If x-fit exists, show it too
        if fit_result.x_fit is not None:
            eq_text_x = f"x(t) = {fit_result.x_fit.equation_str}"
            cv2.putText(
                overlay,
                eq_text_x,
                (10, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            eq_text,
            (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            const_text,
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Draw residual
        if fit_result.residual is not None:
            mae_text = f"Residual: {fit_result.residual:.4f}"
            cv2.putText(
                overlay,
                mae_text,
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return overlay

    def render_video(
        self,
        video_path: str | Path,
        fit_result: FitResult,
        observed_x: list[float],
        observed_y: list[float],
        observed_t: list[float],
        output_path: str | Path,
        meter_to_pixel: float = 100.0,
    ) -> Path:
        """
        Render a full video with prediction overlay.

        Returns path to the output video.
        """
        video_path = str(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        # Generate predictions extending 50% beyond observed time
        t_max_obs = max(observed_t) if observed_t else total_frames / fps
        prediction = self.predict_trajectory(
            fit_result, 0, t_max_obs * 1.5, num_points=int(t_max_obs * 1.5 * fps)
        )

        predicted_x = prediction.get("x")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Build partial observed lists up to current frame
            current_time = frame_idx / fps
            obs_mask = [i for i, t in enumerate(observed_t) if t <= current_time]
            partial_x = [observed_x[i] for i in obs_mask]
            partial_y = [observed_y[i] for i in obs_mask]

            annotated = self.draw_overlay(
                frame,
                partial_x,
                partial_y,
                prediction["y"],
                prediction["t"],
                fps,
                frame_idx,
                fit_result,
                meter_to_pixel,
                predicted_x=predicted_x,
            )
            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()
        return output_path
