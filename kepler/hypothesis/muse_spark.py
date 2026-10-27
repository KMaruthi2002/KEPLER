"""
Muse Spark Open client via Baseten (OpenAI-compatible API).

Muse Spark studies video frames and trajectory data, then proposes
candidate physical models as explicit equations.
"""

import base64
import json
from pathlib import Path

from openai import OpenAI

from kepler.utils.config import get_api_key, KeplerConfig


SYSTEM_PROMPT = """\
You are a physicist AI. You are given video frames showing a physical system
in motion, along with tracked object trajectories (x, y coordinates over time).

Your task:
1. Identify the physical phenomenon (e.g. projectile motion, pendulum, spring, etc.)
2. Propose candidate governing equations as explicit mathematical expressions
3. Identify the physical constants that need to be fitted (e.g. g, k, m, etc.)
4. Express each equation in a format that can be directly evaluated in Python using sympy

Return your response as JSON with this structure:
{
    "phenomenon": "description of what you observe",
    "candidates": [
        {
            "name": "equation name",
            "equation": "sympy-parseable equation string",
            "variables": {"t": "time", "x": "horizontal position", ...},
            "constants": {"g": {"description": "gravitational acceleration", "initial_guess": 9.81}},
            "complexity": 1
        }
    ],
    "reasoning": "your chain of thought"
}

Always propose at least 2 candidates: one simple and one more complex.
Prefer equations that are physically interpretable.
"""


class MuseSpark:
    """Client for Muse Spark Open on Baseten."""

    def __init__(self, config: KeplerConfig | None = None):
        self.config = config or KeplerConfig()
        self.client = OpenAI(
            base_url=self.config.hypothesis.base_url,
            api_key=get_api_key("BASETEN_API_KEY"),
        )
        self.model = self.config.hypothesis.model

    def _encode_frame(self, frame_path: str | Path) -> str:
        """Encode a video frame as base64 for the vision API."""
        with open(frame_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def propose_equations(
        self,
        frame_paths: list[str | Path],
        trajectory_data: dict,
        context: str = "",
    ) -> dict:
        """
        Send frames and trajectory data to Muse Spark, get candidate equations.

        Args:
            frame_paths: Paths to sampled video frames (3-6 recommended).
            trajectory_data: Dict with keys like "t", "x", "y" holding lists of floats.
            context: Optional additional context (e.g. error feedback from refinement).

        Returns:
            Parsed JSON with phenomenon, candidates, and reasoning.
        """
        # Build the message content with vision
        content = []

        # Add frames as images
        for fp in frame_paths:
            b64 = self._encode_frame(fp)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        # Add trajectory data as text
        trajectory_text = (
            f"Tracked trajectory data:\n"
            f"Time points: {len(trajectory_data.get('t', []))} samples\n"
            f"t (seconds): {trajectory_data.get('t', [])}\n"
            f"x (pixels): {trajectory_data.get('x', [])}\n"
            f"y (pixels): {trajectory_data.get('y', [])}\n"
        )
        if context:
            trajectory_text += f"\nAdditional context:\n{context}"

        content.append({"type": "text", "text": trajectory_text})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )

        # Parse the response
        raw = response.choices[0].message.content
        # Try to extract JSON from the response
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON block in markdown
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            raise ValueError(f"Could not parse Muse Spark response as JSON:\n{raw}")

    def refine_equation(
        self,
        frame_paths: list[str | Path],
        trajectory_data: dict,
        current_equation: dict,
        error_data: dict,
    ) -> dict:
        """
        Ask Muse Spark to refine an equation based on simulation error.

        Args:
            frame_paths: Paths to video frames.
            trajectory_data: Observed trajectory.
            current_equation: The current best equation dict.
            error_data: Dict with error metrics from simulation comparison.

        Returns:
            Revised equation candidates.
        """
        context = (
            f"REFINEMENT REQUEST:\n"
            f"Current equation: {current_equation.get('equation', 'unknown')}\n"
            f"Simulation error:\n"
            f"  Mean absolute error: {error_data.get('mae', 'N/A')}\n"
            f"  Max error: {error_data.get('max_error', 'N/A')}\n"
            f"  Error trend: {error_data.get('trend', 'N/A')}\n"
            f"\nThe current model drifts from reality. Please revise the equation "
            f"to better fit the observed data. Consider:\n"
            f"- Are there missing forces (friction, drag, damping)?\n"
            f"- Are the initial conditions correct?\n"
            f"- Is a different model family needed?"
        )
        return self.propose_equations(frame_paths, trajectory_data, context=context)
