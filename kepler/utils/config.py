"""Configuration management for KEPLER."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TrackingConfig(BaseModel):
    backend: str = "auto"  # "auto", "sam2", "opencv"
    sam2_checkpoint: str = "facebook/sam2.1-hiera-large"
    confidence_threshold: float = 0.8
    max_objects: int = 10
    fps_target: int = 30


class HypothesisConfig(BaseModel):
    model: str = "meta-models/muse-spark-open-1026"
    base_url: str = "https://inference.baseten.co/v1"
    max_candidates: int = 5
    complexity_penalty: float = 0.1


class SimulationConfig(BaseModel):
    dt: float = 0.01
    max_horizon: float = 5.0
    fit_max_iter: int = 1000


class RefinementConfig(BaseModel):
    max_iterations: int = 3
    error_threshold: float = 0.05
    simplicity_weight: float = 0.3


class PredictionConfig(BaseModel):
    overlay_fps: int = 30
    trail_length: int = 60
    confidence_display: bool = True


class KeplerConfig(BaseModel):
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    hypothesis: HypothesisConfig = Field(default_factory=HypothesisConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)


def load_config(path: str | Path | None = None) -> KeplerConfig:
    """Load configuration from YAML file, falling back to defaults."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    path = Path(path)

    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return KeplerConfig(**data)

    return KeplerConfig()


def get_api_key(name: str) -> str:
    """Get an API key from environment variables."""
    key = os.environ.get(name)
    if not key:
        raise ValueError(
            f"Missing environment variable: {name}. "
            f"Copy .env.example to .env and fill in your keys."
        )
    return key
