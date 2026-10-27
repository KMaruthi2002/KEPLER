# KEPLER

**An AI scientist that discovers the laws of motion from video and predicts the future.**

KEPLER watches a video of a physical system in motion, extracts object trajectories, proposes candidate governing equations, fits physical constants to observed data, and overlays future predictions back onto the video.

## Pipeline

```
Video -> Track (SAM 3.1 / OpenCV) -> Hypothesize (Muse Spark) -> Simulate & Fit -> Refine -> Predict & Overlay
```

1. **Observe**: Extract object trajectories from video using SAM 3.1 (or OpenCV CSRT fallback)
2. **Hypothesize**: Send frames + trajectory data to Muse Spark Open (via Baseten) to propose candidate physics equations
3. **Experiment**: Convert equations into runnable simulators, fit physical constants (g, k, m, ...) to observed data using scipy optimization
4. **Refine**: Compare simulation to reality, feed error back to Muse Spark for equation revision (up to N iterations)
5. **Predict**: Overlay the winning equation's predictions onto the video as a live trajectory

Occam's razor applies: the simplest equation that explains the data wins.

## Quick Start

```bash
# Clone and install
git clone https://github.com/KMaruthi2002/KEPLER.git
cd KEPLER
pip install -e ".[dev]"

# Set up your API key
cp .env.example .env
# Edit .env and add your BASETEN_API_KEY

# Run the full pipeline on a video
kepler run path/to/video.mp4

# Or run individual steps
kepler track path/to/video.mp4
kepler hypothesize path/to/video.mp4 --trajectory trajectory.json
kepler predict path/to/video.mp4 --results results.json --trajectory trajectory.json
```

## Project Structure

```
kepler/
  __init__.py
  cli.py                  # CLI entry point
  utils/
    config.py             # Pydantic config, API key management
  tracking/
    tracker.py            # Object tracking (OpenCV CSRT, SAM 3.1 planned)
  hypothesis/
    muse_spark.py         # Muse Spark Open client (Baseten, OpenAI-compatible)
  simulation/
    simulator.py          # Equation-to-simulator, constant fitting, ranking
  refinement/
    loop.py               # Error-driven feedback loop
  prediction/
    overlay.py            # Video overlay with predicted trajectories
configs/
  default.yaml            # Default configuration
data/
  videos/                 # Input videos
  trajectories/           # Extracted trajectories
  results/                # Pipeline results
```

## Configuration

Edit `configs/default.yaml` or pass `--config path/to/config.yaml` to the CLI.

Key settings:
- `hypothesis.model`: Muse Spark model string (default: `meta-models/muse-spark-open-1026`)
- `hypothesis.base_url`: Baseten inference endpoint
- `refinement.max_iterations`: Max refinement cycles (default: 3)
- `refinement.error_threshold`: MAE threshold for convergence
- `refinement.simplicity_weight`: Occam's razor weight in ranking (0-1)

## Requirements

- Python 3.11+
- Baseten API key (for Muse Spark Open)
- OpenCV with tracking contrib modules

## Team

**Parallax** - Meta Global AI Developer Hackathon 2026

## License

MIT
