"""
KEPLER CLI entry point.

Usage:
    kepler track <video>           Extract object trajectories
    kepler hypothesize <video>     Propose equations from video
    kepler run <video>             Full pipeline: track -> hypothesize -> fit -> refine -> predict
    kepler predict <video>         Overlay predictions on video
"""

import json
import logging
import sys
from pathlib import Path

import click

from kepler.utils.config import load_config, KeplerConfig


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--config", "-c", default=None, help="Path to config YAML file.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging.")
@click.pass_context
def cli(ctx, config, verbose):
    """KEPLER: An AI scientist that discovers the laws of motion from video."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output JSON path for trajectories.")
@click.option("--roi", default=None, help="Initial ROI as 'x,y,w,h' (pixels).")
@click.pass_context
def track(ctx, video, output, roi):
    """Extract object trajectories from a video."""
    from kepler.tracking.tracker import VideoTracker

    config: KeplerConfig = ctx.obj["config"]
    tracker = VideoTracker(config)

    parsed_roi = None
    if roi:
        parts = [int(x.strip()) for x in roi.split(",")]
        if len(parts) == 4:
            parsed_roi = tuple(parts)

    click.echo(f"Tracking objects in: {video}")
    trajectories = tracker.track_objects_auto(video, roi=parsed_roi)
    click.echo(f"Found {len(trajectories)} trajectory(ies).")

    # Save results
    if output is None:
        output = str(Path(video).with_suffix(".trajectory.json"))

    data = [t.to_dict() for t in trajectories]
    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    click.echo(f"Trajectories saved to: {output}")


@cli.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--frames", "-n", default=6, help="Number of frames to sample.")
@click.option("--trajectory", "-t", default=None, help="Path to trajectory JSON (if already tracked).")
@click.option("--output", "-o", default=None, help="Output JSON path for hypotheses.")
@click.pass_context
def hypothesize(ctx, video, frames, trajectory, output):
    """Propose physics equations from video frames and trajectory data."""
    from kepler.tracking.tracker import VideoTracker
    from kepler.hypothesis.muse_spark import MuseSpark

    config: KeplerConfig = ctx.obj["config"]

    # Extract frames
    tracker = VideoTracker(config)
    frame_dir = Path(video).parent / "frames"
    click.echo(f"Extracting {frames} frames...")
    frame_paths = tracker.save_frames(video, frame_dir, max_frames=frames)

    # Load or extract trajectory
    if trajectory:
        with open(trajectory) as f:
            traj_data = json.load(f)
        if isinstance(traj_data, list):
            traj_data = traj_data[0]  # use first trajectory
    else:
        click.echo("No trajectory provided. Tracking objects first...")
        trajs = tracker.track_objects_auto(video)
        traj_data = trajs[0].to_dict()

    # Call Muse Spark
    click.echo("Sending to Muse Spark for hypothesis generation...")
    muse = MuseSpark(config)
    result = muse.propose_equations(frame_paths, traj_data)

    click.echo(f"Phenomenon: {result.get('phenomenon', 'unknown')}")
    for cand in result.get("candidates", []):
        click.echo(f"  - {cand.get('name', '?')}: {cand.get('equation', '?')}")

    if output is None:
        output = str(Path(video).with_suffix(".hypothesis.json"))
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    click.echo(f"Hypotheses saved to: {output}")


@cli.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None, help="Output directory for all results.")
@click.option("--roi", default=None, help="Initial ROI as 'x,y,w,h' (pixels).")
@click.option("--scale", default=0.01, help="Pixel-to-meter scale factor.")
@click.pass_context
def run(ctx, video, output_dir, roi, scale):
    """Run the full KEPLER pipeline on a video."""
    from kepler.tracking.tracker import VideoTracker
    from kepler.refinement.loop import RefinementLoop

    config: KeplerConfig = ctx.obj["config"]

    if output_dir is None:
        output_dir = str(Path(video).parent / "kepler_results")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Track
    click.echo("Step 1/4: Tracking objects...")
    tracker = VideoTracker(config)
    parsed_roi = None
    if roi:
        parts = [int(x.strip()) for x in roi.split(",")]
        if len(parts) == 4:
            parsed_roi = tuple(parts)

    trajectories = tracker.track_objects_auto(video, roi=parsed_roi)
    traj = trajectories[0]
    click.echo(f"  Tracked {len(traj.t)} points.")

    # Save trajectory
    traj_path = output_dir / "trajectory.json"
    with open(traj_path, "w") as f:
        json.dump(traj.to_dict(), f, indent=2)

    # Step 2: Extract frames
    click.echo("Step 2/4: Extracting frames...")
    frame_dir = output_dir / "frames"
    frame_paths = tracker.save_frames(video, frame_dir, max_frames=6)
    click.echo(f"  Saved {len(frame_paths)} frames.")

    # Step 3: Hypothesize + Refine
    click.echo("Step 3/4: Hypothesis generation and refinement loop...")
    loop = RefinementLoop(config)
    result = loop.run(frame_paths, traj, pixel_to_meter=scale)

    if result.best_fit:
        click.echo(f"  Best equation: {result.best_fit.equation_str}")
        click.echo(f"  Constants: {result.best_fit.constants}")
        click.echo(f"  Converged: {result.converged} ({result.num_iterations} iterations)")

        # Save results
        results_data = {
            "equation": result.best_fit.equation_str,
            "equation_name": result.best_fit.equation_name,
            "constants": result.best_fit.constants,
            "residual": result.best_fit.residual,
            "converged": result.converged,
            "iterations": result.iterations,
        }
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        # Step 4: Prediction overlay
        click.echo("Step 4/4: Rendering prediction overlay...")
        from kepler.prediction.overlay import PredictionOverlay

        overlay = PredictionOverlay(config)
        output_video = output_dir / "prediction.mp4"
        overlay.render_video(
            video, result.best_fit,
            traj.x, traj.y, traj.t,
            output_video,
            meter_to_pixel=1.0 / scale,
        )
        click.echo(f"  Output video: {output_video}")
    else:
        click.echo("  No viable equation found.")

    click.echo("Done.")


@cli.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--results", "-r", required=True, help="Path to results.json from a previous run.")
@click.option("--trajectory", "-t", required=True, help="Path to trajectory.json.")
@click.option("--output", "-o", default=None, help="Output video path.")
@click.option("--scale", default=0.01, help="Pixel-to-meter scale factor.")
@click.pass_context
def predict(ctx, video, results, trajectory, output, scale):
    """Overlay equation predictions onto a video."""
    from kepler.prediction.overlay import PredictionOverlay
    from kepler.simulation.simulator import FitResult

    config: KeplerConfig = ctx.obj["config"]

    # Load results
    with open(results) as f:
        res = json.load(f)

    with open(trajectory) as f:
        traj = json.load(f)

    fit = FitResult(
        equation_name=res.get("equation_name", "loaded"),
        equation_str=res["equation"],
        constants=res["constants"],
        residual=res.get("residual", 0),
        predicted_trajectory={},
        complexity=res.get("complexity", 1),
    )

    if output is None:
        output = str(Path(video).with_suffix(".prediction.mp4"))

    overlay = PredictionOverlay(config)
    overlay.render_video(
        video, fit,
        traj["x"], traj["y"], traj["t"],
        output,
        meter_to_pixel=1.0 / scale,
    )
    click.echo(f"Prediction video saved to: {output}")


def main():
    cli()


if __name__ == "__main__":
    main()
