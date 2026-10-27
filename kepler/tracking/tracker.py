"""
Object tracking module.

Extracts object trajectories from video using SAM 2 for high-accuracy
segmentation-based tracking, with OpenCV CSRT as the fallback when
SAM 2 or a CUDA GPU is not available.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np

from kepler.utils.config import KeplerConfig

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """A tracked object's trajectory through time."""
    object_id: int
    t: list[float] = field(default_factory=list)  # timestamps in seconds
    x: list[float] = field(default_factory=list)  # x positions (pixels)
    y: list[float] = field(default_factory=list)  # y positions (pixels)
    confidence: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
        }

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {
            "t": np.array(self.t),
            "x": np.array(self.x),
            "y": np.array(self.y),
        }


class VideoTracker:
    """
    Extract object trajectories from video.

    Backend selection (controlled by config.tracking.backend):
      - "auto": tries SAM 2 first, falls back to OpenCV CSRT
      - "sam2": SAM 2 only (errors if unavailable)
      - "opencv": OpenCV CSRT only
    """

    def __init__(self, config: KeplerConfig | None = None):
        self.config = config or KeplerConfig()
        self._sam_tracker = None

    def extract_frames(self, video_path: str | Path, max_frames: int = 6) -> list[np.ndarray]:
        """Extract evenly spaced frames from a video for Muse Spark."""
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    def save_frames(
        self, video_path: str | Path, output_dir: str | Path, max_frames: int = 6
    ) -> list[Path]:
        """Extract and save frames as JPEG files. Returns list of saved paths."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = self.extract_frames(video_path, max_frames)
        paths = []
        for i, frame in enumerate(frames):
            p = output_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(p), frame)
            paths.append(p)
        return paths

    def track_objects(self, video_path: str | Path) -> list[Trajectory]:
        """
        Track objects through a video and return their trajectories.

        Routes through the configured backend (SAM 2 or OpenCV).
        """
        # Use the auto-track path which handles backend selection
        return self.track_objects_auto(video_path)

    def track_objects_auto(
        self, video_path: str | Path, roi: tuple[int, int, int, int] | None = None
    ) -> list[Trajectory]:
        """
        Auto-track without GUI. Optionally pass an initial ROI (x, y, w, h).

        If no ROI is given, attempts simple motion-based detection to find
        the primary moving object.

        Backend selection follows config.tracking.backend:
          "auto" -> try SAM 2, fall back to OpenCV
          "sam2" -> SAM 2 only
          "opencv" -> OpenCV only
        """
        backend = self.config.tracking.backend

        # Try SAM 2 first if requested
        if backend in ("auto", "sam2"):
            from kepler.tracking.sam_tracker import is_sam2_available

            if is_sam2_available():
                logger.info("Using SAM 2 backend for tracking")
                return self._track_with_sam2(video_path, roi)
            elif backend == "sam2":
                raise RuntimeError(
                    "SAM 2 backend requested but not available. "
                    "Install sam2 and ensure CUDA GPU is present."
                )
            else:
                logger.info("SAM 2 not available, falling back to OpenCV CSRT")

        # OpenCV fallback
        logger.info("Using OpenCV CSRT backend for tracking")
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")

        if roi is None:
            roi = self._detect_moving_object(cap, first_frame)

        if roi is None:
            cap.release()
            raise ValueError("Could not detect a moving object. Provide an ROI manually.")

        trajectory = self._track_roi(cap, first_frame, roi, fps)
        cap.release()
        return [trajectory]

    def _track_with_sam2(
        self, video_path: str | Path, roi: tuple[int, int, int, int] | None = None
    ) -> list[Trajectory]:
        """Delegate tracking to SAM 2."""
        from kepler.tracking.sam_tracker import SAMTracker

        if self._sam_tracker is None:
            self._sam_tracker = SAMTracker(self.config)
        return self._sam_tracker.track_objects(video_path, roi)

    def _detect_moving_object(
        self, cap: cv2.VideoCapture, first_frame: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """Detect the primary moving object using frame differencing."""
        gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        gray_first = cv2.GaussianBlur(gray_first, (21, 21), 0)

        # Read a frame ~10% into the video for comparison
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total // 10, 5))
        ret, later_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)  # reset

        if not ret:
            return None

        gray_later = cv2.cvtColor(later_frame, cv2.COLOR_BGR2GRAY)
        gray_later = cv2.GaussianBlur(gray_later, (21, 21), 0)

        diff = cv2.absdiff(gray_first, gray_later)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Pick the largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        # Add some padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = w + 2 * pad
        h = h + 2 * pad
        return (x, y, w, h)

    def _track_roi(
        self,
        cap: cv2.VideoCapture,
        first_frame: np.ndarray,
        roi: tuple[int, int, int, int],
        fps: float,
    ) -> Trajectory:
        """Track a single ROI through the video using CSRT."""
        tracker = cv2.TrackerCSRT.create()
        tracker.init(first_frame, roi)

        traj = Trajectory(object_id=0)
        x, y, w, h = roi
        traj.t.append(0.0)
        traj.x.append(x + w / 2)
        traj.y.append(y + h / 2)
        traj.confidence.append(1.0)

        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            success, box = tracker.update(frame)
            if success:
                bx, by, bw, bh = [int(v) for v in box]
                traj.t.append(frame_idx / fps)
                traj.x.append(bx + bw / 2)
                traj.y.append(by + bh / 2)
                traj.confidence.append(1.0)
            else:
                traj.t.append(frame_idx / fps)
                traj.x.append(traj.x[-1])  # hold last known position
                traj.y.append(traj.y[-1])
                traj.confidence.append(0.0)

            frame_idx += 1

        return traj

    def _track_with_opencv(
        self, cap: cv2.VideoCapture, first_frame: np.ndarray, fps: float
    ) -> list[Trajectory]:
        """Interactive tracking with OpenCV (for local development)."""
        roi = self._detect_moving_object(cap, first_frame)
        if roi is None:
            raise ValueError(
                "Could not auto-detect a moving object. "
                "Use track_objects_auto() with a manual ROI."
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.read()  # skip first frame (already have it)
        return [self._track_roi(cap, first_frame, roi, fps)]
