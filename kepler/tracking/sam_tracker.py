"""
SAM 2 video object tracking module.

Uses Meta's Segment Anything Model 2 for accurate object segmentation
and tracking across video frames. Falls back gracefully when SAM 2
is not installed or GPU is unavailable.
"""

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np

from kepler.tracking.tracker import Trajectory
from kepler.utils.config import KeplerConfig

logger = logging.getLogger(__name__)

# Lazy import flag -- SAM 2 is optional
_sam2_available = None


def is_sam2_available() -> bool:
    """Check if SAM 2 is installed and usable."""
    global _sam2_available
    if _sam2_available is None:
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor  # noqa: F401

            _sam2_available = torch.cuda.is_available()
            if not _sam2_available:
                logger.info("SAM 2 found but no CUDA GPU available. Falling back to OpenCV.")
        except ImportError:
            _sam2_available = False
            logger.info("SAM 2 not installed. Using OpenCV tracker.")
    return _sam2_available


def _mask_to_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Convert a binary mask to its centroid (x, y) in pixel coordinates."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Convert a binary mask to a bounding box (x, y, w, h)."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    return (x_min, y_min, x_max - x_min, y_max - y_min)


class SAMTracker:
    """
    Track objects through video using SAM 2 video predictor.

    Workflow:
    1. Extract all frames to a temp JPEG directory
    2. Initialize SAM 2 video predictor with the checkpoint
    3. Use the first-frame ROI as a box prompt
    4. Propagate through all frames
    5. Convert per-frame masks to centroid (x, y) trajectory
    """

    def __init__(self, config: KeplerConfig | None = None):
        self.config = config or KeplerConfig()
        self._predictor = None
        self._device = None

    def _ensure_predictor(self):
        """Lazy-load the SAM 2 video predictor."""
        if self._predictor is not None:
            return

        import torch
        from sam2.build_sam import build_sam2_video_predictor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = self.config.tracking.sam2_checkpoint

        logger.info("Loading SAM 2 video predictor: %s", checkpoint)
        self._predictor = build_sam2_video_predictor(
            checkpoint,
            device=self._device,
        )
        logger.info("SAM 2 predictor loaded on %s", self._device)

    def _extract_all_frames(self, video_path: str, output_dir: Path) -> tuple[int, float]:
        """
        Extract ALL frames from a video as sequentially numbered JPEGs.

        SAM 2 requires frames as a directory of numbered images.
        Returns (total_frames, fps).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_path = output_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            frame_idx += 1

        cap.release()
        logger.info("Extracted %d frames at %.1f fps to %s", frame_idx, fps, output_dir)
        return frame_idx, fps

    def track_objects(
        self,
        video_path: str | Path,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[Trajectory]:
        """
        Track objects through video using SAM 2.

        Args:
            video_path: Path to the input video.
            roi: Optional initial bounding box (x, y, w, h) for the object.
                 If None, uses motion detection to find the primary object.

        Returns:
            List of Trajectory objects.
        """
        import torch

        self._ensure_predictor()
        video_path = str(video_path)

        # Extract frames to a temp directory
        with tempfile.TemporaryDirectory(prefix="kepler_sam_") as tmp_dir:
            frames_dir = Path(tmp_dir) / "frames"
            total_frames, fps = self._extract_all_frames(video_path, frames_dir)

            if total_frames == 0:
                raise ValueError("Video has no frames")

            # If no ROI provided, detect moving object from first frames
            if roi is None:
                roi = self._detect_moving_object(video_path)
                if roi is None:
                    raise ValueError(
                        "Could not auto-detect a moving object. Provide an ROI."
                    )

            logger.info("Tracking with ROI: %s", roi)

            # Initialize SAM 2 video state
            inference_state = self._predictor.init_state(
                video_path=str(frames_dir),
            )

            # Add box prompt on frame 0
            x, y, w, h = roi
            box = np.array([x, y, x + w, y + h], dtype=np.float32)

            _, obj_ids, mask_logits = self._predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=0,
                box=box,
            )

            logger.info("Initialized tracking for object %s", obj_ids)

            # Propagate through all frames
            traj = Trajectory(object_id=0)

            for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(
                inference_state
            ):
                # mask_logits shape: (num_objects, 1, H, W)
                mask = (mask_logits[0] > 0.0).cpu().numpy()
                centroid = _mask_to_centroid(mask)

                t = frame_idx / fps

                if centroid is not None:
                    cx, cy = centroid
                    traj.t.append(t)
                    traj.x.append(cx)
                    traj.y.append(cy)
                    traj.confidence.append(1.0)
                else:
                    # Object lost -- hold last position
                    traj.t.append(t)
                    traj.x.append(traj.x[-1] if traj.x else 0.0)
                    traj.y.append(traj.y[-1] if traj.y else 0.0)
                    traj.confidence.append(0.0)

            # Reset state for next video
            self._predictor.reset_state(inference_state)

        logger.info(
            "SAM 2 tracking complete: %d points over %.1f seconds",
            len(traj.t),
            traj.t[-1] if traj.t else 0,
        )
        return [traj]

    def _detect_moving_object(
        self, video_path: str
    ) -> tuple[int, int, int, int] | None:
        """Detect the primary moving object using frame differencing."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return None

        gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        gray_first = cv2.GaussianBlur(gray_first, (21, 21), 0)

        # Read a frame ~10% into the video
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total // 10, 5))
        ret, later_frame = cap.read()
        cap.release()

        if not ret:
            return None

        gray_later = cv2.cvtColor(later_frame, cv2.COLOR_BGR2GRAY)
        gray_later = cv2.GaussianBlur(gray_later, (21, 21), 0)

        diff = cv2.absdiff(gray_first, gray_later)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = w + 2 * pad
        h = h + 2 * pad
        return (x, y, w, h)
