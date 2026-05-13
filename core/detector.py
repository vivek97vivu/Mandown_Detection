from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single person detection result from YOLO."""
    track_id: int                        # -1 if tracking not enabled
    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2) absolute pixels
    confidence: float
    crop: np.ndarray = field(repr=False) # BGR crop of the person ROI


class PersonDetector:
    """
    Wraps Ultralytics YOLO for person-only detection on Jetson.

    Parameters
    ----------
    model_path : str | Path
        Path to .pt or TensorRT .engine weight file.
    conf_threshold : float
        Minimum confidence to keep a detection (default 0.4).
    iou_threshold : float
        NMS IoU threshold (default 0.45).
    input_size : int
        YOLO input resolution. Use 640 for accuracy, 416/320 for speed.
    device : str
        'cuda' (Jetson GPU) or 'cpu'.
    use_tracking : bool
        If True, uses YOLO's built-in ByteTrack for consistent IDs.
    crop_padding : float
        Fraction of bbox to pad on each side when cropping for RTMPose.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cuda",
        use_tracking: bool = True,
        crop_padding: float = 0.15,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.device = device
        self.use_tracking = use_tracking
        self.crop_padding = crop_padding
        self._use_fp16 = device == "cuda" and torch.cuda.is_available()

        self._model: YOLO | None = None
        self._frame_h: int = 0
        self._frame_w: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights. Call once before inference loop."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.model_path}")

        logger.info("Loading YOLO model from %s (fp16=%s)", self.model_path, self._use_fp16)
        self._model = YOLO(str(self.model_path))

        # Warm-up pass — critical on Jetson to avoid first-frame latency spike
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self._run_inference(dummy)
        logger.info("YOLO model loaded and warmed up.")

    def release(self) -> None:
        """Free GPU memory."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO on a BGR frame and return person detections.

        Parameters
        ----------
        frame : np.ndarray
            Full-resolution BGR frame from the camera.

        Returns
        -------
        List[Detection]
            One Detection per person found. Empty list if none.
        """
        if self._model is None:
            raise RuntimeError("Call load() before detect().")

        self._frame_h, self._frame_w = frame.shape[:2]
        results = self._run_inference(frame)

        detections: List[Detection] = []
        for result in results:
            boxes_data = (
                result.boxes.data.cpu().numpy()
                if result.boxes is not None
                else np.empty((0, 7))
            )
            # boxes_data columns: x1 y1 x2 y2 [track_id] conf cls
            # With tracking: 7 cols. Without: 6 cols.
            for row in boxes_data:
                if self.use_tracking and len(row) == 7:
                    x1, y1, x2, y2, track_id, conf, cls = row
                else:
                    x1, y1, x2, y2, conf, cls = row[:6]
                    track_id = -1

                if int(cls) != self.PERSON_CLASS_ID:
                    continue
                if conf < self.conf_threshold:
                    continue

                bbox = self._clamp_bbox(int(x1), int(y1), int(x2), int(y2))
                crop = self._extract_crop(frame, bbox)
                if crop.size == 0:
                    continue

                detections.append(
                    Detection(
                        track_id=int(track_id),
                        bbox=bbox,
                        confidence=float(conf),
                        crop=crop,
                    )
                )

        return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(self, frame: np.ndarray):
        """Call YOLO with or without ByteTrack."""
        common_kwargs = dict(
            classes=[self.PERSON_CLASS_ID],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            half=self._use_fp16,
            verbose=False,
        )
        if self.use_tracking:
            return self._model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                **common_kwargs,
            )
        return self._model.predict(frame, **common_kwargs)

    def _clamp_bbox(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
        """Clamp bbox to frame boundaries."""
        return (
            max(0, x1),
            max(0, y1),
            min(self._frame_w, x2),
            min(self._frame_h, y2),
        )

    def _extract_crop(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop the person ROI with padding for RTMPose.
        Padding ensures feet and head are included even with loose detections.
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        pad_x = int(w * self.crop_padding)
        pad_y = int(h * self.crop_padding)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(self._frame_w, x2 + pad_x)
        cy2 = min(self._frame_h, y2 + pad_y)

        return frame[cy1:cy2, cx1:cx2].copy()