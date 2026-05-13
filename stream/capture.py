"""
capture.py
----------
Unified frame capture interface for both local (USB/webcam) and
RTSP cameras. Wraps cv2.VideoCapture and GStreamer pipelines behind
a single read() API.

Features
--------
- Auto-selects GStreamer vs plain OpenCV capture from CameraConfig.
- Frame timeout detection — if no frame arrives in frame_timeout_s,
  signals the reconnect module.
- Optional frame resize — set target_size in CameraConfig if you want
  the capture layer to downscale before inference (saves memory bandwidth
  on Jetson when the source is 4K).
- Thread-safe: designed to be called from a single capture thread.

Usage (called by reconnect.py)
-------------------------------
    cap = CameraCapture(cam_cfg, stream_cfg)
    cap.open()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config.config_loader import CameraConfig, StreamConfig
from stream.gstreamer import build_gstreamer_capture, get_stream_info

logger = logging.getLogger(__name__)

# Return type for read()
FrameResult = Tuple[bool, Optional[np.ndarray]]


class CameraCapture:
    """
    Unified capture wrapper for one camera.

    Parameters
    ----------
    cam_cfg    : CameraConfig from config.yaml
    stream_cfg : StreamConfig for timeout settings
    target_size: Optional (W, H) to resize every frame after capture.
                 None = no resize (use native resolution).
    """

    def __init__(
        self,
        cam_cfg:     CameraConfig,
        stream_cfg:  StreamConfig,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.cam_cfg     = cam_cfg
        self.stream_cfg  = stream_cfg
        self.target_size = target_size

        self._cap:            Optional[cv2.VideoCapture] = None
        self._last_frame_ts:  float = 0.0
        self._frame_count:    int   = 0
        self._is_open:        bool  = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Open the video capture.

        Returns
        -------
        bool
            True if opened successfully.
        """
        self.release()   # clean up any existing capture

        cam = self.cam_cfg
        logger.info("[%s] Opening capture — source=%s gstreamer=%s",
                    cam.id, cam.source, cam.use_gstreamer)

        if cam.use_gstreamer:
            self._cap = build_gstreamer_capture(cam)
        else:
            self._cap = cv2.VideoCapture(cam.source)
            # Tune buffer for low-latency live feed
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            logger.error("[%s] Failed to open capture.", cam.id)
            self._is_open = False
            return False

        self._is_open    = True
        self._last_frame_ts = time.monotonic()
        self._frame_count   = 0

        info = get_stream_info(self._cap)
        logger.info(
            "[%s] Capture open — %dx%d @ %.1f fps  backend=%s",
            cam.id, info["width"], info["height"], info["fps"], info["backend"],
        )
        return True

    def release(self) -> None:
        """Release the underlying VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap     = None
            self._is_open = False

    # ── Frame read ────────────────────────────────────────────────────

    def read(self) -> FrameResult:
        """
        Read the next frame.

        Returns
        -------
        (True, frame)  — success
        (False, None)  — capture not open, read error, or frame timeout
        """
        if not self._is_open or self._cap is None:
            return False, None

        # Timeout guard — stream frozen / camera offline
        elapsed = time.monotonic() - self._last_frame_ts
        if elapsed > self.stream_cfg.frame_timeout_s and self._frame_count > 0:
            logger.warning(
                "[%s] Frame timeout — no frame for %.1fs (threshold=%.1fs).",
                self.cam_cfg.id, elapsed, self.stream_cfg.frame_timeout_s,
            )
            return False, None

        ret, frame = self._cap.read()

        if not ret or frame is None:
            logger.debug("[%s] cap.read() returned no frame.", self.cam_cfg.id)
            return False, None

        self._last_frame_ts = time.monotonic()
        self._frame_count  += 1

        # Optional resize
        if self.target_size is not None:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        return True, frame

    # ── Properties ────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def resolution(self) -> Tuple[int, int]:
        """(width, height) of the capture stream."""
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def info(self) -> dict:
        if self._cap is None:
            return {}
        return {
            **get_stream_info(self._cap),
            "camera_id":   self.cam_cfg.id,
            "frame_count": self._frame_count,
            "is_open":     self._is_open,
        }