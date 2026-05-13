"""
worker.py
---------
Main inference pipeline orchestrator.

Wires together all core modules into a single per-frame processing loop:

    Frame → PersonDetector → PoseEstimator → TrackRegistry →
    DecisionEngine → TemporalFilter → AlertHandler → DrawingUtils

Designed for a single camera stream on Jetson.
For multi-camera, spawn one Worker per stream in separate processes.

Threading model
---------------
- Inference runs in the calling thread (main or stream thread).
- Alert I/O (snapshot + video write) runs in a background thread via
  a non-blocking queue so disk writes never stall the inference loop.

Jetson optimisations
--------------------
- Frame is only copied once (for the alert frame buffer).
- All intermediate structures (detections, poses) use pre-allocated
  numpy arrays where possible.
- GPU sync is implicit inside YOLO/RTMPose — no explicit cudaSync needed.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np

from core.alert import AlertConfig, AlertHandler
from core.decision import DecisionConfig, DecisionEngine
from core.detector import Detection, PersonDetector
from core.pose import PoseEstimator, PoseResult
from core.temporal import TemporalConfig, TemporalFilter, TemporalEvent
from core.tracker import TrackRegistry
from utils.drawing import DrawingConfig, draw_frame
from utils.fps import FPSCounter
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WorkerConfig:
    """Top-level config for the worker. Populated from config.yaml."""
    # Model paths
    yolo_model_path:    str = "models/yolo/yolo26s.pt"
    rtmpose_model_path: str = "models/pose/rtmpose.pth"
    rtmpose_backend:    str = "rtmlib"       # rtmlib | mmpose | onnx
    rtmpose_config:     str = ""             # mmpose .py config path (mmpose backend only)

    # Sub-configs (populated by config_loader.py)
    decision: DecisionConfig = None
    temporal: TemporalConfig = None
    alert: AlertConfig       = None
    drawing: DrawingConfig   = None

    # Misc
    device: str    = "cuda"
    log_interval_frames: int = 100    # log FPS + summary every N frames

    def __post_init__(self):
        if self.decision is None:
            self.decision = DecisionConfig()
        if self.temporal is None:
            self.temporal = TemporalConfig()
        if self.alert is None:
            self.alert = AlertConfig()
        if self.drawing is None:
            self.drawing = DrawingConfig()


class Worker:
    """
    Single-camera man-down detection worker.

    Parameters
    ----------
    config : WorkerConfig
        Full pipeline configuration.
    on_alert : Callable[[TemporalEvent, np.ndarray], None] | None
        Optional callback fired when an alert event is raised.
        Receives (event, annotated_frame). Runs in background thread.
    """

    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        on_alert: Optional[Callable[[TemporalEvent, np.ndarray], None]] = None,
    ) -> None:
        self.cfg = config or WorkerConfig()
        self._on_alert_cb = on_alert

        # Core modules
        self._detector  = PersonDetector(
            model_path=self.cfg.yolo_model_path,
            device=self.cfg.device,
        )
        self._pose      = PoseEstimator(
            model_path=self.cfg.rtmpose_model_path,
            device=self.cfg.device,
            backend=getattr(self.cfg, 'rtmpose_backend', 'rtmlib'),
            mmpose_config=getattr(self.cfg, 'rtmpose_config', None) or None,
        )
        self._tracker   = TrackRegistry()
        self._decision  = DecisionEngine(config=self.cfg.decision)
        self._temporal  = TemporalFilter(config=self.cfg.temporal)
        self._alert     = AlertHandler(config=self.cfg.alert)
        self._fps       = FPSCounter(window=30)

        # Alert I/O queue (background writer thread)
        self._alert_queue: queue.Queue = queue.Queue(maxsize=32)
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_idx = 0

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Load models and start the background I/O thread."""
        logger.info("Loading models...")
        self._detector.load()
        self._pose.load()

        self._running = True
        self._writer_thread = threading.Thread(
            target=self._io_worker, daemon=True, name="alert-io"
        )
        self._writer_thread.start()
        logger.info("Worker started. Ready to process frames.")

    def stop(self) -> None:
        """Graceful shutdown — flush the alert queue then release GPU."""
        logger.info("Stopping worker...")
        self._running = False

        # Drain the alert queue
        self._alert_queue.put(None)   # sentinel
        if self._writer_thread:
            self._writer_thread.join(timeout=10)

        self._detector.release()
        self._pose.release()

        summary = self._tracker.summary()
        logger.info("Session summary: %s", summary)
        logger.info("Worker stopped.")

    # ── Main per-frame method ─────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full man-down pipeline on one BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from the camera (any resolution).

        Returns
        -------
        np.ndarray
            Annotated BGR frame with skeleton overlays and alert banners.
            Safe to display directly or pass to a video writer.
        """
        t0 = time.perf_counter()
        self._frame_idx += 1
        idx = self._frame_idx

        # ── 1. Push raw frame to alert buffer (pre-event recording) ───
        self._alert.push_frame(frame)

        # ── 2. YOLO person detection + ByteTrack ──────────────────────
        detections: List[Detection] = self._detector.detect(frame)
        detections = self._tracker.update(detections, idx)

        # ── 3. RTMPose keypoint estimation ────────────────────────────
        poses: List[PoseResult] = self._pose.estimate(frame, detections)

        # ── 4. Per-person classification ──────────────────────────────
        from core.decision import DecisionResult
        decisions: List[DecisionResult] = [
            self._decision.classify(pose) for pose in poses
        ]

        # ── 5. Temporal smoothing + alert gating ──────────────────────
        events: List[TemporalEvent] = self._temporal.update(decisions, idx)

        # ── 6. Draw overlay on frame ──────────────────────────────────
        annotated = draw_frame(
            frame=frame.copy(),
            poses=poses,
            decisions=decisions,
            events=events,
            fps=self._fps.fps,
            config=self.cfg.drawing,
        )

        # ── 7. Handle alerts (non-blocking queue push) ────────────────
        for event in events:
            if event.should_alert:
                self._tracker.mark_alert(event.track_id)
                try:
                    self._alert_queue.put_nowait((event, annotated, frame))
                except queue.Full:
                    logger.warning("Alert queue full — dropping alert for track %d.", event.track_id)

                if self._on_alert_cb:
                    try:
                        self._on_alert_cb(event, annotated)
                    except Exception as e:
                        logger.error("on_alert callback raised: %s", e)

        # ── 8. FPS tracking ───────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._fps.tick()

        if idx % self.cfg.log_interval_frames == 0:
            logger.info(
                "Frame %d | FPS=%.1f | frame_ms=%.1f | persons=%d | alerting=%s",
                idx,
                self._fps.fps,
                elapsed_ms,
                len(detections),
                self._temporal.get_alerting_tracks(),
            )

        return annotated

    # ── Background alert I/O thread ───────────────────────────────────

    def _io_worker(self) -> None:
        """
        Runs in a daemon thread.
        Reads alert events from the queue and performs disk I/O.
        Isolated from the inference loop so disk latency never drops frames.
        """
        logger.debug("Alert I/O thread started.")
        while True:
            item = self._alert_queue.get()
            if item is None:
                break   # sentinel — shutdown
            event, annotated_frame, raw_frame = item
            try:
                self._alert.handle(event, annotated_frame, raw_frame)
            except Exception as e:
                logger.error("Alert I/O error for track %d: %s", event.track_id, e)
            finally:
                self._alert_queue.task_done()
        logger.debug("Alert I/O thread exited.")