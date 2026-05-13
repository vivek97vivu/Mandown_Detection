"""
main.py
-------
Man-Down Detection System — Application Entry Point

Wires together:
  config_loader → Worker → ReconnectLoop(s) → display / alerts

Multi-camera: one ReconnectLoop per enabled camera, each in its own thread.
Single-camera: runs in the main thread (simpler, easier to debug on Jetson).

Usage
-----
    python main.py                          # uses config/config.yaml
    python main.py --config path/to/cfg.yaml
    python main.py --no-display             # headless (SSH / edge deployment)
    python main.py --cam cam_1              # run only one specific camera
    python main.py --log-level DEBUG        # verbose logging
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from typing import Dict, List

import cv2
import numpy as np

from config.config_loader import AppConfig, CameraConfig, get_config
from core.worker import Worker, WorkerConfig
from stream.reconnect import ReconnectLoop
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Global shutdown flag ──────────────────────────────────────────────
_shutdown_event = threading.Event()


def _signal_handler(sig, frame):
    logger.info("Signal %s received — shutting down...", sig)
    _shutdown_event.set()


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Per-camera state ──────────────────────────────────────────────────

class CameraSession:
    """Holds the Worker + ReconnectLoop for one camera."""

    def __init__(
        self,
        cam_cfg:    CameraConfig,
        app_cfg:    AppConfig,
        display:    bool,
        scale:      float,
    ) -> None:
        self.cam_cfg  = cam_cfg
        self.app_cfg  = app_cfg
        self.display  = display
        self.scale    = scale

        self.worker   = Worker(config=app_cfg.worker)
        self.loop     = ReconnectLoop(
            cam_cfg        = cam_cfg,
            stream_cfg     = app_cfg.stream,
            frame_callback = self._on_frame,
            on_reconnect   = self._on_reconnect,
            on_give_up     = self._on_give_up,
        )
        self._thread: threading.Thread | None = None
        self._win_name = f"ManDown — {cam_cfg.name or cam_cfg.id}"

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        self.worker.start()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"cam-{self.cam_cfg.id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("[%s] Camera session started.", self.cam_cfg.id)

    def stop(self) -> None:
        self.loop.stop()
        if self._thread:
            self._thread.join(timeout=10)
        self.worker.stop()
        if self.display:
            cv2.destroyWindow(self._win_name)
        logger.info("[%s] Camera session stopped.", self.cam_cfg.id)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal ──────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self.loop.run()

    def _on_frame(self, frame: np.ndarray, cam_id: str) -> None:
        """Called by ReconnectLoop for every captured frame."""
        if _shutdown_event.is_set():
            self.loop.stop()
            return

        # Run full inference pipeline
        annotated = self.worker.process_frame(frame)

        # Display
        if self.display:
            show = annotated
            if self.scale != 1.0:
                w = int(annotated.shape[1] * self.scale)
                h = int(annotated.shape[0] * self.scale)
                show = cv2.resize(annotated, (w, h))
            cv2.imshow(self._win_name, show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # q or ESC
                logger.info("[%s] Quit key pressed.", cam_id)
                _shutdown_event.set()

    def _on_reconnect(self, cam_id: str, attempt: int) -> None:
        logger.info("[%s] Reconnected (attempt #%d) — resetting FPS counter.", cam_id, attempt)
        self.worker._fps.reset()

    def _on_give_up(self, cam_id: str) -> None:
        logger.error("[%s] Giving up on reconnection.", cam_id)
        _shutdown_event.set()


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    # Load config
    try:
        app_cfg = get_config(args.config)
    except Exception as e:
        print(f"[FATAL] Config error: {e}", file=sys.stderr)
        return 1

    # Override log level if passed via CLI
    if args.log_level:
        import logging
        logging.getLogger().setLevel(args.log_level.upper())

    # Filter cameras if --cam specified
    cameras = [c for c in app_cfg.cameras if c.enabled]
    if args.cam:
        cameras = [c for c in cameras if c.id == args.cam]
        if not cameras:
            logger.error("Camera '%s' not found or not enabled in config.", args.cam)
            return 1

    display = app_cfg.stream.display and not args.no_display
    scale   = app_cfg.stream.display_scale

    logger.info(
        "Starting man-down detection — %d camera(s) | display=%s",
        len(cameras), display,
    )

    # ── Launch sessions ───────────────────────────────────────────────
    sessions: List[CameraSession] = []
    for cam_cfg in cameras:
        session = CameraSession(cam_cfg, app_cfg, display=display, scale=scale)
        session.start()
        sessions.append(session)

    logger.info("All sessions started. Press Ctrl+C or 'q' in any window to stop.")

    # ── Main thread: wait for shutdown ────────────────────────────────
    try:
        while not _shutdown_event.is_set():
            # Check if all session threads died unexpectedly
            if all(not s.is_alive() for s in sessions):
                logger.warning("All camera sessions ended — exiting.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    finally:
        logger.info("Stopping all sessions...")
        for session in sessions:
            session.stop()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")

    return 0


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Man-Down Detection — Construction & Oil/Gas Safety System"
    )
    p.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    p.add_argument(
        "--cam",
        default=None,
        metavar="CAMERA_ID",
        help="Run only the specified camera ID (e.g. cam_1). Default: all enabled.",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow — headless mode for SSH / edge deployment.",
    )
    p.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override logging level from config.",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())