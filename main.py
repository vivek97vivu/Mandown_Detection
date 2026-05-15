"""
main.py
-------
Man-Down Detection System — Application Entry Point

Usage
-----
    python main.py                        # all enabled cameras
    python main.py --cam cam_1            # single camera
    python main.py --no-display           # headless
    python main.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from typing import List, Optional

import cv2
import numpy as np

from config.config_loader import AppConfig, CameraConfig, get_config
from core.worker import Worker
from stream.reconnect import ReconnectLoop
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Global shutdown ───────────────────────────────────────────────────
_shutdown_event = threading.Event()

def _signal_handler(sig, frame):
    logger.info("Signal %s — shutting down.", sig)
    _shutdown_event.set()

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ── Per-camera latest-frame store ─────────────────────────────────────
# Each camera writes its latest annotated frame here.
# Main thread reads and displays — only ever shows the LATEST frame,
# never queues up. This eliminates the queue-full drop problem entirely.
_latest_frames: dict[str, Optional[np.ndarray]] = {}
_frame_lock = threading.Lock()


class CameraSession:
    def __init__(
        self,
        cam_cfg:  CameraConfig,
        app_cfg:  AppConfig,
        display:  bool,
        scale:    float,
    ) -> None:
        self.cam_cfg  = cam_cfg
        self.app_cfg  = app_cfg
        self.display  = display
        self.scale    = scale
        self.win_name = f"ManDown | {cam_cfg.name or cam_cfg.id}"

        self.worker = Worker(config=app_cfg.worker)
        self.loop   = ReconnectLoop(
            cam_cfg        = cam_cfg,
            stream_cfg     = app_cfg.stream,
            frame_callback = self._on_frame,
            on_reconnect   = self._on_reconnect,
            on_give_up     = self._on_give_up,
        )
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        self.worker.start()
        # Register this camera in the shared latest-frame store
        with _frame_lock:
            _latest_frames[self.win_name] = None
        self._thread = threading.Thread(
            target=self.loop.run,
            name=f"cam-{self.cam_cfg.id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("[%s] Session started.", self.cam_cfg.id)

    def stop(self) -> None:
        self.loop.stop()
        if self._thread:
            self._thread.join(timeout=10)
        self.worker.stop()
        logger.info("[%s] Session stopped.", self.cam_cfg.id)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Frame callback (runs in camera thread) ────────────────────────

    def _on_frame(self, frame: np.ndarray, cam_id: str) -> None:
        if _shutdown_event.is_set():
            self.loop.stop()
            return

        # Run inference
        annotated = self.worker.process_frame(frame)

        if not self.display:
            return

        # Resize if needed
        if self.scale != 1.0:
            w = int(annotated.shape[1] * self.scale)
            h = int(annotated.shape[0] * self.scale)
            annotated = cv2.resize(annotated, (w, h))

        # Overwrite latest frame — main thread will pick it up on next tick
        with _frame_lock:
            _latest_frames[self.win_name] = annotated

    def _on_reconnect(self, cam_id: str, attempt: int) -> None:
        logger.info("[%s] Reconnected (#%d) — resetting FPS.", cam_id, attempt)
        self.worker._fps.reset()

    def _on_give_up(self, cam_id: str) -> None:
        logger.error("[%s] Giving up reconnection.", cam_id)
        _shutdown_event.set()


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    try:
        app_cfg = get_config(args.config)
    except Exception as e:
        print(f"[FATAL] Config error: {e}", file=sys.stderr)
        return 1

    if args.log_level:
        import logging
        logging.getLogger().setLevel(args.log_level.upper())

    cameras = [c for c in app_cfg.cameras if c.enabled]
    if args.cam:
        cameras = [c for c in cameras if c.id == args.cam]
        if not cameras:
            logger.error("Camera '%s' not found or not enabled.", args.cam)
            return 1

    display = app_cfg.stream.display and not args.no_display
    scale   = app_cfg.stream.display_scale

    logger.info("Starting — %d camera(s) | display=%s", len(cameras), display)

    # ── Create windows BEFORE starting threads ────────────────────────
    # This is critical — namedWindow must be called from the main thread
    # before any imshow calls, otherwise windows may not appear on Linux.
    if display:
        for cam_cfg in cameras:
            win = f"ManDown | {cam_cfg.name or cam_cfg.id}"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 960, 540)
            # Show a black placeholder so window appears immediately
            placeholder = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                f"Connecting to {cam_cfg.name or cam_cfg.id}...",
                (30, 270),
                cv2.FONT_HERSHEY_DUPLEX, 1.0,
                (100, 200, 100), 2, cv2.LINE_AA,
            )
            cv2.imshow(win, placeholder)
        cv2.waitKey(1)   # flush all window creates

    # ── Launch sessions ───────────────────────────────────────────────
    sessions: List[CameraSession] = []
    for cam_cfg in cameras:
        s = CameraSession(cam_cfg, app_cfg, display=display, scale=scale)
        s.start()
        sessions.append(s)

    logger.info("All sessions started. Press 'q' or Ctrl+C to stop.")

    # ── Main display loop ─────────────────────────────────────────────
    try:
        while not _shutdown_event.is_set():
            # Check all threads still alive
            if all(not s.is_alive() for s in sessions):
                logger.warning("All camera threads ended.")
                break

            if display:
                # Snapshot latest frames under lock, then display outside lock
                with _frame_lock:
                    snapshot = {k: v for k, v in _latest_frames.items()}

                for win_name, frame in snapshot.items():
                    if frame is not None:
                        cv2.imshow(win_name, frame)

                # waitKey drives the GUI event loop — MUST be called every iteration
                key = cv2.waitKey(30) & 0xFF   # 30ms = ~33fps display refresh
                if key == ord("q") or key == 27:
                    logger.info("Quit key — shutting down.")
                    _shutdown_event.set()
                    break
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Ctrl+C received.")
    finally:
        logger.info("Stopping sessions...")
        _shutdown_event.set()
        for s in sessions:
            s.stop()
        if display:
            cv2.destroyAllWindows()
        logger.info("Done.")

    return 0


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Man-Down Detection")
    p.add_argument("--config",     default="config/config.yaml")
    p.add_argument("--cam",        default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--log-level",  default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())