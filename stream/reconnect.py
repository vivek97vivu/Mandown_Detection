

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np

from config.config_loader import CameraConfig, StreamConfig
from stream.capture import CameraCapture

logger = logging.getLogger(__name__)

# Signature: (frame: np.ndarray, camera_id: str) -> None
FrameCallback     = Callable[[np.ndarray, str], None]
ReconnectCallback = Callable[[str, int], None]      # (camera_id, attempt_number)


class ReconnectLoop:
    """
    Auto-reconnecting capture loop for one camera.

    Parameters
    ----------
    cam_cfg           : CameraConfig
    stream_cfg        : StreamConfig (reconnect_delay_s, max_reconnect_attempts, ...)
    frame_callback    : Called with (frame, camera_id) for every good frame.
    on_reconnect      : Optional callback fired each time a reconnect succeeds.
    on_give_up        : Optional callback fired when max attempts is exceeded.
    target_size       : Optional (W, H) resize passed to CameraCapture.
    """

    _MAX_BACKOFF_S = 60.0   # cap on reconnect delay regardless of config

    def __init__(
        self,
        cam_cfg:        CameraConfig,
        stream_cfg:     StreamConfig,
        frame_callback: FrameCallback,
        on_reconnect:   Optional[ReconnectCallback] = None,
        on_give_up:     Optional[Callable[[str], None]] = None,
        target_size=None,
    ) -> None:
        self.cam_cfg        = cam_cfg
        self.stream_cfg     = stream_cfg
        self.frame_callback = frame_callback
        self.on_reconnect   = on_reconnect
        self.on_give_up     = on_give_up
        self.target_size    = target_size

        self._capture = CameraCapture(cam_cfg, stream_cfg, target_size)
        self._running = False

        self._total_frames:    int = 0
        self._connect_attempts:int = 0
        self._successful_reads:int = 0

    # ── Public API ────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the capture loop. Blocks until stop() is called or
        max_reconnect_attempts is exceeded.
        """
        self._running = True
        cam_id = self.cam_cfg.id
        max_attempts = self.stream_cfg.max_reconnect_attempts   # 0 = infinite

        logger.info("[%s] Starting reconnect loop (max_attempts=%s).",
                    cam_id, "∞" if max_attempts == 0 else max_attempts)

        attempt = 0

        while self._running:
            # ── Open / reconnect ──────────────────────────────────
            attempt += 1
            self._connect_attempts += 1

            if max_attempts > 0 and attempt > max_attempts:
                logger.error(
                    "[%s] Max reconnect attempts (%d) exceeded. Giving up.",
                    cam_id, max_attempts,
                )
                if self.on_give_up:
                    self.on_give_up(cam_id)
                break

            logger.info("[%s] Connect attempt %d...", cam_id, attempt)
            opened = self._capture.open()

            if not opened:
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "[%s] Open failed — retrying in %.1fs.", cam_id, delay
                )
                self._sleep(delay)
                continue

            # Reconnect succeeded
            attempt = 0   # reset counter on successful open
            if self.on_reconnect:
                try:
                    self.on_reconnect(cam_id, self._connect_attempts)
                except Exception as e:
                    logger.error("[%s] on_reconnect callback error: %s", cam_id, e)

            logger.info("[%s] Stream opened successfully.", cam_id)

            # ── Frame loop ────────────────────────────────────────
            consecutive_failures = 0

            while self._running:
                ret, frame = self._capture.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        logger.warning(
                            "[%s] %d consecutive read failures — triggering reconnect.",
                            cam_id, consecutive_failures,
                        )
                        break   # break inner loop → reconnect
                    continue

                consecutive_failures = 0
                self._total_frames += 1
                self._successful_reads += 1

                # ── Deliver frame to callback ──────────────────
                try:
                    self.frame_callback(frame, cam_id)
                except Exception as e:
                    logger.error("[%s] frame_callback raised: %s", cam_id, e)

            # Inner loop exited — release and wait before reconnecting
            self._capture.release()
            if self._running:
                delay = self._backoff_delay(1)   # short initial reconnect delay
                logger.info("[%s] Stream lost — reconnecting in %.1fs.", cam_id, delay)
                self._sleep(delay)

        self._capture.release()
        logger.info(
            "[%s] Reconnect loop exited — total_frames=%d connect_attempts=%d.",
            cam_id, self._total_frames, self._connect_attempts,
        )

    def stop(self) -> None:
        """Signal the loop to stop after the current frame."""
        logger.info("[%s] Stop requested.", self.cam_cfg.id)
        self._running = False

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "camera_id":        self.cam_cfg.id,
            "total_frames":     self._total_frames,
            "connect_attempts": self._connect_attempts,
            "successful_reads": self._successful_reads,
            "is_running":       self._running,
        }

    # ── Private ───────────────────────────────────────────────────────

    def _backoff_delay(self, attempt: int) -> float:
        """
        Exponential backoff: base_delay * 2^(attempt-1), capped at max.
        attempt=1 → base_delay, attempt=2 → 2x, attempt=3 → 4x ...
        """
        base  = self.stream_cfg.reconnect_delay_s
        delay = base * (2 ** (attempt - 1))
        return min(delay, self._MAX_BACKOFF_S)

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep — checks _running every 0.5s."""
        deadline = time.monotonic() + seconds
        while self._running and time.monotonic() < deadline:
            time.sleep(0.5)