"""
fps.py
------
Lightweight FPS counter for the inference loop.

Two modes:
  - Sliding window (default) — exact mean over last N frame durations.
    Best for dashboards — stable and truthful.
  - EMA (exponential moving average) — smoothed FPS, responds faster to
    sudden drops. Better for Jetson where thermal throttling causes bursts.

Usage
-----
    fps = FPSCounter(window=30)
    while True:
        frame = cam.read()
        process(frame)
        fps.tick()
        print(fps.fps)        # current FPS
        print(fps.frame_time_ms)  # last frame latency in ms
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional


class FPSCounter:
    """
    Thread-safe FPS counter with sliding-window average.

    Parameters
    ----------
    window : int
        Number of recent frames to average over (default 30).
    ema_alpha : float
        EMA smoothing factor — only used by fps_ema property.
        Higher = more reactive, lower = smoother.
    """

    def __init__(self, window: int = 30, ema_alpha: float = 0.1) -> None:
        self._window: int = window
        self._alpha: float = ema_alpha
        self._timestamps: Deque[float] = deque(maxlen=window + 1)
        self._ema_fps: float = 0.0
        self._last_tick: float = 0.0
        self._frame_count: int = 0

    def tick(self) -> None:
        """Call once per processed frame, immediately after processing."""
        now = time.perf_counter()
        self._timestamps.append(now)
        self._frame_count += 1

        if self._last_tick > 0:
            dt = now - self._last_tick
            instant = 1.0 / dt if dt > 0 else 0.0
            if self._ema_fps == 0.0:
                self._ema_fps = instant
            else:
                self._ema_fps = self._alpha * instant + (1 - self._alpha) * self._ema_fps

        self._last_tick = now

    @property
    def fps(self) -> float:
        """
        Sliding-window FPS average.
        Returns 0.0 if fewer than 2 frames have been ticked.
        """
        ts = list(self._timestamps)
        if len(ts) < 2:
            return 0.0
        elapsed = ts[-1] - ts[0]
        if elapsed <= 0:
            return 0.0
        return (len(ts) - 1) / elapsed

    @property
    def fps_ema(self) -> float:
        """EMA-smoothed FPS. More stable during thermal throttle on Jetson."""
        return round(self._ema_fps, 2)

    @property
    def frame_time_ms(self) -> float:
        """Last frame-to-frame interval in milliseconds."""
        ts = list(self._timestamps)
        if len(ts) < 2:
            return 0.0
        return (ts[-1] - ts[-2]) * 1000.0

    @property
    def total_frames(self) -> int:
        """Total frames ticked since creation."""
        return self._frame_count

    def reset(self) -> None:
        """Reset all counters (e.g. after a stream reconnect)."""
        self._timestamps.clear()
        self._ema_fps = 0.0
        self._last_tick = 0.0
        self._frame_count = 0

    def report(self) -> dict:
        """Return a snapshot dict for logging."""
        return {
            "fps":          round(self.fps, 2),
            "fps_ema":      self.fps_ema,
            "frame_time_ms":round(self.frame_time_ms, 2),
            "total_frames": self.total_frames,
        }


class SectionTimer:
    """
    Context-manager timer for profiling individual pipeline sections.

    Usage
    -----
        timer = SectionTimer()
        with timer("yolo"):
            detections = detector.detect(frame)
        with timer("pose"):
            poses = estimator.estimate(frame, detections)
        print(timer.report())
    """

    def __init__(self) -> None:
        self._times: dict[str, list[float]] = {}
        self._current_section: Optional[str] = None
        self._section_start: float = 0.0

    def __call__(self, section: str) -> "SectionTimer":
        self._current_section = section
        return self

    def __enter__(self) -> "SectionTimer":
        self._section_start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed = (time.perf_counter() - self._section_start) * 1000  # ms
        name = self._current_section or "unknown"
        self._times.setdefault(name, []).append(elapsed)

    def mean_ms(self, section: str) -> float:
        """Mean execution time in ms for a named section."""
        times = self._times.get(section, [])
        return sum(times) / len(times) if times else 0.0

    def report(self, last_n: int = 30) -> dict:
        """
        Mean ms per section over the last N calls.
        Useful for finding the bottleneck on Jetson.
        """
        return {
            name: round(sum(vals[-last_n:]) / min(len(vals), last_n), 2)
            for name, vals in self._times.items()
        }

    def reset(self) -> None:
        self._times.clear()