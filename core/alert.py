
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from core.temporal import TemporalEvent, AlertState

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Mirrors [alert] section in config.yaml."""
    snapshot_dir: str = "alerts/snapshots"
    pose_confirm_dir: str = "alerts/pose_confirm"
    video_clip_dir: str = "alerts/video_clips"

    save_snapshot: bool = True
    save_video_clip: bool = True

    # Video clip: frames before and after the alert trigger
    clip_pre_frames: int  = 50   # ~2s pre-event at 25fps
    clip_post_frames: int = 75   # ~3s post-event

    jpeg_quality: int = 85
    video_fps: float  = 25.0
    video_codec: str  = "MJPG"   # or 'mp4v' if ffmpeg available

    # Minimum seconds between repeat snapshots for the same track
    snapshot_cooldown_s: float = 10.0


class FrameBuffer:
    """
    Rolling ring buffer of recent frames for pre-event video clip generation.
    One buffer per pipeline (not per track — all tracks share the same stream).
    """

    def __init__(self, max_frames: int) -> None:
        self._buf: Deque[np.ndarray] = deque(maxlen=max_frames)

    def push(self, frame: np.ndarray) -> None:
        self._buf.append(frame.copy())

    def get_recent(self, n: int) -> list[np.ndarray]:
        """Return the last n frames (or fewer if buffer not full yet)."""
        frames = list(self._buf)
        return frames[-n:] if len(frames) >= n else frames

    def __len__(self) -> int:
        return len(self._buf)


class AlertHandler:
    """
    Handles alert output for man-down events.

    Usage
    -----
    handler = AlertHandler(config)
    handler.push_frame(frame)          # call every frame to maintain buffer
    handler.handle(event, frame, pose_frame)  # call when should_alert=True
    """

    def __init__(self, config: Optional[AlertConfig] = None) -> None:
        self.cfg = config or AlertConfig()
        self._frame_buffer = FrameBuffer(max_frames=self.cfg.clip_pre_frames + 10)
        self._last_snapshot_time: dict[int, float] = {}   # track_id → timestamp

        # Post-event clip collectors: track_id → list of frames still needed
        self._pending_clips: dict[int, _ClipCollector] = {}

        self._ensure_dirs()

    # ── Public API ────────────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Must be called every frame BEFORE handle().
        Maintains the pre-event rolling buffer.
        """
        self._frame_buffer.push(frame)

        # Feed pending post-event collectors
        for collector in list(self._pending_clips.values()):
            collector.add(frame)
            if collector.is_complete():
                self._write_clip(collector)
                del self._pending_clips[collector.track_id]

    def handle(
        self,
        event: TemporalEvent,
        annotated_frame: np.ndarray,
        raw_frame: np.ndarray,
    ) -> None:
        """
        Process one alert event.

        Parameters
        ----------
        event : TemporalEvent
            The triggering temporal event (should_alert must be True).
        annotated_frame : np.ndarray
            Frame with skeleton/bbox overlay drawn (for snapshot visual).
        raw_frame : np.ndarray
            Clean frame without overlay (for video clip).
        """
        if not event.should_alert:
            return

        tid = event.track_id
        now = time.time()

        # Cooldown guard — don't spam snapshots for the same track
        last = self._last_snapshot_time.get(tid, 0.0)
        if (now - last) < self.cfg.snapshot_cooldown_s:
            return

        self._last_snapshot_time[tid] = now
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"mandown_track{tid}_{timestamp_str}"

        if self.cfg.save_snapshot:
            self._save_snapshot(annotated_frame, event, base_name)

        if self.cfg.save_video_clip:
            self._start_clip_collection(tid, base_name, raw_frame.shape)

    # ── Private: snapshot ─────────────────────────────────────────────

    def _save_snapshot(
        self, frame: np.ndarray, event: TemporalEvent, base_name: str
    ) -> None:
        snap_path = Path(self.cfg.snapshot_dir) / f"{base_name}.jpg"
        meta_path = Path(self.cfg.pose_confirm_dir) / f"{base_name}.json"

        try:
            cv2.imwrite(
                str(snap_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality],
            )
            logger.info("Snapshot saved: %s", snap_path)
        except Exception as e:
            logger.error("Failed to save snapshot: %s", e)

        # Save metadata JSON alongside snapshot for audit trail
        meta = {
            "track_id":        event.track_id,
            "alert_state":     event.alert_state.name,
            "smoothed_score":  event.smoothed_score,
            "fallen_frames":   event.fallen_counter,
            "alert_duration_s":event.alert_duration_s,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "decision": {
                "state":  event.decision.state.name,
                "score":  event.decision.score,
                "reason": event.decision.reason,
                "metrics": {
                    k: (float(v) if isinstance(v, (int, float)) else str(v))
                    for k, v in event.decision.metrics.items()
                },
            },
        }
        try:
            meta_path.write_text(json.dumps(meta, indent=2))
            logger.info("Alert metadata saved: %s", meta_path)
        except Exception as e:
            logger.error("Failed to save metadata: %s", e)

    # ── Private: video clip ───────────────────────────────────────────

    def _start_clip_collection(
        self, track_id: int, base_name: str, frame_shape: Tuple[int, ...]
    ) -> None:
        pre_frames = self._frame_buffer.get_recent(self.cfg.clip_pre_frames)
        collector = _ClipCollector(
            track_id=track_id,
            base_name=base_name,
            output_dir=self.cfg.video_clip_dir,
            pre_frames=pre_frames,
            post_frames_needed=self.cfg.clip_post_frames,
            fps=self.cfg.video_fps,
            codec=self.cfg.video_codec,
            frame_shape=frame_shape,
        )
        self._pending_clips[track_id] = collector
        logger.info("Started clip collection for track %d (%d pre-frames).", track_id, len(pre_frames))

    def _write_clip(self, collector: "_ClipCollector") -> None:
        out_path = Path(collector.output_dir) / f"{collector.base_name}.avi"
        h, w = collector.frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*collector.codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, collector.fps, (w, h))
        if not writer.isOpened():
            logger.error("VideoWriter failed to open: %s", out_path)
            return
        for f in collector.all_frames():
            writer.write(f)
        writer.release()
        logger.info("Video clip saved: %s (%d frames)", out_path, len(list(collector.all_frames())))

    # ── Private: setup ────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        for d in [self.cfg.snapshot_dir, self.cfg.pose_confirm_dir, self.cfg.video_clip_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


class _ClipCollector:
    """Accumulates post-event frames until the clip is complete."""

    def __init__(
        self,
        track_id: int,
        base_name: str,
        output_dir: str,
        pre_frames: list[np.ndarray],
        post_frames_needed: int,
        fps: float,
        codec: str,
        frame_shape: Tuple[int, ...],
    ) -> None:
        self.track_id = track_id
        self.base_name = base_name
        self.output_dir = output_dir
        self._pre = pre_frames
        self._post: list[np.ndarray] = []
        self._post_needed = post_frames_needed
        self.fps = fps
        self.codec = codec
        self.frame_shape = frame_shape

    def add(self, frame: np.ndarray) -> None:
        if not self.is_complete():
            self._post.append(frame.copy())

    def is_complete(self) -> bool:
        return len(self._post) >= self._post_needed

    def all_frames(self):
        return iter(self._pre + self._post)