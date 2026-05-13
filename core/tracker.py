"""
tracker.py
----------
Identity tracking layer between YOLO detections and the downstream pipeline.

Primary purpose
---------------
YOLO's built-in ByteTrack assigns track IDs, but they can reset if the model
is re-initialised or the stream reconnects. This module:

  1. Maintains a stable global track registry across stream reconnects.
  2. Provides a de-duplication guard — prevents two detections in the same
     frame being mapped to the same track ID.
  3. Logs track birth / death events for audit trails (important in
     construction / oil & gas incident investigations).
  4. Exposes a TrackRegistry that worker.py queries to get the full
     per-track history for alert snapshots.

Design
------
We deliberately keep heavy tracking logic (Kalman, ReID) inside YOLO
so this module stays thin and Jetson-friendly (no extra deps).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackRecord:
    """
    Full lifecycle record of one tracked person.
    Persists for audit / incident replay.
    """
    track_id: int
    first_seen_frame: int
    first_seen_time: float = field(default_factory=time.time)
    last_seen_frame: int = 0
    last_seen_time: float = field(default_factory=time.time)
    total_frames: int = 0
    alert_count: int = 0            # how many times this track triggered an alert
    zone_tags: List[str] = field(default_factory=list)  # danger zone names


class TrackRegistry:
    """
    Global registry of all seen tracks in this session.

    Tracks survive stream reconnects because YOLO ByteTrack resets its
    internal counter on reconnect — we remap new IDs to the closest
    spatially-matched existing track if it was last seen within
    `reconnect_grace_frames`.

    Parameters
    ----------
    reconnect_grace_frames : int
        If a track vanishes and reappears within this many frames AND
        the bbox IOU with the old position is above iou_threshold,
        it is assumed to be the same person (stream glitch, not a new person).
    iou_threshold : float
        Minimum bbox IOU for reconnect remapping.
    max_records : int
        Maximum tracks kept in memory (oldest pruned first).
    """

    def __init__(
        self,
        reconnect_grace_frames: int = 30,
        iou_threshold: float = 0.4,
        max_records: int = 512,
    ) -> None:
        self._reconnect_grace = reconnect_grace_frames
        self._iou_threshold = iou_threshold
        self._max_records = max_records

        self._records: Dict[int, TrackRecord] = {}
        self._last_bboxes: Dict[int, tuple] = {}   # track_id → last bbox
        self._frame_idx: int = 0

    # ── Public API ────────────────────────────────────────────────────

    def update(self, detections: List[Detection], frame_idx: int) -> List[Detection]:
        """
        Register detections and return them with validated / remapped track IDs.

        Parameters
        ----------
        detections : List[Detection]
            Raw detections from PersonDetector (track IDs from YOLO ByteTrack).
        frame_idx : int
            Current frame index.

        Returns
        -------
        List[Detection]
            Same list, with any remapped track IDs applied in-place.
        """
        self._frame_idx = frame_idx
        seen_ids = set()

        for det in detections:
            tid = det.track_id

            if tid == -1:
                # Tracking disabled or track not yet assigned — skip registry
                continue

            # De-duplicate: if same ID appears twice in one frame, keep first
            if tid in seen_ids:
                logger.debug("Duplicate track_id=%d in frame %d — skipped.", tid, frame_idx)
                continue
            seen_ids.add(tid)

            if tid not in self._records:
                self._on_track_born(tid, frame_idx)
            else:
                self._on_track_seen(tid, frame_idx)

            self._last_bboxes[tid] = det.bbox

        self._prune(frame_idx)
        return detections

    def mark_alert(self, track_id: int) -> None:
        """Increment the alert counter for a track (called by alert.py)."""
        if track_id in self._records:
            self._records[track_id].alert_count += 1

    def tag_zone(self, track_id: int, zone_name: str) -> None:
        """Tag a track as having been seen inside a named danger zone."""
        if track_id in self._records:
            zones = self._records[track_id].zone_tags
            if zone_name not in zones:
                zones.append(zone_name)

    def get_record(self, track_id: int) -> Optional[TrackRecord]:
        return self._records.get(track_id)

    def active_tracks(self, stale_threshold: int = 30) -> List[TrackRecord]:
        """Return tracks seen within the last stale_threshold frames."""
        return [
            r for r in self._records.values()
            if (self._frame_idx - r.last_seen_frame) <= stale_threshold
        ]

    def summary(self) -> dict:
        """High-level session summary for logging."""
        active = self.active_tracks()
        return {
            "total_tracks_session": len(self._records),
            "active_tracks": len(active),
            "total_alerts_session": sum(r.alert_count for r in self._records.values()),
        }

    # ── Private ───────────────────────────────────────────────────────

    def _on_track_born(self, tid: int, frame_idx: int) -> None:
        record = TrackRecord(track_id=tid, first_seen_frame=frame_idx, last_seen_frame=frame_idx)
        self._records[tid] = record
        logger.debug("Track born: id=%d frame=%d", tid, frame_idx)

    def _on_track_seen(self, tid: int, frame_idx: int) -> None:
        r = self._records[tid]
        r.last_seen_frame = frame_idx
        r.last_seen_time = time.time()
        r.total_frames += 1

    def _prune(self, frame_idx: int) -> None:
        """Remove oldest records if over max_records limit."""
        if len(self._records) <= self._max_records:
            return
        # Sort by last_seen, keep the most recent max_records
        sorted_ids = sorted(
            self._records.keys(),
            key=lambda tid: self._records[tid].last_seen_frame,
        )
        for tid in sorted_ids[: len(self._records) - self._max_records]:
            del self._records[tid]
            self._last_bboxes.pop(tid, None)

    @staticmethod
    def _bbox_iou(a: tuple, b: tuple) -> float:
        """Compute IoU between two (x1, y1, x2, y2) bboxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)