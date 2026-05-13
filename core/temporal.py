"""
temporal.py
-----------
Temporal state machine per tracked person.

Prevents false alarms by requiring a person to remain in FALLEN state
for a configurable dwell-time before an alert fires.
Also handles:
  - Hysteresis: person must return to UPRIGHT for a sustained period
    before their alert state resets (avoids alert flapping).
  - Score smoothing: exponential moving average over per-frame scores
    to reduce jitter from noisy keypoint estimates.
  - Track death: remove stale tracks that haven't been seen for N frames.

Construction / oil & gas context
---------------------------------
Typical settings:
  - dwell_frames=75 (~3s at 25fps) to avoid false alerts from stumbles
  - recovery_frames=50 to ensure person is genuinely back on feet
  - score_ema_alpha=0.35 for moderate smoothing on Jetson (25-30fps)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

from core.decision import DecisionResult, PersonState

logger = logging.getLogger(__name__)


class AlertState(Enum):
    NORMAL    = auto()  # person is upright / no concern
    SUSPECTED = auto()  # fallen frames accumulating (dwell counter ticking)
    ALERTING  = auto()  # dwell time exceeded — alert active
    RECOVERING= auto()  # alert was active, person now upright (recovery dwell)


@dataclass
class TemporalConfig:
    """Mirrors [temporal] section in config.yaml."""
    dwell_frames: int      = 75     # frames FALLEN before alert fires (~3s @25fps)
    recovery_frames: int   = 50     # frames UPRIGHT before alert resets (~2s)
    score_ema_alpha: float = 0.35   # EMA smoothing factor (higher = less smooth)
    max_unseen_frames: int = 60     # frames before a track is dropped from memory
    crouching_as_fallen: bool = False  # set True for high-risk sites


@dataclass
class TrackState:
    """Internal per-person state machine node."""
    track_id: int
    alert_state: AlertState = AlertState.NORMAL
    fallen_counter: int = 0         # consecutive FALLEN frames
    recovery_counter: int = 0       # consecutive UPRIGHT frames (post-alert)
    smoothed_score: float = 0.0     # EMA of decision score
    last_seen_frame: int = 0        # frame index of last detection
    alert_start_time: Optional[float] = None  # wall-clock time alert began
    total_alert_seconds: float = 0.0           # cumulative alert duration


class TemporalFilter:
    """
    Maintains per-track state machines and converts per-frame DecisionResults
    into alert events.

    Usage
    -----
    tf = TemporalFilter(config)
    for frame_idx, results in stream:
        events = tf.update(results, frame_idx)
        for event in events:
            if event.should_alert:
                alert_system.fire(event)
    """

    def __init__(self, config: Optional[TemporalConfig] = None) -> None:
        self.cfg = config or TemporalConfig()
        self._tracks: Dict[int, TrackState] = {}

    # ── Public API ────────────────────────────────────────────────────

    def update(
        self, results: list[DecisionResult], frame_idx: int
    ) -> list["TemporalEvent"]:
        """
        Process one frame's worth of DecisionResults.

        Parameters
        ----------
        results : list[DecisionResult]
            All persons classified in this frame.
        frame_idx : int
            Monotonically increasing frame counter.

        Returns
        -------
        list[TemporalEvent]
            One event per tracked person whose state changed this frame.
            Empty if no state changes occurred.
        """
        events: list[TemporalEvent] = []
        seen_ids = set()

        for result in results:
            tid = result.track_id
            seen_ids.add(tid)
            track = self._get_or_create(tid, frame_idx)
            event = self._step(track, result, frame_idx)
            if event is not None:
                events.append(event)

        # Age unseen tracks
        self._prune_unseen(seen_ids, frame_idx)
        return events

    def get_alerting_tracks(self) -> list[int]:
        """Return track IDs currently in ALERTING state."""
        return [tid for tid, t in self._tracks.items() if t.alert_state == AlertState.ALERTING]

    def reset_track(self, track_id: int) -> None:
        """Manually reset a track (e.g. operator acknowledged the alert)."""
        if track_id in self._tracks:
            t = self._tracks[track_id]
            t.alert_state = AlertState.NORMAL
            t.fallen_counter = 0
            t.recovery_counter = 0
            t.alert_start_time = None
            logger.info("Track %d manually reset by operator.", track_id)

    # ── Private: state machine step ───────────────────────────────────

    def _step(
        self, track: TrackState, result: DecisionResult, frame_idx: int
    ) -> Optional["TemporalEvent"]:
        cfg = self.cfg
        track.last_seen_frame = frame_idx

        # Score smoothing (EMA)
        alpha = cfg.score_ema_alpha
        track.smoothed_score = alpha * result.score + (1 - alpha) * track.smoothed_score

        is_fallen = result.state == PersonState.FALLEN or (
            cfg.crouching_as_fallen and result.state == PersonState.CROUCHING
        )
        is_upright = result.state == PersonState.UPRIGHT

        prev_state = track.alert_state
        event: Optional[TemporalEvent] = None

        if track.alert_state == AlertState.NORMAL:
            if is_fallen:
                track.fallen_counter += 1
                track.alert_state = AlertState.SUSPECTED
                event = self._make_event(track, result, changed=True)
            else:
                track.fallen_counter = max(0, track.fallen_counter - 1)

        elif track.alert_state == AlertState.SUSPECTED:
            if is_fallen:
                track.fallen_counter += 1
                if track.fallen_counter >= cfg.dwell_frames:
                    track.alert_state = AlertState.ALERTING
                    track.alert_start_time = time.time()
                    logger.warning(
                        "MAN DOWN ALERT — track_id=%d fallen for %d frames",
                        track.track_id, track.fallen_counter
                    )
                    event = self._make_event(track, result, changed=True)
                else:
                    event = self._make_event(track, result, changed=False)
            else:
                # Interrupted — partial reset with decay
                track.fallen_counter = max(0, track.fallen_counter - 3)
                if track.fallen_counter == 0:
                    track.alert_state = AlertState.NORMAL
                    event = self._make_event(track, result, changed=True)

        elif track.alert_state == AlertState.ALERTING:
            # Stay alerting until recovery dwell satisfied
            now = time.time()
            if track.alert_start_time:
                track.total_alert_seconds = now - track.alert_start_time
            event = self._make_event(track, result, changed=False)

            if is_upright:
                track.recovery_counter += 1
                if track.recovery_counter >= cfg.recovery_frames:
                    track.alert_state = AlertState.RECOVERING
                    logger.info("Track %d leaving alert — recovery dwell satisfied.", track.track_id)
            else:
                track.recovery_counter = 0

        elif track.alert_state == AlertState.RECOVERING:
            if is_upright:
                track.recovery_counter += 1
                if track.recovery_counter >= cfg.recovery_frames * 2:
                    track.alert_state = AlertState.NORMAL
                    track.fallen_counter = 0
                    track.recovery_counter = 0
                    logger.info("Track %d fully recovered — alert state cleared.", track.track_id)
                    event = self._make_event(track, result, changed=True)
            else:
                # Person fell again during recovery — go straight to ALERTING
                track.alert_state = AlertState.ALERTING
                track.alert_start_time = time.time()
                track.recovery_counter = 0
                logger.warning("Track %d re-fallen during recovery — re-alerting.", track.track_id)
                event = self._make_event(track, result, changed=True)

        return event

    # ── Private: helpers ──────────────────────────────────────────────

    def _get_or_create(self, track_id: int, frame_idx: int) -> TrackState:
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackState(track_id=track_id, last_seen_frame=frame_idx)
        return self._tracks[track_id]

    def _prune_unseen(self, seen_ids: set, frame_idx: int) -> None:
        stale = [
            tid for tid, t in self._tracks.items()
            if tid not in seen_ids
            and (frame_idx - t.last_seen_frame) > self.cfg.max_unseen_frames
        ]
        for tid in stale:
            logger.debug("Dropping stale track %d.", tid)
            del self._tracks[tid]

    def _make_event(
        self, track: TrackState, result: DecisionResult, changed: bool
    ) -> "TemporalEvent":
        return TemporalEvent(
            track_id=track.track_id,
            alert_state=track.alert_state,
            smoothed_score=round(track.smoothed_score, 3),
            fallen_counter=track.fallen_counter,
            state_changed=changed,
            should_alert=track.alert_state == AlertState.ALERTING,
            alert_duration_s=track.total_alert_seconds,
            decision=result,
        )


@dataclass
class TemporalEvent:
    """Emitted by TemporalFilter.update() for each state transition."""
    track_id: int
    alert_state: AlertState
    smoothed_score: float
    fallen_counter: int
    state_changed: bool
    should_alert: bool             # True → alert system should fire
    alert_duration_s: float        # seconds since alert started (0 if not alerting)
    decision: DecisionResult       # raw per-frame decision for context