"""
decision.py
-----------
Multi-factor man-down classifier.

Combines skeleton geometry metrics from skeleton.py into a final
UPRIGHT / FALLEN / UNCERTAIN classification per person per frame.

Design principles
-----------------
- No state — this module is purely functional. Per-frame classification
  only. Temporal smoothing lives in temporal.py.
- Configurable thresholds — all tunable via config.yaml so site engineers
  can adjust without touching code (e.g. oil & gas vs construction differ
  in crouch/PPE norms).
- Confidence score — returns a 0..1 score, not a hard bool, so temporal.py
  can apply hysteresis rather than a raw threshold.

Evidence weighting (construction + oil & gas context)
------------------------------------------------------
Workers in these environments may:
  - Crouch/kneel (normal work posture) → must not trigger false alarm
  - Wear bulky PPE that distorts bbox aspect ratio
  - Partially occlude behind equipment → fewer visible keypoints
  - Work on elevated surfaces, ladders (prone but not fallen)

Thresholds are set conservatively; temporal.py enforces dwell-time before
an alert fires, which is the primary false-positive guard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from core.pose import PoseResult
from core import skeleton

logger = logging.getLogger(__name__)


class PersonState(Enum):
    UPRIGHT   = auto()   # standing / walking — no concern
    CROUCHING = auto()   # kneeling / bending — monitor
    FALLEN    = auto()   # man-down — trigger temporal counter
    UNCERTAIN = auto()   # insufficient keypoints to decide


@dataclass
class DecisionResult:
    """Output of the classifier for one person in one frame."""
    track_id: int
    state: PersonState
    score: float          # 0.0 (definitely upright) → 1.0 (definitely fallen)
    reason: str           # human-readable explanation for logging/debugging
    metrics: dict         # raw skeleton metrics (for dashboard / logging)


@dataclass
class DecisionConfig:
    """
    All thresholds in one place — mirrors the [decision] section of config.yaml.
    Instantiated by config_loader.py and passed into DecisionEngine.
    """
    # Body angle thresholds (degrees from vertical)
    angle_upright_max: float = 30.0     # below this → definitely upright
    angle_fallen_min: float  = 60.0     # above this → strong fallen signal

    # Bbox aspect ratio thresholds (width / height)
    aspect_upright_max: float = 0.55    # tall bbox → upright
    aspect_fallen_min: float  = 0.80    # wide bbox → lying

    # Head height ratio (head near bottom of bbox → fallen)
    head_height_fallen_min: float = 0.45

    # Minimum visible keypoints to make a confident decision
    min_keypoints: int = 6

    # Minimum YOLO detection confidence to run pose analysis
    min_detection_conf: float = 0.35

    # Weights for weighted-average score computation
    weight_angle:      float = 0.40
    weight_aspect:     float = 0.25
    weight_head:       float = 0.15
    weight_lower_body: float = 0.10
    weight_symmetry:   float = 0.10

    # Final score threshold: above this → FALLEN (pre-temporal)
    fallen_score_threshold: float = 0.55

    # Score threshold below which state is CROUCHING (not upright)
    crouching_score_threshold: float = 0.30


class DecisionEngine:
    """
    Classifies a single person's pose each frame.

    Usage
    -----
    engine = DecisionEngine(config)
    result = engine.classify(pose_result)
    """

    def __init__(self, config: Optional[DecisionConfig] = None) -> None:
        self.cfg = config or DecisionConfig()

    def classify(self, pose: PoseResult) -> DecisionResult:
        """
        Classify one PoseResult into a PersonState with confidence score.

        Parameters
        ----------
        pose : PoseResult
            Output of PoseEstimator.estimate() for a single person.

        Returns
        -------
        DecisionResult
        """
        metrics = skeleton.compute_all(pose)

        # ── Guard: insufficient data ───────────────────────────────────
        if pose.confidence < self.cfg.min_detection_conf:
            return self._uncertain(pose.track_id, metrics, "low detection confidence")

        if metrics["kp_visible_count"] < self.cfg.min_keypoints:
            return self._uncertain(
                pose.track_id, metrics,
                f"only {metrics['kp_visible_count']} keypoints visible"
            )

        # ── Compute per-evidence scores ────────────────────────────────
        angle_score   = self._angle_score(metrics["body_angle"])
        aspect_score  = self._aspect_score(metrics["bbox_aspect_ratio"])
        head_score    = self._head_score(metrics["head_height_ratio"])
        lower_score   = self._lower_body_score(metrics["lower_body_on_ground"])
        sym_score     = self._symmetry_score(metrics["limb_symmetry"])

        # ── Weighted combination ───────────────────────────────────────
        cfg = self.cfg
        total_weight = (
            cfg.weight_angle + cfg.weight_aspect + cfg.weight_head
            + cfg.weight_lower_body + cfg.weight_symmetry
        )

        # Accumulate only evidence that is available (not None)
        weighted_sum  = 0.0
        active_weight = 0.0

        for score, weight in [
            (angle_score,  cfg.weight_angle),
            (aspect_score, cfg.weight_aspect),
            (head_score,   cfg.weight_head),
            (lower_score,  cfg.weight_lower_body),
            (sym_score,    cfg.weight_symmetry),
        ]:
            if score is not None:
                weighted_sum  += score * weight
                active_weight += weight

        if active_weight < 0.3 * total_weight:
            # Too little evidence even with available keypoints
            return self._uncertain(pose.track_id, metrics, "insufficient evidence coverage")

        final_score = weighted_sum / active_weight

        # ── Hard-override: head below hips is a very strong signal ─────
        if metrics["head_below_hips"]:
            final_score = max(final_score, 0.85)
            reason = "head below hips (override)"
        else:
            reason = self._build_reason(angle_score, aspect_score, head_score, final_score)

        # ── State assignment ───────────────────────────────────────────
        if final_score >= cfg.fallen_score_threshold:
            state = PersonState.FALLEN
        elif final_score >= cfg.crouching_score_threshold:
            state = PersonState.CROUCHING
        else:
            state = PersonState.UPRIGHT

        logger.debug(
            "track=%d  state=%-10s  score=%.2f  reason=%s",
            pose.track_id, state.name, final_score, reason
        )

        return DecisionResult(
            track_id=pose.track_id,
            state=state,
            score=round(final_score, 3),
            reason=reason,
            metrics=metrics,
        )

    # ── Private: per-evidence score functions ─────────────────────────
    # Each returns a float in [0, 1] or None if data unavailable.
    # 0.0 = strong upright signal, 1.0 = strong fallen signal.

    def _angle_score(self, angle: Optional[float]) -> Optional[float]:
        if angle is None:
            return None
        cfg = self.cfg
        if angle <= cfg.angle_upright_max:
            return 0.0
        if angle >= cfg.angle_fallen_min:
            return 1.0
        # Linear interpolation in between
        return (angle - cfg.angle_upright_max) / (cfg.angle_fallen_min - cfg.angle_upright_max)

    def _aspect_score(self, ratio: float) -> float:
        cfg = self.cfg
        if ratio <= cfg.aspect_upright_max:
            return 0.0
        if ratio >= cfg.aspect_fallen_min:
            return 1.0
        return (ratio - cfg.aspect_upright_max) / (cfg.aspect_fallen_min - cfg.aspect_upright_max)

    def _head_score(self, head_ratio: Optional[float]) -> Optional[float]:
        if head_ratio is None:
            return None
        # Head high in frame (small ratio) → upright
        # Head at middle/low of bbox → fallen
        threshold = self.cfg.head_height_fallen_min
        if head_ratio <= 0.2:
            return 0.0
        if head_ratio >= threshold:
            return 1.0
        return (head_ratio - 0.2) / (threshold - 0.2)

    def _lower_body_score(self, on_ground: Optional[bool]) -> Optional[float]:
        if on_ground is None:
            return None
        return 1.0 if on_ground else 0.0

    def _symmetry_score(self, sym: Optional[float]) -> Optional[float]:
        """High asymmetry → likely one side fallen → higher score."""
        if sym is None:
            return None
        # Clip at 0.4 — beyond that it's clearly asymmetric
        return float(min(sym / 0.4, 1.0))

    # ── Private: helpers ──────────────────────────────────────────────

    def _uncertain(self, track_id: int, metrics: dict, reason: str) -> DecisionResult:
        return DecisionResult(
            track_id=track_id,
            state=PersonState.UNCERTAIN,
            score=0.0,
            reason=reason,
            metrics=metrics,
        )

    def _build_reason(
        self,
        angle_score: Optional[float],
        aspect_score: float,
        head_score: Optional[float],
        final: float,
    ) -> str:
        parts = []
        if angle_score is not None:
            parts.append(f"angle={angle_score:.2f}")
        parts.append(f"aspect={aspect_score:.2f}")
        if head_score is not None:
            parts.append(f"head={head_score:.2f}")
        parts.append(f"→ final={final:.2f}")
        return " | ".join(parts)