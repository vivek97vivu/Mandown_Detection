"""
drawing.py
----------
On-screen overlay renderer for the man-down detection pipeline.

Draws per-frame:
  - Bounding box with track ID
  - 17-keypoint skeleton with COCO limb connections
  - Decision state label + confidence score
  - Full-frame alert banner when a person is in ALERTING state

Visual style: High-visibility
  - Thick lines, large fonts, high-contrast colours
  - Designed for outdoor LCD monitors in direct sunlight
  - Colour-coded by person state:
      UPRIGHT   → green
      CROUCHING → amber/yellow
      FALLEN    → red  (+ flashing alert banner)
      UNCERTAIN → grey

All drawing is done with OpenCV primitives — no external deps.
The frame is modified in-place for zero-copy performance on Jetson.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.decision import DecisionResult, PersonState
from core.pose import PoseResult, KEYPOINT_NAMES
from core.temporal import AlertState, TemporalEvent

# ── COCO 17-keypoint limb pairs (index pairs to connect) ──────────────
LIMB_PAIRS: List[Tuple[int, int]] = [
    (0, 1), (0, 2),          # nose → eyes
    (1, 3), (2, 4),          # eyes → ears
    (3, 5), (4, 6),          # ears → shoulders
    (5, 6),                  # shoulder bar
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (5, 11), (6, 12),        # torso sides
    (11, 12),                # hip bar
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
]

# ── Colours (BGR) ─────────────────────────────────────────────────────
_GREEN  = (0,  210,  80)
_AMBER  = (0,  180, 240)
_RED    = (30,  30, 230)
_GREY   = (140, 140, 140)
_WHITE  = (255, 255, 255)
_BLACK  = (0,    0,   0)
_CYAN   = (220, 200,   0)

_STATE_COLOUR: Dict[PersonState, Tuple[int, int, int]] = {
    PersonState.UPRIGHT:   _GREEN,
    PersonState.CROUCHING: _AMBER,
    PersonState.FALLEN:    _RED,
    PersonState.UNCERTAIN: _GREY,
}

# Limb colour map: left-side limbs cyan, right-side amber, centre green
_LIMB_COLOURS: List[Tuple[int, int, int]] = [
    _GREY,  _GREY,           # nose-eye
    _CYAN,  _AMBER,          # eye-ear
    _CYAN,  _AMBER,          # ear-shoulder
    _GREEN,                  # shoulder bar
    _CYAN,  _CYAN,           # left arm
    _AMBER, _AMBER,          # right arm
    _CYAN,  _AMBER,          # torso sides
    _GREEN,                  # hip bar
    _CYAN,  _CYAN,           # left leg
    _AMBER, _AMBER,          # right leg
]


@dataclass
class DrawingConfig:
    """Mirrors [drawing] section of config.yaml."""
    # Keypoints
    kp_radius:       int   = 5
    kp_min_score:    float = 0.30      # below this → don't draw keypoint
    limb_thickness:  int   = 3

    # Bounding box
    bbox_thickness:  int   = 2
    show_track_id:   bool  = True
    show_score:      bool  = True
    show_state:      bool  = True

    # Fonts
    font:            int   = cv2.FONT_HERSHEY_DUPLEX
    font_scale:      float = 0.65
    font_thickness:  int   = 2
    small_font_scale:float = 0.50

    # Alert banner
    banner_height:   int   = 56        # pixels from top of frame
    banner_alpha:    float = 0.80      # opacity of banner background
    flash_hz:        float = 2.0       # alert banner flash frequency

    # HUD
    show_fps:        bool  = True
    show_frame_idx:  bool  = True
    hud_margin:      int   = 10        # pixels from bottom-left


def draw_frame(
    frame: np.ndarray,
    poses: List[PoseResult],
    decisions: List[DecisionResult],
    events: List[TemporalEvent],
    fps: float,
    config: Optional[DrawingConfig] = None,
) -> np.ndarray:
    """
    Main draw entry point. Renders all overlays onto the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame to annotate (modified in-place).
    poses : List[PoseResult]
        Keypoint results for all detected persons.
    decisions : List[DecisionResult]
        Per-person classification results (same order as poses).
    events : List[TemporalEvent]
        Temporal events from this frame (used for alert banner).
    fps : float
        Current pipeline FPS for HUD.
    config : DrawingConfig | None
        Drawing options. Defaults used if None.

    Returns
    -------
    np.ndarray
        Same frame reference (modified in-place).
    """
    cfg = config or DrawingConfig()
    alerting_ids = {e.track_id for e in events if e.should_alert}
    decision_map = {d.track_id: d for d in decisions}

    # Draw each person
    for pose in poses:
        decision = decision_map.get(pose.track_id)
        state    = decision.state if decision else PersonState.UNCERTAIN
        score    = decision.score if decision else 0.0
        colour   = _STATE_COLOUR[state]
        is_alert = pose.track_id in alerting_ids

        _draw_skeleton(frame, pose, cfg, colour)
        _draw_bbox(frame, pose, state, score, colour, is_alert, cfg)

    # Alert banner (full-width, top of frame)
    if alerting_ids:
        _draw_alert_banner(frame, alerting_ids, cfg)

    # HUD: FPS + frame stats
    _draw_hud(frame, fps, len(poses), cfg)

    return frame


# ── Per-person renderers ───────────────────────────────────────────────

def _draw_skeleton(
    frame: np.ndarray,
    pose: PoseResult,
    cfg: DrawingConfig,
    colour: Tuple[int, int, int],
) -> None:
    """Draw limb connections then keypoint circles."""
    kps    = pose.keypoints.astype(int)
    scores = pose.kp_scores

    # Limb lines — zip LIMB_PAIRS directly with _LIMB_COLOURS
    for (a, b), limb_col in zip(LIMB_PAIRS, _LIMB_COLOURS):
        if scores[a] < cfg.kp_min_score or scores[b] < cfg.kp_min_score:
            continue
        pt_a = (int(kps[a][0]), int(kps[a][1]))
        pt_b = (int(kps[b][0]), int(kps[b][1]))
        cv2.line(frame, pt_a, pt_b, limb_col, cfg.limb_thickness, cv2.LINE_AA)

    # Keypoint dots
    for idx, (kp, score) in enumerate(zip(kps, scores)):
        if score < cfg.kp_min_score:
            continue
        pt = (int(kp[0]), int(kp[1]))
        cv2.circle(frame, pt, cfg.kp_radius + 1, _BLACK,  -1, cv2.LINE_AA)
        cv2.circle(frame, pt, cfg.kp_radius,     colour,  -1, cv2.LINE_AA)


def _draw_bbox(
    frame:    np.ndarray,
    pose:     PoseResult,
    state:    PersonState,
    score:    float,
    colour:   Tuple[int, int, int],
    is_alert: bool,
    cfg:      DrawingConfig,
) -> None:
    """Draw bounding box, track ID, state label, and score."""
    x1, y1, x2, y2 = pose.bbox
    thickness = cfg.bbox_thickness + (2 if is_alert else 0)

    # Main bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)

    # Corner accents for high-visibility
    _draw_corner_accents(frame, x1, y1, x2, y2, colour, length=18, thickness=thickness + 1)

    # Label block above the bbox
    lines: List[str] = []
    if cfg.show_track_id:
        lines.append(f"ID:{pose.track_id}")
    if cfg.show_state:
        lines.append(state.name)
    if cfg.show_score:
        lines.append(f"{score:.0%}")

    label = "  ".join(lines)
    _draw_label(frame, label, x1, y1, colour, cfg)


def _draw_corner_accents(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour: Tuple[int, int, int],
    length: int = 16,
    thickness: int = 3,
) -> None:
    """Tactical corner-tick accents — improves bbox readability on busy scenes."""
    corners = [
        # (start, h_end, v_end)
        ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
        ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
        ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
        ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
    ]
    for origin, h_end, v_end in corners:
        cv2.line(frame, origin, h_end, colour, thickness, cv2.LINE_AA)
        cv2.line(frame, origin, v_end, colour, thickness, cv2.LINE_AA)


def _draw_label(
    frame:  np.ndarray,
    text:   str,
    x:      int,
    y:      int,
    colour: Tuple[int, int, int],
    cfg:    DrawingConfig,
) -> None:
    """Draw a filled pill label above the bounding box."""
    (tw, th), baseline = cv2.getTextSize(text, cfg.font, cfg.font_scale, cfg.font_thickness)
    pad_x, pad_y = 6, 4
    lx1 = x
    ly1 = max(0, y - th - pad_y * 2 - baseline)
    lx2 = x + tw + pad_x * 2
    ly2 = y

    # Filled background
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), colour, -1, cv2.LINE_AA)
    # Text
    cv2.putText(
        frame, text,
        (lx1 + pad_x, ly2 - baseline - pad_y // 2),
        cfg.font, cfg.font_scale, _BLACK, cfg.font_thickness, cv2.LINE_AA,
    )


# ── Alert banner ──────────────────────────────────────────────────────

def _draw_alert_banner(
    frame:        np.ndarray,
    alerting_ids: set,
    cfg:          DrawingConfig,
) -> None:
    """
    Full-width red banner at the top of the frame.
    Flashes at cfg.flash_hz by alternating opacity.
    """
    h, w = frame.shape[:2]
    bh   = cfg.banner_height

    # Flash: visible for half the period, hidden for half
    period = 1.0 / max(cfg.flash_hz, 0.1)
    phase  = (time.monotonic() % period) / period
    if phase > 0.5:
        # Dim flash — still visible but less intense
        alpha = 0.45
    else:
        alpha = cfg.banner_alpha

    # Overlay translucent red strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh), _RED, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Warning icon  ▐ ⚠ ▌  + message
    ids_str = ", ".join(f"#{i}" for i in sorted(alerting_ids))
    msg     = f"  \u26A0  MAN DOWN DETECTED  |  Track {ids_str}  |  ALERT ACTIVE  \u26A0  "

    (tw, th), _ = cv2.getTextSize(msg, cfg.font, 0.80, 2)
    tx = max(0, (w - tw) // 2)
    ty = (bh + th) // 2

    # Text shadow for contrast
    cv2.putText(frame, msg, (tx + 2, ty + 2), cfg.font, 0.80, _BLACK,  3, cv2.LINE_AA)
    cv2.putText(frame, msg, (tx,     ty),     cfg.font, 0.80, _WHITE,  2, cv2.LINE_AA)

    # Bottom border line
    cv2.line(frame, (0, bh), (w, bh), _RED, 2, cv2.LINE_AA)


# ── HUD ───────────────────────────────────────────────────────────────

def _draw_hud(
    frame:   np.ndarray,
    fps:     float,
    n_persons: int,
    cfg:     DrawingConfig,
) -> None:
    """Bottom-left HUD: FPS, person count."""
    if not cfg.show_fps:
        return

    h, w = frame.shape[:2]
    lines = []
    if cfg.show_fps:
        lines.append(f"FPS: {fps:.1f}")
    lines.append(f"Persons: {n_persons}")

    margin = cfg.hud_margin
    line_h = 22
    for i, line in enumerate(reversed(lines)):
        y = h - margin - i * line_h
        # Shadow
        cv2.putText(frame, line, (margin + 1, y + 1),
                    cfg.font, cfg.small_font_scale, _BLACK, 2, cv2.LINE_AA)
        # Text
        cv2.putText(frame, line, (margin, y),
                    cfg.font, cfg.small_font_scale, _WHITE, 1, cv2.LINE_AA)