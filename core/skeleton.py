from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from core.pose import KP, PoseResult


# ── Visibility thresholds ─────────────────────────────────────────────
_MIN_VIS = 0.3     # keypoint score below this → treat as invisible


# ── Public API ────────────────────────────────────────────────────────

def body_angle(pose: PoseResult) -> Optional[float]:
    """
    Angle (degrees) of the spine axis relative to vertical (0° = upright).

    Uses midpoint(shoulders) → midpoint(hips) as the spine vector.
    Returns None if key joints are not visible.

    Interpretation
    --------------
    ≈  0°   person is standing (spine vertical)
    ≈ 90°   person is lying horizontally → strong man-down indicator
    45-80°  person is crouching, bending, or falling
    """
    mid_sh = _midpoint(pose, "left_shoulder", "right_shoulder")
    mid_hip = _midpoint(pose, "left_hip", "right_hip")

    if mid_sh is None or mid_hip is None:
        return None

    dx = mid_hip[0] - mid_sh[0]
    dy = mid_hip[1] - mid_sh[1]

    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return None

    # Angle from vertical axis (dy=dominant → upright = near 0°)
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle  # always in [0, 90]


def bbox_aspect_ratio(pose: PoseResult) -> float:
    """
    Width / Height of the person bounding box.

    < 0.5  → tall/upright  (normal)
    > 0.8  → wide/flat     (lying down — man-down indicator)
    """
    x1, y1, x2, y2 = pose.bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    return w / h


def head_height_ratio(pose: PoseResult) -> Optional[float]:
    """
    Vertical position of the head (nose) relative to the bbox height.

    0.0 = head at the top of the bbox (normal standing)
    0.5 = head at the mid-point      (crouching or fallen)
    1.0 = head at the bottom          (inverted — unlikely but handled)

    Returns None if nose is not visible.
    """
    nose_score = pose.kp_score("nose")
    if nose_score < _MIN_VIS:
        return None

    nose_y = pose.keypoint("nose")[1]
    x1, y1, x2, y2 = pose.bbox
    bbox_h = max(y2 - y1, 1)
    return float((nose_y - y1) / bbox_h)


def lower_body_on_ground(pose: PoseResult) -> Optional[bool]:
    """
    Heuristic: are the hips at approximately the same height as the ankles?

    True  → hips and ankles on similar Y plane → likely lying down
    False → hips clearly above ankles          → likely standing/sitting
    None  → insufficient keypoint visibility
    """
    hip_y = _mean_y(pose, "left_hip", "right_hip")
    ankle_y = _mean_y(pose, "left_ankle", "right_ankle")

    if hip_y is None or ankle_y is None:
        return None

    # On screen, larger Y = lower in the image (feet further down than hips normally)
    vertical_diff = ankle_y - hip_y       # positive when standing (ankles below hips)
    bbox_h = max(pose.bbox[3] - pose.bbox[1], 1)

    # If hips and ankles within 15% of bbox height → effectively flat
    return (vertical_diff / bbox_h) < 0.15


def limb_symmetry(pose: PoseResult) -> Optional[float]:
    """
    Bilateral symmetry score of left/right limbs in the vertical axis.

    Returns the mean absolute Y-difference between paired joints,
    normalised by bbox height.

    Low  (< 0.1) → symmetric, person likely upright or evenly lying
    High (> 0.3) → asymmetric, one side up / one down → fallen pose

    Returns None if fewer than 2 joint pairs are visible.
    """
    pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_hip",      "right_hip"),
        ("left_knee",     "right_knee"),
        ("left_ankle",    "right_ankle"),
    ]
    bbox_h = max(pose.bbox[3] - pose.bbox[1], 1)
    diffs = []

    for l_name, r_name in pairs:
        if pose.visible(l_name) and pose.visible(r_name):
            ly = pose.keypoint(l_name)[1]
            ry = pose.keypoint(r_name)[1]
            diffs.append(abs(ly - ry) / bbox_h)

    if len(diffs) < 2:
        return None
    return float(np.mean(diffs))


def keypoint_visibility_count(pose: PoseResult, min_score: float = _MIN_VIS) -> int:
    """Count how many of the 17 keypoints are above the visibility threshold."""
    return int(np.sum(pose.kp_scores >= min_score))


def head_below_hips(pose: PoseResult) -> Optional[bool]:
    """
    True if the nose/head Y-coordinate is below (higher pixel Y) than the hip midpoint.
    Indicates an inverted or severely fallen pose.
    Returns None if either is not visible.
    """
    if not pose.visible("nose"):
        return None
    hip_y = _mean_y(pose, "left_hip", "right_hip")
    if hip_y is None:
        return None
    nose_y = pose.keypoint("nose")[1]
    return nose_y > hip_y  # screen coordinates: larger Y = lower on screen


def compute_all(pose: PoseResult) -> dict:
    """
    Convenience: compute all skeleton metrics for a PoseResult.
    Returns a dict consumed directly by decision.py.
    """
    return {
        "body_angle":           body_angle(pose),
        "bbox_aspect_ratio":    bbox_aspect_ratio(pose),
        "head_height_ratio":    head_height_ratio(pose),
        "lower_body_on_ground": lower_body_on_ground(pose),
        "limb_symmetry":        limb_symmetry(pose),
        "kp_visible_count":     keypoint_visibility_count(pose),
        "head_below_hips":      head_below_hips(pose),
        "mean_kp_score":        pose.mean_kp_score,
    }


# ── Private helpers ───────────────────────────────────────────────────

def _midpoint(
    pose: PoseResult, name_a: str, name_b: str
) -> Optional[Tuple[float, float]]:
    """Mean (x, y) of two named keypoints if both are visible."""
    if not (pose.visible(name_a) and pose.visible(name_b)):
        return None
    xa, ya = pose.keypoint(name_a)
    xb, yb = pose.keypoint(name_b)
    return ((xa + xb) / 2, (ya + yb) / 2)


def _mean_y(pose: PoseResult, name_a: str, name_b: str) -> Optional[float]:
    """Mean Y of two keypoints. Uses one if only one is visible."""
    vis_a = pose.visible(name_a)
    vis_b = pose.visible(name_b)
    if not vis_a and not vis_b:
        return None
    if vis_a and vis_b:
        return (pose.keypoint(name_a)[1] + pose.keypoint(name_b)[1]) / 2
    return pose.keypoint(name_a)[1] if vis_a else pose.keypoint(name_b)[1]