"""
geometry.py
-----------
Pure geometric utility functions used across the pipeline.

Covers:
  - Bounding box operations (IoU, area, intersection, scale)
  - Point / polygon operations (point-in-polygon for danger zones)
  - Angle and vector math
  - Frame coordinate normalisation helpers

All functions operate on plain Python types or numpy arrays.
No OpenCV dependency in this file — geometry is framework-agnostic.

Construction / oil & gas context
---------------------------------
Point-in-polygon is used by tracker.py and worker.py to tag whether
a person is inside a defined danger zone (e.g. "confined space entry",
"rotating equipment perimeter", "gas leak exclusion zone").
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Type aliases
Point  = Tuple[float, float]          # (x, y)
Bbox   = Tuple[int, int, int, int]    # (x1, y1, x2, y2)
Poly   = List[Tuple[float, float]]    # list of (x, y) vertices


# ── Bounding box ──────────────────────────────────────────────────────

def bbox_area(bbox: Bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float((x2 - x1) * (y2 - y1)))


def bbox_iou(a: Bbox, b: Bbox) -> float:
    """
    Intersection over Union of two bboxes.
    Returns 0.0 if there is no overlap.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center(bbox: Bbox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_scale(bbox: Bbox, scale: float) -> Bbox:
    """
    Scale a bbox around its centre by `scale` factor.
    Useful for expanding a detection region for RTMPose cropping.
    """
    cx, cy = bbox_center(bbox)
    x1, y1, x2, y2 = bbox
    hw = (x2 - x1) / 2 * scale
    hh = (y2 - y1) / 2 * scale
    return (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))


def bbox_clamp(bbox: Bbox, frame_w: int, frame_h: int) -> Bbox:
    """Clamp a bbox so all coordinates lie within the frame."""
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1), max(0, y1),
        min(frame_w, x2), min(frame_h, y2),
    )


def bbox_aspect_ratio(bbox: Bbox) -> float:
    """Width / height. > 1 = landscape (lying down), < 1 = portrait (standing)."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    return w / h


def bbox_from_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_score: float = 0.3,
    padding: float = 0.10,
) -> Optional[Bbox]:
    """
    Derive a tight bounding box from visible keypoints.
    Useful as a fallback when YOLO detection is poor but pose is confident.

    Parameters
    ----------
    keypoints : np.ndarray  shape (N, 2)
    scores    : np.ndarray  shape (N,)
    min_score : float       keypoints below this are ignored
    padding   : float       fractional padding added on all sides
    """
    visible = keypoints[scores >= min_score]
    if len(visible) < 2:
        return None

    x1, y1 = visible.min(axis=0)
    x2, y2 = visible.max(axis=0)
    w, h   = x2 - x1, y2 - y1

    return (
        int(x1 - w * padding),
        int(y1 - h * padding),
        int(x2 + w * padding),
        int(y2 + h * padding),
    )


# ── Point / polygon ───────────────────────────────────────────────────

def point_in_polygon(point: Point, polygon: Poly) -> bool:
    """
    Ray-casting algorithm for point-in-polygon test.
    Works for any simple (non-self-intersecting) polygon.

    Used for danger zone membership:
      zone_poly = [(x1,y1), (x2,y2), ...]   # pixel coordinates
      inside = point_in_polygon(bbox_center(det.bbox), zone_poly)

    Parameters
    ----------
    point   : (x, y)
    polygon : list of (x, y) vertices (closed polygon, last → first implied)
    """
    x, y    = point
    n       = len(polygon)
    inside  = False
    j       = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i

    return inside


def polygon_area(polygon: Poly) -> float:
    """Shoelace formula — area of a simple polygon."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    j = n - 1
    for i in range(n):
        area += (polygon[j][0] + polygon[i][0]) * (polygon[j][1] - polygon[i][1])
        j = i
    return abs(area) / 2.0


def bbox_in_zone(bbox: Bbox, zone_poly: Poly, threshold: float = 0.5) -> bool:
    """
    Returns True if the bbox centre is inside the zone polygon.
    For a stricter check, also verify that the bottom-centre (feet) is inside.
    """
    cx, cy   = bbox_center(bbox)
    feet_x   = cx
    feet_y   = float(bbox[3])         # bottom edge Y
    centre_in = point_in_polygon((cx, cy),     zone_poly)
    feet_in   = point_in_polygon((feet_x, feet_y), zone_poly)
    return centre_in or feet_in


# ── Vector / angle math ───────────────────────────────────────────────

def angle_between_points(
    a: Point, b: Point, c: Point
) -> float:
    """
    Angle (degrees) at vertex B in the triangle A-B-C.
    Useful for joint angle calculation (e.g. knee bend angle).
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot   = ba[0] * bc[0] + ba[1] * bc[1]
    mag_a = math.hypot(*ba)
    mag_c = math.hypot(*bc)

    if mag_a < 1e-6 or mag_c < 1e-6:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_angle))


def vector_angle_from_vertical(p1: Point, p2: Point) -> float:
    """
    Angle (degrees) of vector p1→p2 from the vertical axis.
    0° = perfectly vertical (upright spine), 90° = horizontal (lying).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


# ── Coordinate normalisation ──────────────────────────────────────────

def normalize_point(point: Point, frame_w: int, frame_h: int) -> Point:
    """Convert pixel coords to [0, 1] normalised coords."""
    return (point[0] / frame_w, point[1] / frame_h)


def denormalize_point(point: Point, frame_w: int, frame_h: int) -> Point:
    """Convert [0, 1] normalised coords back to pixel coords."""
    return (point[0] * frame_w, point[1] * frame_h)


def scale_polygon_to_frame(
    poly_normalised: Poly, frame_w: int, frame_h: int
) -> Poly:
    """
    Convert a polygon defined in normalised [0,1] coords to pixel coords.
    Allows zone definitions in config.yaml to be resolution-independent.

    Example in config.yaml:
      zones:
        - name: "confined_space_entry"
          polygon: [[0.1, 0.2], [0.4, 0.2], [0.4, 0.9], [0.1, 0.9]]
    """
    return [
        (int(x * frame_w), int(y * frame_h))
        for x, y in poly_normalised
    ]


# ── NMS helper (used in detector.py fallback path) ────────────────────

def nms(
    bboxes: List[Bbox],
    scores: List[float],
    iou_threshold: float = 0.45,
) -> List[int]:
    """
    Non-maximum suppression. Returns indices of kept bboxes.
    Used as a fallback when Ultralytics NMS is bypassed.
    """
    if not bboxes:
        return []

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept: List[int] = []

    while order:
        best = order.pop(0)
        kept.append(best)
        order = [
            i for i in order
            if bbox_iou(bboxes[best], bboxes[i]) < iou_threshold
        ]

    return kept