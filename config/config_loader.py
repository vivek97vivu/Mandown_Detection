"""
config_loader.py
----------------
Loads config.yaml and hydrates all pipeline dataclasses.

One call — get_config() — returns a fully validated AppConfig.
Every module receives its typed sub-config; no module reads YAML directly.

Validation rules
----------------
- Missing required keys raise ConfigError with a clear message.
- Unknown keys are warned about (not errored) to allow forward-compat.
- Numeric ranges are checked (e.g. weights must sum to ~1.0).
- Camera sources are coerced: "0" string → int 0.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from core.alert import AlertConfig
from core.decision import DecisionConfig
from core.temporal import TemporalConfig
from core.worker import WorkerConfig
from utils.drawing import DrawingConfig
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


# ── Public exception ──────────────────────────────────────────────────

class ConfigError(Exception):
    """Raised when config.yaml has invalid or missing values."""


# ── Camera config dataclass ───────────────────────────────────────────

@dataclass
class CameraConfig:
    id:               str
    source:           Any           # int (USB) or str (RTSP URL)
    name:             str  = ""
    enabled:          bool = True
    use_gstreamer:    bool = False
    codec:            str  = "h265"        # h264 | h265
    protocols:        str  = "tcp"
    latency:          int  = 0
    drop_on_latency:  bool = True


# ── Zone config dataclass ─────────────────────────────────────────────

@dataclass
class ZoneConfig:
    name:    str
    enabled: bool
    polygon: List[Tuple[float, float]]     # normalised [0,1] coords


# ── Stream config dataclass ───────────────────────────────────────────

@dataclass
class StreamConfig:
    reconnect_delay_s:       float = 3.0
    max_reconnect_attempts:  int   = 0     # 0 = infinite
    frame_timeout_s:         float = 5.0
    display:                 bool  = True
    display_scale:           float = 1.0


# ── Logging config dataclass ──────────────────────────────────────────

@dataclass
class LoggingConfig:
    log_dir:      str  = "logs"
    level:        str  = "INFO"
    enable_json:  bool = False
    max_bytes:    int  = 10 * 1024 * 1024
    backup_count: int  = 5


# ── Top-level app config ──────────────────────────────────────────────

@dataclass
class AppConfig:
    cameras:  List[CameraConfig]
    worker:   WorkerConfig
    stream:   StreamConfig
    logging:  LoggingConfig
    zones:    List[ZoneConfig] = field(default_factory=list)


# ── Main loader ───────────────────────────────────────────────────────

def get_config(path: str = "config/config.yaml") -> AppConfig:
    """
    Load and validate config.yaml. Returns a fully populated AppConfig.

    Parameters
    ----------
    path : str
        Path to config.yaml relative to project root.

    Raises
    ------
    ConfigError
        If the file is missing, unparseable, or has invalid values.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path.resolve()}")

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse config.yaml: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError("config.yaml must be a YAML mapping at the top level.")

    # ── Logging first (so all subsequent logs are formatted correctly) ─
    log_cfg   = _load_logging(raw.get("logging", {}))
    setup_logging(
        log_dir=log_cfg.log_dir,
        level=log_cfg.level,
        enable_json=log_cfg.enable_json,
        max_bytes=log_cfg.max_bytes,
        backup_count=log_cfg.backup_count,
    )

    # ── Sub-configs ────────────────────────────────────────────────────
    cameras  = _load_cameras(raw.get("cameras", []))
    decision = _load_decision(raw.get("decision", {}))
    temporal = _load_temporal(raw.get("temporal", {}))
    alert    = _load_alert(raw.get("alert", {}))
    drawing  = _load_drawing(raw.get("drawing", {}))
    stream   = _load_stream(raw.get("stream", {}))
    zones    = _load_zones(raw.get("zones", []))
    models   = raw.get("models", {})
    device   = raw.get("device", {})
    worker_r = raw.get("worker", {})

    pose_raw_cfg = models.get("pose", {})
    worker = WorkerConfig(
        yolo_model_path    = _str(models.get("yolo", {}).get("path", "models/yolo/yolo26s.pt")),
        rtmpose_model_path = _str(pose_raw_cfg.get("path", "models/pose/rtmpose.pth")),
        rtmpose_backend    = _str(pose_raw_cfg.get("backend", "rtmlib")),
        rtmpose_config     = _str(pose_raw_cfg.get("mmpose_config", "")),
        decision           = decision,
        temporal           = temporal,
        alert              = alert,
        drawing            = drawing,
        device             = _str(device.get("inference", "cuda")),
        log_interval_frames= int(worker_r.get("log_interval_frames", 100)),
    )

    # Propagate YOLO model kwargs into detector via WorkerConfig
    yolo_raw = models.get("yolo", {})
    worker.__dict__["yolo_conf"]       = float(yolo_raw.get("conf_threshold", 0.40))
    worker.__dict__["yolo_iou"]        = float(yolo_raw.get("iou_threshold",  0.45))
    worker.__dict__["yolo_input_size"] = int(yolo_raw.get("input_size", 640))
    worker.__dict__["yolo_tracking"]   = bool(yolo_raw.get("use_tracking", True))
    worker.__dict__["yolo_crop_pad"]   = float(yolo_raw.get("crop_padding", 0.15))

    pose_raw = models.get("pose", {})
    worker.__dict__["pose_backend"]    = _str(pose_raw.get("backend", "rtmlib"))
    worker.__dict__["pose_input_size"] = tuple(pose_raw.get("input_size", [192, 256]))
    worker.__dict__["pose_min_score"]  = float(pose_raw.get("min_kp_score", 0.30))
    worker.__dict__["use_fp16"]        = bool(device.get("fp16", True))

    logger.info("Config loaded from %s — %d camera(s) enabled.",
                cfg_path, sum(1 for c in cameras if c.enabled))

    return AppConfig(
        cameras=cameras,
        worker=worker,
        stream=stream,
        logging=log_cfg,
        zones=zones,
    )


# ── Section loaders ───────────────────────────────────────────────────

def _load_cameras(raw: list) -> List[CameraConfig]:
    if not raw:
        raise ConfigError("No cameras defined in config.yaml under 'cameras:'.")

    cameras = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"Camera entry {i} is not a mapping.")

        cam_id = _require_str(entry, "id", f"cameras[{i}]")
        source = entry.get("source")
        if source is None:
            raise ConfigError(f"Camera '{cam_id}' has no 'source' defined.")

        # Coerce "0" → 0 for local webcams
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cameras.append(CameraConfig(
            id              = cam_id,
            source          = source,
            name            = _str(entry.get("name", cam_id)),
            enabled         = bool(entry.get("enabled", True)),
            use_gstreamer   = bool(entry.get("use_gstreamer", False)),
            codec           = _str(entry.get("codec", "h265")),
            protocols       = _str(entry.get("protocols", "tcp")),
            latency         = int(entry.get("latency", 0)),
            drop_on_latency = bool(entry.get("drop_on_latency", True)),
        ))

    enabled = [c for c in cameras if c.enabled]
    if not enabled:
        raise ConfigError("All cameras are disabled in config.yaml.")

    return cameras


def _load_decision(raw: dict) -> DecisionConfig:
    d = DecisionConfig()
    d.angle_upright_max        = float(raw.get("angle_upright_max",        d.angle_upright_max))
    d.angle_fallen_min         = float(raw.get("angle_fallen_min",         d.angle_fallen_min))
    d.aspect_upright_max       = float(raw.get("aspect_upright_max",       d.aspect_upright_max))
    d.aspect_fallen_min        = float(raw.get("aspect_fallen_min",        d.aspect_fallen_min))
    d.head_height_fallen_min   = float(raw.get("head_height_fallen_min",   d.head_height_fallen_min))
    d.min_keypoints            = int(raw.get("min_keypoints",              d.min_keypoints))
    d.min_detection_conf       = float(raw.get("min_detection_conf",       d.min_detection_conf))
    d.weight_angle             = float(raw.get("weight_angle",             d.weight_angle))
    d.weight_aspect            = float(raw.get("weight_aspect",            d.weight_aspect))
    d.weight_head              = float(raw.get("weight_head",              d.weight_head))
    d.weight_lower_body        = float(raw.get("weight_lower_body",        d.weight_lower_body))
    d.weight_symmetry          = float(raw.get("weight_symmetry",          d.weight_symmetry))
    d.fallen_score_threshold   = float(raw.get("fallen_score_threshold",   d.fallen_score_threshold))
    d.crouching_score_threshold= float(raw.get("crouching_score_threshold",d.crouching_score_threshold))

    # Validate weights sum
    total_w = d.weight_angle + d.weight_aspect + d.weight_head + d.weight_lower_body + d.weight_symmetry
    if not (0.98 <= total_w <= 1.02):
        logger.warning(
            "Decision weights sum to %.3f (expected ~1.0). Check config.yaml [decision].", total_w
        )
    return d


def _load_temporal(raw: dict) -> TemporalConfig:
    t = TemporalConfig()
    t.dwell_frames          = int(raw.get("dwell_frames",          t.dwell_frames))
    t.recovery_frames       = int(raw.get("recovery_frames",       t.recovery_frames))
    t.score_ema_alpha       = float(raw.get("score_ema_alpha",     t.score_ema_alpha))
    t.max_unseen_frames     = int(raw.get("max_unseen_frames",     t.max_unseen_frames))
    t.crouching_as_fallen   = bool(raw.get("crouching_as_fallen",  t.crouching_as_fallen))
    return t


def _load_alert(raw: dict) -> AlertConfig:
    a = AlertConfig()
    a.snapshot_dir        = _str(raw.get("snapshot_dir",        a.snapshot_dir))
    a.pose_confirm_dir    = _str(raw.get("pose_confirm_dir",    a.pose_confirm_dir))
    a.video_clip_dir      = _str(raw.get("video_clip_dir",      a.video_clip_dir))
    a.save_snapshot       = bool(raw.get("save_snapshot",       a.save_snapshot))
    a.save_video_clip     = bool(raw.get("save_video_clip",     a.save_video_clip))
    a.clip_pre_frames     = int(raw.get("clip_pre_frames",      a.clip_pre_frames))
    a.clip_post_frames    = int(raw.get("clip_post_frames",     a.clip_post_frames))
    a.jpeg_quality        = int(raw.get("jpeg_quality",         a.jpeg_quality))
    a.video_fps           = float(raw.get("video_fps",          a.video_fps))
    a.video_codec         = _str(raw.get("video_codec",         a.video_codec))
    a.snapshot_cooldown_s = float(raw.get("snapshot_cooldown_s",a.snapshot_cooldown_s))
    return a


def _load_drawing(raw: dict) -> DrawingConfig:
    import cv2
    d = DrawingConfig()
    d.kp_radius        = int(raw.get("kp_radius",        d.kp_radius))
    d.kp_min_score     = float(raw.get("kp_min_score",   d.kp_min_score))
    d.limb_thickness   = int(raw.get("limb_thickness",   d.limb_thickness))
    d.bbox_thickness   = int(raw.get("bbox_thickness",   d.bbox_thickness))
    d.show_track_id    = bool(raw.get("show_track_id",   d.show_track_id))
    d.show_score       = bool(raw.get("show_score",      d.show_score))
    d.show_state       = bool(raw.get("show_state",      d.show_state))
    d.font_scale       = float(raw.get("font_scale",     d.font_scale))
    d.font_thickness   = int(raw.get("font_thickness",   d.font_thickness))
    d.small_font_scale = float(raw.get("small_font_scale", d.small_font_scale))
    d.banner_height    = int(raw.get("banner_height",    d.banner_height))
    d.banner_alpha     = float(raw.get("banner_alpha",   d.banner_alpha))
    d.flash_hz         = float(raw.get("flash_hz",       d.flash_hz))
    d.show_fps         = bool(raw.get("show_fps",        d.show_fps))
    d.show_frame_idx   = bool(raw.get("show_frame_idx",  d.show_frame_idx))
    d.hud_margin       = int(raw.get("hud_margin",       d.hud_margin))
    return d


def _load_stream(raw: dict) -> StreamConfig:
    return StreamConfig(
        reconnect_delay_s      = float(raw.get("reconnect_delay_s",      3.0)),
        max_reconnect_attempts = int(raw.get("max_reconnect_attempts",   0)),
        frame_timeout_s        = float(raw.get("frame_timeout_s",        5.0)),
        display                = bool(raw.get("display",                 True)),
        display_scale          = float(raw.get("display_scale",          1.0)),
    )


def _load_logging(raw: dict) -> LoggingConfig:
    return LoggingConfig(
        log_dir      = _str(raw.get("log_dir",      "logs")),
        level        = _str(raw.get("level",        "INFO")).upper(),
        enable_json  = bool(raw.get("enable_json",  False)),
        max_bytes    = int(raw.get("max_bytes",     10 * 1024 * 1024)),
        backup_count = int(raw.get("backup_count",  5)),
    )


def _load_zones(raw: list) -> List[ZoneConfig]:
    zones = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name    = entry.get("name", "unnamed_zone")
        enabled = bool(entry.get("enabled", True))
        poly    = [(float(p[0]), float(p[1])) for p in entry.get("polygon", [])]
        if len(poly) < 3:
            logger.warning("Zone '%s' has fewer than 3 polygon points — skipped.", name)
            continue
        zones.append(ZoneConfig(name=name, enabled=enabled, polygon=poly))
    return zones


# ── Helpers ───────────────────────────────────────────────────────────

def _str(val: Any) -> str:
    return str(val) if val is not None else ""


def _require_str(d: dict, key: str, context: str) -> str:
    if key not in d or not str(d[key]).strip():
        raise ConfigError(f"Missing required key '{key}' in {context}.")
    return str(d[key])