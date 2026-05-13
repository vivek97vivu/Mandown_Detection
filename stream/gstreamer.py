"""
gstreamer.py
------------
Builds and opens GStreamer pipelines for RTSP streams on Jetson.

Handles:
  - H265 (HEVC) — nvh265dec  (your current camera)
  - H264        — nvh264dec
  - Auto-selects hardware decoder based on config codec field
  - Returns a cv2.VideoCapture opened on the GStreamer pipeline string

All pipeline parameters come from CameraConfig — nothing hardcoded.

Usage (internal — called by capture.py)
----------------------------------------
    cap = build_gstreamer_capture(cam_cfg)
    ret, frame = cap.read()
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


# ── Pipeline templates ────────────────────────────────────────────────
# {url}       → RTSP URL
# {protocols} → tcp | udp
# {latency}   → ms (0 for live)
# {drop}      → true | false

_H265_PIPELINE = (
    "rtspsrc location={url} protocols={protocols} "
    "latency={latency} drop-on-latency={drop} ! "
    "rtph265depay ! h265parse ! nvh265dec ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true sync=false max-buffers=1"
)

_H264_PIPELINE = (
    "rtspsrc location={url} protocols={protocols} "
    "latency={latency} drop-on-latency={drop} ! "
    "rtph264depay ! h264parse ! nvh264dec ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true sync=false max-buffers=1"
)

# Software fallback (no Jetson hardware decoder — useful for testing)
_SOFT_H265_PIPELINE = (
    "rtspsrc location={url} protocols={protocols} "
    "latency={latency} drop-on-latency={drop} ! "
    "rtph265depay ! h265parse ! avdec_h265 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true sync=false max-buffers=1"
)

_SOFT_H264_PIPELINE = (
    "rtspsrc location={url} protocols={protocols} "
    "latency={latency} drop-on-latency={drop} ! "
    "rtph264depay ! h264parse ! avdec_h264 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true sync=false max-buffers=1"
)

_CODEC_MAP = {
    ("h265", True):  _H265_PIPELINE,
    ("h264", True):  _H264_PIPELINE,
    ("h265", False): _SOFT_H265_PIPELINE,
    ("h264", False): _SOFT_H264_PIPELINE,
}


def build_pipeline_string(
    url:              str,
    codec:            str  = "h265",
    protocols:        str  = "tcp",
    latency:          int  = 0,
    drop_on_latency:  bool = True,
    use_hw_decoder:   bool = True,
) -> str:
    """
    Build the GStreamer pipeline string from parameters.

    Parameters
    ----------
    url             : RTSP stream URL
    codec           : 'h264' or 'h265'
    protocols       : 'tcp' or 'udp'
    latency         : rtspsrc latency in ms
    drop_on_latency : drop frames when latency exceeded
    use_hw_decoder  : use nvh265dec/nvh264dec (Jetson) vs avdec (CPU)

    Returns
    -------
    str
        GStreamer pipeline string ready for cv2.VideoCapture.
    """
    codec_key = codec.lower().strip()
    if codec_key not in ("h264", "h265"):
        logger.warning("Unknown codec '%s', defaulting to h265.", codec)
        codec_key = "h265"

    template = _CODEC_MAP.get((codec_key, use_hw_decoder), _H265_PIPELINE)
    pipeline  = template.format(
        url=url,
        protocols=protocols,
        latency=latency,
        drop=str(drop_on_latency).lower(),
    )
    logger.debug("GStreamer pipeline: %s", pipeline)
    return pipeline


def build_gstreamer_capture(cam_cfg, use_hw_decoder: bool = True) -> cv2.VideoCapture:
    """
    Open a cv2.VideoCapture using a GStreamer pipeline built from CameraConfig.

    Parameters
    ----------
    cam_cfg : CameraConfig
        Camera configuration from config.yaml.
    use_hw_decoder : bool
        Use Jetson hardware decoder. Set False for CPU fallback.

    Returns
    -------
    cv2.VideoCapture
        Opened capture. Caller must check .isOpened().
    """
    pipeline = build_pipeline_string(
        url             = str(cam_cfg.source),
        codec           = cam_cfg.codec,
        protocols       = cam_cfg.protocols,
        latency         = cam_cfg.latency,
        drop_on_latency = cam_cfg.drop_on_latency,
        use_hw_decoder  = use_hw_decoder,
    )

    logger.info(
        "[%s] Opening GStreamer capture — codec=%s hw=%s",
        cam_cfg.id, cam_cfg.codec, use_hw_decoder,
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        if use_hw_decoder:
            logger.warning(
                "[%s] HW decoder failed. Retrying with software decoder...", cam_cfg.id
            )
            return build_gstreamer_capture(cam_cfg, use_hw_decoder=False)
        else:
            logger.error(
                "[%s] GStreamer capture failed to open (both HW and SW decoders tried).",
                cam_cfg.id,
            )

    return cap


def get_stream_info(cap: cv2.VideoCapture) -> dict:
    """Return basic stream info from an opened VideoCapture."""
    return {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "backend": cap.getBackendName(),
    }