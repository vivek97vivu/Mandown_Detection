

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


# ── Decoder capability detection ─────────────────────────────────────

@lru_cache(maxsize=None)
def _gst_plugin_available(plugin_name: str) -> bool:
    """Return True if a GStreamer plugin/element is installed."""
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", plugin_name],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _detect_best_decoder(codec: str) -> str:
    """
    Detect the best available GStreamer decoder element for the given codec.

    Returns one of:
      'jetson_hw'  — nvh265dec / nvh264dec (Jetson Tegra)
      'nvidia_gpu' — nvdec (x86 NVIDIA GPU via NVDEC)
      'vaapi'      — vaapih265dec / vaapih264dec (Intel/AMD VA-API)
      'software'   — avdec_h265 / avdec_h264 (CPU libav)
    """
    codec = codec.lower()

    # 1. Jetson hardware decoder (Tegra-specific plugin)
    jetson_elem = f"nvh{codec[1:]}dec" if codec in ("h264", "h265") else "nvh265dec"
    # nvh265dec is Jetson-only — on x86 NVIDIA it's called nvdec
    if _gst_plugin_available(jetson_elem):
        # Extra check: nvh265dec exists on x86 RTX but behaves differently
        # Jetson plugin lives in the 'nvv4l2' package; distinguish by inspect output
        try:
            result = subprocess.run(
                ["gst-inspect-1.0", jetson_elem],
                capture_output=True, timeout=5, text=True,
            )
            if "nvv4l2" in result.stdout or "Tegra" in result.stdout:
                logger.info("Decoder selected: Jetson HW (%s)", jetson_elem)
                return "jetson_hw"
        except Exception:
            pass

    # 2. NVIDIA GPU NVDEC (x86 with NVIDIA dGPU)
    if _gst_plugin_available("nvdec"):
        logger.info("Decoder selected: NVIDIA GPU NVDEC (nvdec)")
        return "nvidia_gpu"

    # 3. VA-API (Intel/AMD iGPU, common on x86 Linux)
    vaapi_elem = f"vaapi{codec}dec"
    if _gst_plugin_available(vaapi_elem):
        logger.info("Decoder selected: VA-API (%s)", vaapi_elem)
        return "vaapi"

    # 4. Software fallback — always available if gstreamer-plugins-bad/ugly installed
    logger.info("Decoder selected: software libav (avdec_%s)", codec)
    return "software"


# ── Pipeline templates ────────────────────────────────────────────────
# {url}       → RTSP URL
# {protocols} → tcp | udp
# {latency}   → ms (0 for live)
# {drop}      → true | false

_TEMPLATES: dict[str, dict[str, str]] = {
    # Jetson hardware decoders (nvv4l2 package)
    "jetson_hw": {
        "h265": (
            "rtspsrc location={url} protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph265depay ! h265parse ! nvh265dec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
        "h264": (
            "rtspsrc location={url} protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph264depay ! h264parse ! nvh264dec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
    },
    # NVIDIA NVDEC on x86 GPU (gst-plugins-bad nvdec element)
    "nvidia_gpu": {
        "h265": (
            "rtspsrc location={url} protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph265depay ! h265parse ! nvdec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
        "h264": (
            "rtspsrc location={url} protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph264depay ! h264parse ! nvdec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
    },
    # VA-API (Intel/AMD iGPU)
    "vaapi": {
        "h265": (
            "rtspsrc location={url} protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph265depay ! h265parse ! vaapih265dec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
        "h264": (
            "rtspsrc location={url} protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph264depay ! h264parse ! vaapih264dec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
    },
    # Software decoder — universal fallback
    "software": {
        "h265": (
            "rtspsrc location={url} protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph265depay ! h265parse ! avdec_h265 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
        "h264": (
            "rtspsrc location={url} protocols={protocols} "
            "latency={latency} drop-on-latency={drop} ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ),
    },
}

# Decoder fallback order — if preferred decoder fails, try the next
_DECODER_FALLBACK_ORDER = ["jetson_hw", "nvidia_gpu", "vaapi", "software"]


# ── Public API ────────────────────────────────────────────────────────

def build_pipeline_string(
    url:             str,
    codec:           str  = "h265",
    protocols:       str  = "tcp",
    latency:         int  = 0,
    drop_on_latency: bool = True,
    decoder:         Optional[str] = None,   # None = auto-detect
) -> str:
    """
    Build the GStreamer pipeline string.

    Parameters
    ----------
    url             : RTSP stream URL
    codec           : 'h264' or 'h265'
    protocols       : 'tcp' or 'udp'
    latency         : rtspsrc latency in ms
    drop_on_latency : drop frames when latency exceeded
    decoder         : force a specific decoder tier, or None for auto-detect

    Returns
    -------
    str
        GStreamer pipeline string ready for cv2.VideoCapture.
    """
    codec_key = codec.lower().strip()
    if codec_key not in ("h264", "h265"):
        logger.warning("Unknown codec '%s', defaulting to h265.", codec)
        codec_key = "h265"

    selected = decoder or _detect_best_decoder(codec_key)
    template = _TEMPLATES.get(selected, {}).get(codec_key)
    if template is None:
        # Should not happen, but fall back to software
        logger.warning("No template for decoder=%s codec=%s, using software.", selected, codec_key)
        template = _TEMPLATES["software"][codec_key]

    pipeline = template.format(
        url=url,
        protocols=protocols,
        latency=latency,
        drop=str(drop_on_latency).lower(),
    )
    logger.debug("[pipeline] %s", pipeline)
    return pipeline


def build_gstreamer_capture(cam_cfg) -> cv2.VideoCapture:
    """
    Open a cv2.VideoCapture using a GStreamer pipeline.
    Tries each decoder tier in order until one succeeds.

    Parameters
    ----------
    cam_cfg : CameraConfig

    Returns
    -------
    cv2.VideoCapture — caller must check .isOpened()
    """
    codec_key = cam_cfg.codec.lower().strip()
    cam_id    = cam_cfg.id

    # Start from the best available decoder, fall through on failure
    best = _detect_best_decoder(codec_key)
    order = _DECODER_FALLBACK_ORDER[_DECODER_FALLBACK_ORDER.index(best):]

    for decoder_tier in order:
        pipeline = build_pipeline_string(
            url=str(cam_cfg.source),
            codec=codec_key,
            protocols=cam_cfg.protocols,
            latency=cam_cfg.latency,
            drop_on_latency=cam_cfg.drop_on_latency,
            decoder=decoder_tier,
        )

        logger.info(
            "[%s] Trying GStreamer capture — decoder=%s codec=%s",
            cam_id, decoder_tier, codec_key,
        )

        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            logger.warning(
                "[%s] decoder=%s failed to open — trying next tier.",
                cam_id, decoder_tier,
            )
            cap.release()
            continue

        # Verify frames actually flow (isOpened() can return True on broken pipelines)
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning(
                "[%s] decoder=%s opened but first frame read failed — trying next tier.",
                cam_id, decoder_tier,
            )
            cap.release()
            continue

        logger.info(
            "[%s] GStreamer capture OK — decoder=%s  %dx%d",
            cam_id, decoder_tier, frame.shape[1], frame.shape[0],
        )
        # NOTE: We consumed one frame in the test read. The caller's first cap.read()
        # will get the *second* frame. This is acceptable — the capture.py open()
        # method already handles the consumed frame (sets frame_count=1).
        return cap

    logger.error(
        "[%s] All GStreamer decoder tiers failed for codec=%s. "
        "Install gstreamer1.0-libav (avdec_h265) as a fallback: "
        "  sudo apt install gstreamer1.0-libav",
        cam_id, codec_key,
    )
    # Return a dead capture — caller checks isOpened()
    return cv2.VideoCapture()


def get_stream_info(cap: cv2.VideoCapture) -> dict:
    """Return basic stream info from an opened VideoCapture."""
    return {
        "width":   int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":     cap.get(cv2.CAP_PROP_FPS),
        "backend": cap.getBackendName(),
    }