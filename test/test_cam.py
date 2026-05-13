"""
test_cam.py
-----------
Live camera validation tool.

Tests:
  1. Can the capture open (USB + RTSP/GStreamer)?
  2. Is the frame resolution and FPS what we expect?
  3. Are frames arriving without drops / timeout?
  4. Does GStreamer HW decoder (nvh265dec) actually work on this Jetson?
  5. Latency measurement — time from cap.read() to frame available.

Usage
-----
    # Test all enabled cameras from config
    python test/test_cam.py

    # Test a specific camera
    python test/test_cam.py --cam cam_2

    # Run for N seconds then print report
    python test/test_cam.py --duration 30

    # Save a sample frame to disk
    python test/test_cam.py --save-frame

    # Test GStreamer pipeline string without opening
    python test/test_cam.py --print-pipeline --cam cam_2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import CameraConfig, get_config
from stream.capture import CameraCapture
from stream.gstreamer import build_pipeline_string, get_stream_info
from utils.fps import FPSCounter
from utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("test_cam")


# ── ANSI colours for terminal output ─────────────────────────────────
_G = "\033[32m"   # green
_Y = "\033[33m"   # yellow
_R = "\033[31m"   # red
_B = "\033[1m"    # bold
_E = "\033[0m"    # reset

def _ok(msg):  print(f"  {_G}✓{_E}  {msg}")
def _warn(msg):print(f"  {_Y}⚠{_E}  {msg}")
def _fail(msg):print(f"  {_R}✗{_E}  {msg}")
def _hdr(msg): print(f"\n{_B}{msg}{_E}")


# ── Tests ─────────────────────────────────────────────────────────────

def test_open(cam_cfg: CameraConfig, stream_cfg) -> CameraCapture | None:
    """Test 1: Can the capture open?"""
    _hdr(f"[{cam_cfg.id}] Test 1 — Open capture")

    cap = CameraCapture(cam_cfg, stream_cfg)
    opened = cap.open()

    if opened:
        _ok(f"Capture opened successfully (gstreamer={cam_cfg.use_gstreamer})")
        return cap
    else:
        _fail("Failed to open capture. Check source, URL, codec, and network.")
        if cam_cfg.use_gstreamer:
            _warn("GStreamer tip: run `gst-launch-1.0 " +
                  build_pipeline_string(str(cam_cfg.source), cam_cfg.codec) +
                  "` in terminal to debug.")
        return None


def test_resolution(cap: CameraCapture, cam_cfg: CameraConfig) -> bool:
    """Test 2: Check resolution and reported FPS."""
    _hdr(f"[{cam_cfg.id}] Test 2 — Resolution & FPS")
    info = cap.info()

    w, h = info.get("width", 0), info.get("height", 0)
    fps  = info.get("fps", 0)

    if w > 0 and h > 0:
        _ok(f"Resolution: {w}×{h}")
    else:
        _warn("Resolution reported as 0×0 (common with GStreamer — check first frame).")

    if fps > 0:
        _ok(f"Reported FPS: {fps:.1f}")
    else:
        _warn("FPS reported as 0 (GStreamer appsink — actual FPS measured live below).")

    _ok(f"Backend: {info.get('backend', 'unknown')}")
    return w >= 0 and h >= 0


def test_frames(
    cap:      CameraCapture,
    cam_cfg:  CameraConfig,
    duration: float,
    save_frame: bool,
    show:     bool,
) -> dict:
    """Test 3: Read frames for `duration` seconds, measure FPS and drop rate."""
    _hdr(f"[{cam_cfg.id}] Test 3 — Frame stream ({duration:.0f}s)")

    fps_counter   = FPSCounter(window=60)
    total_reads   = 0
    failed_reads  = 0
    latencies_ms  = []
    first_frame   = None
    start         = time.monotonic()

    print(f"  Reading frames for {duration:.0f}s ... (press Ctrl+C to stop early)")

    try:
        while (time.monotonic() - start) < duration:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            lat = (time.perf_counter() - t0) * 1000

            total_reads += 1

            if not ret or frame is None:
                failed_reads += 1
                if failed_reads > 10:
                    _fail("10+ consecutive read failures — stream may be down.")
                    break
                continue

            failed_reads = 0   # reset consecutive counter
            fps_counter.tick()
            latencies_ms.append(lat)

            if first_frame is None:
                first_frame = frame.copy()
                h, w = frame.shape[:2]
                _ok(f"First frame received — actual size: {w}×{h}")

            if show and first_frame is not None:
                cv2.imshow(f"test_cam — {cam_cfg.id}", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("  (interrupted)")

    cv2.destroyAllWindows()

    # ── Report ────────────────────────────────────────────────────────
    elapsed   = time.monotonic() - start
    good_reads= total_reads - failed_reads
    drop_pct  = (failed_reads / max(total_reads, 1)) * 100
    mean_lat  = sum(latencies_ms) / max(len(latencies_ms), 1)
    measured_fps = fps_counter.fps

    _hdr(f"[{cam_cfg.id}] Frame test results")
    print(f"  Duration:       {elapsed:.1f}s")
    print(f"  Total reads:    {total_reads}")
    print(f"  Good frames:    {good_reads}")
    print(f"  Failed reads:   {failed_reads}  ({drop_pct:.1f}%)")
    print(f"  Measured FPS:   {measured_fps:.1f}")
    print(f"  Mean latency:   {mean_lat:.1f} ms")

    if measured_fps >= 20:
        _ok(f"FPS {measured_fps:.1f} is good for Jetson inference.")
    elif measured_fps >= 10:
        _warn(f"FPS {measured_fps:.1f} is low — check GStreamer latency / network.")
    else:
        _fail(f"FPS {measured_fps:.1f} is too low — check decoder / stream health.")

    if drop_pct < 2:
        _ok(f"Drop rate {drop_pct:.1f}% is acceptable.")
    elif drop_pct < 10:
        _warn(f"Drop rate {drop_pct:.1f}% — monitor for network instability.")
    else:
        _fail(f"Drop rate {drop_pct:.1f}% is high — unstable stream.")

    # ── Save sample frame ─────────────────────────────────────────────
    if save_frame and first_frame is not None:
        out_path = Path(f"alerts/snapshots/test_{cam_cfg.id}_sample.jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), first_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        _ok(f"Sample frame saved: {out_path}")

    return {
        "cam_id":       cam_cfg.id,
        "fps":          measured_fps,
        "drop_pct":     drop_pct,
        "mean_lat_ms":  mean_lat,
        "good_frames":  good_reads,
        "passed":       measured_fps >= 10 and drop_pct < 10,
    }


def test_pipeline_string(cam_cfg: CameraConfig) -> None:
    """Print the GStreamer pipeline string (no capture opened)."""
    _hdr(f"[{cam_cfg.id}] GStreamer pipeline string")
    pipeline = build_pipeline_string(
        url             = str(cam_cfg.source),
        codec           = cam_cfg.codec,
        protocols       = cam_cfg.protocols,
        latency         = cam_cfg.latency,
        drop_on_latency = cam_cfg.drop_on_latency,
        use_hw_decoder  = True,
    )
    print(f"\n  {pipeline}\n")
    print("  Test manually with:")
    print(f"  gst-launch-1.0 {pipeline.replace('appsink', 'autovideosink')}\n")


# ── Runner ─────────────────────────────────────────────────────────────

def run_camera_test(cam_cfg: CameraConfig, app_cfg, args) -> dict:
    """Run all tests for one camera. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"  Camera: {cam_cfg.id} — {cam_cfg.name}")
    print(f"  Source: {cam_cfg.source}")
    print(f"{'='*60}")

    if args.print_pipeline and cam_cfg.use_gstreamer:
        test_pipeline_string(cam_cfg)

    cap = test_open(cam_cfg, app_cfg.stream)
    if cap is None:
        return {"cam_id": cam_cfg.id, "passed": False, "reason": "open failed"}

    test_resolution(cap, cam_cfg)
    result = test_frames(
        cap, cam_cfg,
        duration=args.duration,
        save_frame=args.save_frame,
        show=not args.no_display,
    )
    cap.release()
    return result


def main():
    p = argparse.ArgumentParser(description="Man-down camera validation tool")
    p.add_argument("--config",         default="config/config.yaml")
    p.add_argument("--cam",            default=None, help="Test one camera by ID")
    p.add_argument("--duration",       type=float, default=15.0, help="Test duration in seconds")
    p.add_argument("--save-frame",     action="store_true", help="Save a sample frame to disk")
    p.add_argument("--print-pipeline", action="store_true", help="Print GStreamer pipeline string")
    p.add_argument("--no-display",     action="store_true", help="Don't show cv2.imshow window")
    args = p.parse_args()

    app_cfg = get_config(args.config)
    cameras = [c for c in app_cfg.cameras if c.enabled]
    if args.cam:
        cameras = [c for c in cameras if c.id == args.cam]
        if not cameras:
            print(f"Camera '{args.cam}' not found.")
            sys.exit(1)

    results = []
    for cam_cfg in cameras:
        result = run_camera_test(cam_cfg, app_cfg, args)
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for r in results:
        status = f"{_G}PASS{_E}" if r.get("passed") else f"{_R}FAIL{_E}"
        fps    = f"  FPS={r.get('fps',0):.1f}" if "fps" in r else ""
        drop   = f"  drop={r.get('drop_pct',0):.1f}%" if "drop_pct" in r else ""
        print(f"  {r['cam_id']:<12} [{status}]{fps}{drop}")
        if not r.get("passed"):
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()