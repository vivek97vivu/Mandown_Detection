
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import get_config
from core.decision import DecisionEngine
from core.detector import PersonDetector
from core.pose import PoseEstimator
from core.temporal import TemporalFilter
from utils.drawing import DrawingConfig, draw_frame
from utils.fps import FPSCounter, SectionTimer
from utils.logger import get_logger, setup_logging

setup_logging(level="WARNING")   # quiet during benchmark
logger = get_logger("benchmark")

# ── Terminal colours ──────────────────────────────────────────────────
_B = "\033[1m"
_G = "\033[32m"
_Y = "\033[33m"
_R = "\033[31m"
_C = "\033[36m"
_E = "\033[0m"


# ── Helpers ───────────────────────────────────────────────────────────

def _dummy_frame(w: int, h: int) -> np.ndarray:
    """Generate a realistic-looking dummy BGR frame (gradient + noise)."""
    base  = np.zeros((h, w, 3), dtype=np.uint8)
    # Horizontal gradient
    base[:, :, 0] = np.linspace(30, 80, w, dtype=np.uint8)
    base[:, :, 2] = np.linspace(20, 60, w, dtype=np.uint8)
    # Random noise to prevent detector optimisations
    noise = np.random.randint(0, 15, (h, w, 3), dtype=np.uint8)
    return cv2.add(base, noise)


def _gpu_memory_mb() -> Optional[float]:
    """Query GPU memory usage via nvidia-smi (Jetson compatible)."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return float(out.split("\n")[0])
    except Exception:
        return None


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_d) - 1)
    return sorted_d[lo] + (sorted_d[hi] - sorted_d[lo]) * (k - lo)


def _bar(value: float, max_val: float, width: int = 30, color: str = _G) -> str:
    filled = int(round(value / max(max_val, 1e-6) * width))
    filled = min(filled, width)
    return color + "█" * filled + _E + "░" * (width - filled)


# ── Stage benchmarks ──────────────────────────────────────────────────

def benchmark_yolo(
    detector: PersonDetector,
    frames:   List[np.ndarray],
) -> Dict:
    """Benchmark YOLO detection stage alone."""
    print(f"\n{_B}Stage 1 — YOLO Detection{_E}")
    latencies = []

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(frames)}] mean={statistics.mean(latencies[-100:]):.1f}ms  "
                  f"persons={len(detections)}")

    return _stats("yolo", latencies)


def benchmark_pose(
    detector:  PersonDetector,
    estimator: PoseEstimator,
    frames:    List[np.ndarray],
) -> Dict:
    """Benchmark Stage 1 + Stage 2 combined (realistic: pose only runs on detections)."""
    print(f"\n{_B}Stage 2 — YOLO + RTMPose Combined{_E}")
    yolo_lats, pose_lats, total_lats = [], [], []

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t1 = time.perf_counter()

        poses = estimator.estimate(frame, detections) if detections else []
        t2 = time.perf_counter()

        yolo_lats.append((t1 - t0) * 1000)
        pose_lats.append((t2 - t1) * 1000)
        total_lats.append((t2 - t0) * 1000)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(frames)}]  "
                  f"yolo={statistics.mean(yolo_lats[-100:]):.1f}ms  "
                  f"pose={statistics.mean(pose_lats[-100:]):.1f}ms  "
                  f"persons={len(detections)}")

    return {
        "yolo":  _stats("yolo_in_combined", yolo_lats),
        "pose":  _stats("pose", pose_lats),
        "total": _stats("yolo+pose", total_lats),
    }


def benchmark_full_pipeline(
    detector:  PersonDetector,
    estimator: PoseEstimator,
    decision:  DecisionEngine,
    temporal:  TemporalFilter,
    frames:    List[np.ndarray],
    draw_cfg:  DrawingConfig,
) -> Dict:
    """Benchmark the complete end-to-end pipeline per frame."""
    print(f"\n{_B}Full Pipeline — End-to-End{_E}")
    timer    = SectionTimer()
    fps_ctr  = FPSCounter(window=50)
    frame_idx = 0
    fps_snapshots: List[float] = []

    for i, frame in enumerate(frames):
        frame_idx += 1

        with timer("detect"):
            dets = detector.detect(frame)

        with timer("pose"):
            poses = estimator.estimate(frame, dets) if dets else []

        with timer("decision"):
            decisions = [decision.classify(p) for p in poses]

        with timer("temporal"):
            events = temporal.update(decisions, frame_idx)

        with timer("draw"):
            _ = draw_frame(frame.copy(), poses, decisions, events, fps_ctr.fps, draw_cfg)

        fps_ctr.tick()

        if (i + 1) % 50 == 0:
            fps_snapshots.append(fps_ctr.fps)
            print(f"  [{i+1}/{len(frames)}]  FPS={fps_ctr.fps:.1f}  "
                  f"detect={timer.mean_ms('detect'):.1f}ms  "
                  f"pose={timer.mean_ms('pose'):.1f}ms  "
                  f"draw={timer.mean_ms('draw'):.1f}ms")

    breakdown = timer.report(last_n=len(frames))
    return {
        "breakdown_ms":   breakdown,
        "sustained_fps":  fps_ctr.fps,
        "fps_snapshots":  fps_snapshots,
        "thermal_drop":   _detect_thermal_throttle(fps_snapshots),
    }


def benchmark_person_scaling(
    detector:  PersonDetector,
    estimator: PoseEstimator,
    frame:     np.ndarray,
    n_trials:  int = 100,
) -> Dict:
    """
    Measure how pipeline latency scales with number of detected persons.
    Injects synthetic detections with fake crops to isolate pose cost.
    """
    print(f"\n{_B}Person Scaling — Pose Latency vs Person Count{_E}")
    results = {}

    for n_persons in [0, 1, 2, 4, 8]:
        # Build fake detections with real crops
        from core.detector import Detection
        h, w = frame.shape[:2]
        crop = frame[:min(256, h), :min(192, w)].copy()
        fake_dets = [
            Detection(track_id=i, bbox=(0, 0, 192, 256), confidence=0.9, crop=crop)
            for i in range(n_persons)
        ]

        latencies = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            estimator.estimate(frame, fake_dets)
            latencies.append((time.perf_counter() - t0) * 1000)

        mean_ms = statistics.mean(latencies)
        bar     = _bar(mean_ms, 200)
        print(f"  {n_persons:>2} person(s)  {mean_ms:>7.1f} ms  {bar}")
        results[str(n_persons)] = round(mean_ms, 2)

    return results


# ── Thermal throttle detection ────────────────────────────────────────

def _detect_thermal_throttle(fps_snapshots: List[float]) -> bool:
    """
    True if FPS dropped more than 20% from peak to end.
    Indicates Jetson thermal throttling during the run.
    """
    if len(fps_snapshots) < 3:
        return False
    peak = max(fps_snapshots)
    last = fps_snapshots[-1]
    return (peak - last) / max(peak, 1) > 0.20


# ── Stats helper ──────────────────────────────────────────────────────

def _stats(name: str, data: List[float]) -> Dict:
    if not data:
        return {"name": name}
    return {
        "name":   name,
        "n":      len(data),
        "mean":   round(statistics.mean(data),   2),
        "median": round(statistics.median(data), 2),
        "p95":    round(_percentile(data, 95),   2),
        "p99":    round(_percentile(data, 99),   2),
        "min":    round(min(data),               2),
        "max":    round(max(data),               2),
        "fps":    round(1000 / max(statistics.mean(data), 0.01), 1),
    }


# ── Report printer ────────────────────────────────────────────────────

def print_report(results: dict, w: int, h: int) -> None:
    sep = "─" * 62

    print(f"\n{_B}{'='*62}")
    print(f"  MAN-DOWN DETECTION — BENCHMARK REPORT")
    print(f"  Resolution: {w}×{h}  |  Device: Jetson (CUDA)")
    print(f"{'='*62}{_E}")

    # Stage latencies
    print(f"\n{_B}Pipeline Stage Latencies{_E}")
    print(f"  {sep}")
    print(f"  {'Stage':<25} {'Mean':>8} {'P95':>8} {'P99':>8} {'FPS':>7}")
    print(f"  {sep}")

    stages = {
        "YOLO Detection":     results.get("yolo", {}),
        "RTMPose (per frame)":results.get("pose_combined", {}).get("pose", {}),
        "YOLO + Pose":        results.get("pose_combined", {}).get("total", {}),
    }
    for label, s in stages.items():
        if s:
            thr_color = _G if s.get("fps", 0) >= 25 else (_Y if s.get("fps", 0) >= 15 else _R)
            print(f"  {label:<25} {s.get('mean',0):>7.1f}ms {s.get('p95',0):>7.1f}ms "
                  f"{s.get('p99',0):>7.1f}ms {thr_color}{s.get('fps',0):>6.1f}{_E}")

    # Full pipeline breakdown
    fp = results.get("full_pipeline", {})
    breakdown = fp.get("breakdown_ms", {})
    if breakdown:
        print(f"\n{_B}Full Pipeline Breakdown (mean ms){_E}")
        print(f"  {sep}")
        total_ms = sum(breakdown.values())
        for stage, ms in breakdown.items():
            bar = _bar(ms, total_ms, width=25)
            print(f"  {stage:<12}  {ms:>6.1f} ms  {bar}")
        fps = fp.get("sustained_fps", 0)
        fps_color = _G if fps >= 25 else (_Y if fps >= 15 else _R)
        print(f"\n  Sustained FPS:  {fps_color}{_B}{fps:.1f}{_E}")

        # Thermal warning
        if fp.get("thermal_drop"):
            print(f"\n  {_Y}⚠  Thermal throttling detected — FPS dropped >20% during run.")
            print(f"     Consider reducing input_size or adding a heatsink.{_E}")
        else:
            print(f"  {_G}✓  No thermal throttling detected.{_E}")

    # Person scaling
    scaling = results.get("person_scaling", {})
    if scaling:
        print(f"\n{_B}Pose Latency by Person Count{_E}")
        print(f"  {sep}")
        for n, ms in scaling.items():
            bar = _bar(ms, 200, width=25)
            print(f"  {n:>2} person(s)  {ms:>7.1f} ms  {bar}")

    # GPU memory
    mem = results.get("gpu_memory_mb")
    if mem:
        print(f"\n  GPU memory used: {mem:.0f} MB")

    # Jetson targets
    print(f"\n{_B}Jetson Targets{_E}")
    fps_val = fp.get("sustained_fps", 0)
    checks = [
        ("FPS ≥ 25 (real-time)",   fps_val >= 25),
        ("FPS ≥ 15 (acceptable)",  fps_val >= 15),
        ("No thermal throttling",  not fp.get("thermal_drop", True)),
    ]
    for label, passed in checks:
        icon  = f"{_G}✓{_E}" if passed else f"{_R}✗{_E}"
        print(f"  {icon}  {label}")

    print(f"\n{_B}{'='*62}{_E}\n")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Man-down pipeline benchmark")
    p.add_argument("--config",      default="config/config.yaml")
    p.add_argument("--frames",      type=int,   default=500)
    p.add_argument("--resolution",  type=int,   nargs=2, default=[1280, 720],
                   metavar=("W", "H"))
    p.add_argument("--video",       default=None, help="Use real video instead of dummy frames")
    p.add_argument("--no-pose",     action="store_true", help="Skip RTMPose (YOLO only)")
    p.add_argument("--no-scaling",  action="store_true", help="Skip person scaling test")
    p.add_argument("--save-report", action="store_true", help="Save benchmark.json")
    args = p.parse_args()

    w, h = args.resolution
    n_frames = args.frames

    print(f"{_B}Man-Down Detection — Jetson Benchmark{_E}")
    print(f"  Frames: {n_frames}  |  Resolution: {w}×{h}")

    app_cfg  = get_config(args.config)
    wcfg     = app_cfg.worker

    # ── Load frames ───────────────────────────────────────────────────
    if args.video:
        cap = cv2.VideoCapture(args.video)
        frames = []
        while len(frames) < n_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret:
                frames.append(cv2.resize(frame, (w, h)))
        cap.release()
        print(f"  Loaded {len(frames)} frames from {args.video}")
    else:
        print("  Generating dummy frames...")
        frames = [_dummy_frame(w, h) for _ in range(n_frames)]

    # ── Load models ───────────────────────────────────────────────────
    gpu_mem_before = _gpu_memory_mb()

    print("\n  Loading YOLO...")
    detector = PersonDetector(
        model_path=wcfg.yolo_model_path,
        device=wcfg.device,
        input_size=getattr(wcfg, "yolo_input_size", 640),
        conf_threshold=getattr(wcfg, "yolo_conf", 0.4),
        use_tracking=False,    # tracking off for clean benchmark
    )
    detector.load()

    estimator = None
    if not args.no_pose:
        print("  Loading RTMPose...")
        estimator = PoseEstimator(
            model_path=wcfg.rtmpose_model_path,
            device=wcfg.device,
            backend=getattr(wcfg, "pose_backend", "rtmlib"),
        )
        estimator.load()

    gpu_mem_after = _gpu_memory_mb()
    gpu_mem_used  = (gpu_mem_after - gpu_mem_before) if (gpu_mem_before and gpu_mem_after) else None

    # ── Run benchmarks ────────────────────────────────────────────────
    results: dict = {}

    if gpu_mem_used:
        results["gpu_memory_mb"] = round(gpu_mem_used, 1)

    results["yolo"] = benchmark_yolo(detector, frames[:min(200, n_frames)])

    if estimator and not args.no_pose:
        results["pose_combined"] = benchmark_pose(detector, estimator, frames[:min(200, n_frames)])

        decision = DecisionEngine(wcfg.decision)
        temporal = TemporalFilter(wcfg.temporal)
        draw_cfg = wcfg.drawing

        results["full_pipeline"] = benchmark_full_pipeline(
            detector, estimator, decision, temporal, frames, draw_cfg
        )

        if not args.no_scaling:
            results["person_scaling"] = benchmark_person_scaling(
                detector, estimator, frames[0], n_trials=50
            )

    # ── Print report ──────────────────────────────────────────────────
    print_report(results, w, h)

    # ── Save JSON ─────────────────────────────────────────────────────
    if args.save_report:
        out_path = Path("results/benchmark.json")
        out_path.parent.mkdir(exist_ok=True)
        results["meta"] = {
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "resolution": f"{w}x{h}",
            "n_frames":   n_frames,
        }
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  Report saved: {out_path}")

    detector.release()
    if estimator:
        estimator.release()


if __name__ == "__main__":
    main()