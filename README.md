# 🦺 Man-Down Detection Engine
### 🚨 Real-Time Fall & Incapacitation Detection for Industrial Safety Systems

A **production-grade AI pipeline** built for **real-time CCTV / RTSP monitoring** on construction sites and oil & gas facilities, combining **fast person detection + pose estimation + temporal reasoning** for high-precision man-down alerts.

> ⚙️ Powered by **YOLO TensorRT (Detection)** + **RTMPose (Keypoint Estimation)**
> 🧠 Designed for **low false positives, high reliability industrial deployments**
> 🧩 Part of the **CampNeuron AI Series** — engineered by the **Algosium AI Team**

---

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![YOLO](https://img.shields.io/badge/YOLO-TensorRT-success?logo=nvidia&logoColor=white)](#)
[![RTMPose](https://img.shields.io/badge/RTMPose-SimCC-orange)](#)
[![GStreamer](https://img.shields.io/badge/GStreamer-H265-blue)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20|%20Jetson%20|%20x86__64-lightgrey?logo=linux&logoColor=white)](#)

---

## ⚡ Core Stack

| Component | Purpose |
|---|---|
| 🔍 **YOLO TensorRT Engine** | Real-time person detection — hardware accelerated, FP16 |
| 🦴 **RTMPose SimCC** | 17-keypoint body pose estimation per detected person |
| 📐 **Skeleton Geometry Engine** | Body angle, aspect ratio, head position, limb symmetry analysis |
| 🧠 **Multi-Factor Decision Engine** | Weighted evidence scoring — UPRIGHT / CROUCHING / FALLEN |
| ⏱️ **Temporal State Machine** | Dwell-time gating — prevents false alerts from stumbles or bends |
| 🎥 **GStreamer H265 Pipeline** | Low-latency RTSP decode via GPU (nvh265dec / avdec_h265) |
| 🔁 **ByteTrack Identity Tracking** | Per-person consistent ID across frames and reconnects |
| 📸 **Alert Evidence System** | Auto-saves snapshot + metadata JSON + pre/post video clip |
| 🧵 **Multi-Camera Threading** | Multiple RTSP + USB cameras, GPU-serialised, Qt-safe display |
| ⚙️ **YAML Config Engine** | All thresholds configurable — no code changes needed |

---

## 🚀 Pipeline Overview

```
RTSP / USB Camera
       │
       ▼
┌─────────────────┐
│  GStreamer /    │  H265 hardware decode (nvh265dec)
│  V4L2 Capture  │  or software fallback (avdec_h265)
└────────┬────────┘
         │  Raw BGR Frame
         ▼
┌─────────────────┐
│  YOLO Detection │  Person bbox + ByteTrack ID
│  (TRT Engine)   │  FP16 · ~5ms per frame
└────────┬────────┘
         │  Person crops (with padding)
         ▼
┌─────────────────┐
│  RTMPose        │  17 COCO keypoints per person
│  Pose Estimator │  SimCC decoding · ~1ms per person
└────────┬────────┘
         │  PoseResult (keypoints + scores)
         ▼
┌─────────────────┐
│  Skeleton       │  Body angle · Aspect ratio
│  Geometry       │  Head height · Limb symmetry
└────────┬────────┘
         │  Metrics dict
         ▼
┌─────────────────┐
│  Decision       │  Weighted multi-factor score
│  Engine         │  UPRIGHT / CROUCHING / FALLEN / UNCERTAIN
└────────┬────────┘
         │  DecisionResult (state + score)
         ▼
┌─────────────────┐
│  Temporal       │  Dwell-time gate (5s default)
│  State Machine  │  Hysteresis · EMA smoothing
└────────┬────────┘
         │  Alert events
         ▼
┌─────────────────┐
│  Alert Handler  │  Snapshot · Metadata JSON · Video clip
│  + Overlay Draw │  Skeleton · BBox · Banner · HUD
└─────────────────┘
```

---

## 📁 Project Structure

```
mandown_detection/
├── main.py                    # Entry point — multi-camera orchestrator
├── config/
│   ├── config.yaml            # Master config — all tuneable parameters
│   └── config_loader.py       # YAML → typed dataclass loader + validation
├── core/
│   ├── detector.py            # Stage 1 — YOLO person detection + ByteTrack
│   ├── pose.py                # Stage 2 — RTMPose keypoint estimation
│   ├── pose_trt.py            # TensorRT SimCC inference backend
│   ├── skeleton.py            # Geometric body analysis (angle, ratio, etc.)
│   ├── decision.py            # Multi-factor man-down classifier
│   ├── temporal.py            # Dwell-time state machine + alert gating
│   ├── tracker.py             # Track registry — identity across reconnects
│   ├── alert.py               # Snapshot + video clip alert handler
│   └── worker.py              # Pipeline orchestrator — wires all modules
├── stream/
│   ├── capture.py             # Unified USB + RTSP capture interface
│   ├── gstreamer.py           # GStreamer H264/H265 pipeline builder
│   └── reconnect.py           # Auto-reconnect loop with exponential backoff
├── utils/
│   ├── drawing.py             # On-screen overlay — skeleton, bbox, banner, HUD
│   ├── fps.py                 # FPS counter + section profiler
│   ├── geometry.py            # Bbox ops, point-in-polygon, danger zones
│   └── logger.py              # Coloured console + rotating file + JSON logger
├── models/
│   ├── yolo/
│   │   └── yolo26s.engine     # YOLO TensorRT engine (FP16)
│   └── pose/
│       ├── rtmpose-m.onnx     # RTMPose-M ONNX (rtmlib backend)
│       └── rtmpose-m.engine   # RTMPose-M TRT engine (optional)
├── alerts/
│   ├── snapshots/             # Alert JPEG images with overlay
│   ├── pose_confirm/          # Alert metadata JSON files
│   └── video_clips/           # Pre+post event AVI clips
├── test/
│   ├── test_cam.py            # Camera validation + stream health check
│   ├── benchmark.py           # Full pipeline latency benchmark
│   └── export_engine.py       # PTH → ONNX → TRT engine export utility
└── logs/
    ├── mandown.log            # Rotating pipeline log
    └── alerts.log             # WARNING+ alert-only log
```

---

## ⚙️ Configuration — `config/config.yaml`

Every parameter is tunable without touching code:

```yaml
cameras:
  - id: cam_1
    source: 0                       # USB webcam
    use_gstreamer: false

  - id: cam_2
    source: "rtsp://..."            # RTSP stream
    use_gstreamer: true
    codec: "h265"                   # h264 | h265

decision:
  angle_fallen_min: 75.0            # degrees from vertical → fallen
  aspect_fallen_min: 1.20           # bbox wider than tall → lying flat
  fallen_score_threshold: 0.70      # raise to reduce false positives
  min_bbox_height_ratio: 0.15       # ignore tiny partial detections

temporal:
  dwell_frames: 150                 # frames on ground before alert (~6s @25fps)
  recovery_frames: 75               # frames upright before alert clears

alert:
  save_snapshot: true
  save_video_clip: true
  clip_pre_frames: 50               # 2s before the fall
  clip_post_frames: 75              # 3s after the fall
  snapshot_cooldown_s: 30.0         # min gap between repeat alerts
```

**Tuning guide for environments:**

| Environment | Key Setting | Recommended Value |
|---|---|---|
| Construction site (outdoor) | `dwell_frames` | 100–150 |
| Oil & gas confined spaces | `crouching_as_fallen` | `true` |
| Office / indoor (many desks) | `aspect_fallen_min` | 1.20+ |
| High camera angle (overhead) | `angle_fallen_min` | 60–65° |
| Crowded scene | `min_detection_conf` | 0.50+ |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
conda activate mandown
pip install ultralytics rtmlib onnxruntime-gpu pycuda
pip install ~/TensorRT-10.7.0.23/python/tensorrt-10.7.0-cp312-none-linux_x86_64.whl
```

### 2. Place models

```
models/yolo/yolo26s.engine        ← YOLO TensorRT engine
models/pose/rtmpose-m.onnx        ← RTMPose ONNX model
```

### 3. Configure cameras in `config/config.yaml`

### 4. Run

```bash
# All cameras
python main.py

# Single camera only
python main.py --cam cam_2

# Headless (no display — SSH / server deployment)
python main.py --no-display

# Debug mode
python main.py --log-level DEBUG
```

---

## 🧪 Testing & Benchmarking

```bash
# Test camera streams — resolution, FPS, drop rate
python test/test_cam.py --cam cam_2 --duration 30

# Full pipeline benchmark — stage latencies, sustained FPS, thermal check
python test/benchmark.py --frames 500 --resolution 1920 1080

# Export RTMPose .pth → .onnx (needed for rtmlib backend)
python test/export_engine.py --export-onnx

# Verify ONNX model runs correctly
python test/export_engine.py --verify-onnx
```

---

## 📊 Performance

| Stage | RTX 4080 Super | Jetson Orin |
|---|---|---|
| YOLO Detection (TRT FP16) | ~5 ms | ~12 ms |
| RTMPose per person (ONNX) | ~1 ms | ~4 ms |
| Skeleton + Decision | < 0.5 ms | < 0.5 ms |
| Drawing / Overlay | ~2 ms | ~5 ms |
| **End-to-end (1 person)** | **~9 ms (~110 FPS)** | **~22 ms (~45 FPS)** |
| **End-to-end (4 persons)** | **~15 ms (~67 FPS)** | **~35 ms (~28 FPS)** |

---

## 🚨 Alert Evidence

Every man-down event generates three files automatically:

```
alerts/snapshots/mandown_track6_20260514_121421.jpg      ← Annotated frame
alerts/pose_confirm/mandown_track6_20260514_121421.json  ← Full metadata
alerts/video_clips/mandown_track6_20260514_121421.avi    ← 2s before + 3s after
```

**Metadata JSON contains:**
```json
{
  "track_id": 6,
  "alert_state": "ALERTING",
  "smoothed_score": 0.847,
  "fallen_frames": 150,
  "timestamp": "2026-05-14T12:14:21",
  "decision": {
    "state": "FALLEN",
    "score": 0.86,
    "reason": "angle=0.95 | head=0.88 → final=0.86",
    "metrics": {
      "body_angle": 82.4,
      "bbox_aspect_ratio": 1.35,
      "head_height_ratio": 0.61,
      "kp_visible_count": 11
    }
  }
}
```

---

## 🛡️ False Positive Prevention

The system uses **four layers** of protection against false alerts:

| Layer | Mechanism | Guards Against |
|---|---|---|
| **1. Size filter** | `min_bbox_height_ratio` | Heads peeking over partitions |
| **2. Multi-factor score** | Weighted evidence (5 signals) | Single-metric errors |
| **3. Temporal dwell** | 150 frames (~6s) must stay fallen | Stumbles, bends, sitting |
| **4. EMA smoothing** | `score_ema_alpha: 0.20` | Frame-to-frame jitter |

---

## 🔧 Danger Zones (Optional)

Define restricted areas in `config.yaml` using normalised `[0,1]` coordinates:

```yaml
zones:
  - name: "confined_space_entry"
    enabled: true
    polygon:
      - [0.10, 0.20]
      - [0.40, 0.20]
      - [0.40, 0.90]
      - [0.10, 0.90]
```

Persons detected inside a zone are tagged in their track record — useful for HSE audit trails.

---

## 📋 Requirements

```
Python        3.12
CUDA          12.x
TensorRT      10.7.0
OpenCV        4.12.0 (with GStreamer + CUDA)
ultralytics   ≥ 8.x
onnxruntime-gpu
rtmlib        0.0.15
pycuda
PyYAML
numpy
```

---

## 🏗️ Engineered by

**Algosium AI Team** — CampNeuron AI Series

> Built for real-world industrial safety. Tested on construction sites and office environments.
> Designed to run 24/7 unattended with automatic stream reconnection and rotating log management.