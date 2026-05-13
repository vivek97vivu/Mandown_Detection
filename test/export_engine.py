"""
export_engine.py
----------------
Export utilities for converting model weights to optimised formats.

  --export-onnx    Export RTMPose .pth → .onnx  (needed for rtmlib backend)
  --export-trt     Export YOLO .pt → TensorRT .engine  (already done in your case)

Usage
-----
    # Export RTMPose to ONNX (do this once, then set backend: rtmlib in config.yaml)
    python test/export_engine.py --export-onnx

    # Export YOLO to TensorRT engine (already done — yolo26s.engine exists)
    python test/export_engine.py --export-trt

    # Custom paths
    python test/export_engine.py --export-onnx \\
        --pth  models/pose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \\
        --cfg  models/pose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192.py \\
        --out  models/pose/rtmpose.onnx

After successful export
-----------------------
Edit config/config.yaml:

    pose:
      path: "models/pose/rtmpose.onnx"
      backend: "rtmlib"          # ← change from mmpose to rtmlib
      # mmpose_config no longer needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("export_engine")

_G = "\033[32m"
_Y = "\033[33m"
_R = "\033[31m"
_B = "\033[1m"
_E = "\033[0m"


# ── RTMPose .pth → .onnx ─────────────────────────────────────────────

def export_rtmpose_onnx(
    pth_path:   str,
    cfg_path:   str,
    output:     str,
    input_size: tuple = (192, 256),   # (W, H)
    device:     str   = "cuda",
    opset:      int   = 17,
) -> bool:
    """
    Export RTMPose .pth to ONNX using MMPose's built-in export tool.

    Parameters
    ----------
    pth_path   : path to .pth checkpoint
    cfg_path   : path to matching MMPose .py config
    output     : desired output .onnx path
    input_size : (W, H) model input — must match the config
    device     : 'cuda' or 'cpu'
    opset      : ONNX opset version (17 recommended)

    Returns True on success.
    """
    print(f"\n{_B}Exporting RTMPose .pth → .onnx{_E}")
    print(f"  PTH:    {pth_path}")
    print(f"  Config: {cfg_path}")
    print(f"  Output: {output}")
    print(f"  Size:   {input_size[0]}×{input_size[1]}  opset={opset}\n")

    pth  = Path(pth_path)
    cfg  = Path(cfg_path)
    out  = Path(output)

    if not pth.exists():
        print(f"{_R}✗ .pth not found: {pth}{_E}")
        return False
    if not cfg.exists():
        print(f"{_R}✗ MMPose config not found: {cfg}{_E}")
        print(f"  Download from:")
        print(f"  https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py")
        return False

    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from mmpose.apis import init_model  # type: ignore

        device_str = "cuda:0" if device == "cuda" else "cpu"
        print(f"  Loading MMPose model on {device_str}...")
        model = init_model(str(cfg), str(pth), device=device_str)
        model.eval()

        W, H   = input_size
        dummy  = torch.randn(1, 3, H, W).to(device_str)

        print(f"  Running torch.onnx.export (opset {opset})...")
        import torch.onnx
        torch.onnx.export(
            model,
            dummy,
            str(out),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
        )

        print(f"\n{_G}✓ ONNX exported successfully: {out}{_E}")
        print(f"  File size: {out.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"\n  Next step — update config.yaml:")
        print(f"    pose:")
        print(f"      path: \"{out}\"")
        print(f"      backend: \"rtmlib\"")
        return True

    except ImportError:
        print(f"{_R}✗ MMPose not installed.{_E}")
        print("  Install: pip install mmpose mmcv")
        return False
    except Exception as e:
        print(f"{_R}✗ Export failed: {e}{_E}")
        logger.exception("ONNX export error")
        return False


# ── YOLO .pt → TensorRT .engine ──────────────────────────────────────

def export_yolo_trt(
    pt_path:    str,
    output:     str,
    input_size: int  = 640,
    device:     int  = 0,
    fp16:       bool = True,
) -> bool:
    """
    Export YOLO .pt to TensorRT .engine using Ultralytics built-in export.
    Your yolo26s.engine is already done — use this if you switch models.
    """
    print(f"\n{_B}Exporting YOLO .pt → TensorRT .engine{_E}")
    print(f"  PT:     {pt_path}")
    print(f"  Output: {output}")
    print(f"  Size:   {input_size}  FP16={fp16}\n")

    pt = Path(pt_path)
    if not pt.exists():
        print(f"{_R}✗ .pt not found: {pt}{_E}")
        return False

    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO(str(pt))
        model.export(
            format="engine",
            imgsz=input_size,
            device=device,
            half=fp16,
            simplify=True,
        )
        engine_path = pt.with_suffix(".engine")
        if engine_path.exists():
            if str(engine_path) != output:
                engine_path.rename(output)
            print(f"\n{_G}✓ TensorRT engine exported: {output}{_E}")
            print(f"  File size: {Path(output).stat().st_size / 1024 / 1024:.1f} MB")
            print(f"\n  Next step — update config.yaml:")
            print(f"    yolo:")
            print(f"      path: \"{output}\"")
            return True
        else:
            print(f"{_R}✗ Engine file not found after export.{_E}")
            return False

    except ImportError:
        print(f"{_R}✗ Ultralytics not installed: pip install ultralytics{_E}")
        return False
    except Exception as e:
        print(f"{_R}✗ TRT export failed: {e}{_E}")
        logger.exception("TRT export error")
        return False


# ── Verify ONNX ───────────────────────────────────────────────────────

def verify_onnx(onnx_path: str) -> bool:
    """Quick sanity-check that the exported ONNX loads and runs."""
    print(f"\n{_B}Verifying ONNX: {onnx_path}{_E}")
    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np

        session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        inp     = session.get_inputs()[0]
        W, H    = 192, 256
        dummy   = np.random.randn(1, 3, H, W).astype(np.float32)
        outputs = session.run(None, {inp.name: dummy})

        print(f"  Input  shape: {inp.shape}")
        print(f"  Output shape: {[o.shape for o in outputs]}")
        print(f"  Providers:    {session.get_providers()}")
        print(f"{_G}✓ ONNX verified — model runs correctly.{_E}")
        return True
    except Exception as e:
        print(f"{_R}✗ Verification failed: {e}{_E}")
        return False


# ── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Model export utility for man-down detection")
    p.add_argument("--export-onnx",  action="store_true", help="Export RTMPose .pth → .onnx")
    p.add_argument("--export-trt",   action="store_true", help="Export YOLO .pt → TensorRT .engine")
    p.add_argument("--verify-onnx",  action="store_true", help="Verify an existing .onnx file")
    p.add_argument("--pth",    default=None, help="Path to RTMPose .pth file")
    p.add_argument("--cfg",    default=None, help="Path to MMPose .py config")
    p.add_argument("--pt",     default=None, help="Path to YOLO .pt file")
    p.add_argument("--out",    default=None, help="Output path")
    p.add_argument("--opset",  type=int, default=17, help="ONNX opset version")
    p.add_argument("--fp16",   action="store_true", default=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not any([args.export_onnx, args.export_trt, args.verify_onnx]):
        p.print_help()
        sys.exit(0)

    # ── Load config for defaults ───────────────────────────────────
    try:
        from config.config_loader import get_config
        cfg = get_config()
        default_pth = cfg.worker.rtmpose_model_path
        default_cfg = getattr(cfg.worker, "rtmpose_config", "")
        default_pt  = cfg.worker.yolo_model_path.replace(".engine", ".pt")
    except Exception:
        default_pth = "models/pose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
        default_cfg = "test/rtmpose-m_8xb256-420e_coco-256x192.py"
        default_pt  = "models/yolo/yolo26s.pt"

    ok = True

    if args.export_onnx:
        pth = args.pth or default_pth
        cfg_path = args.cfg or default_cfg
        out = args.out or str(Path(pth).with_suffix(".onnx"))
        ok &= export_rtmpose_onnx(pth, cfg_path, out, opset=args.opset, device=args.device)
        if ok:
            verify_onnx(out)

    if args.export_trt:
        pt  = args.pt or default_pt
        out = args.out or str(Path(pt).with_suffix(".engine"))
        ok &= export_yolo_trt(pt, out, fp16=args.fp16)

    if args.verify_onnx:
        onnx = args.out or str(Path(default_pth).with_suffix(".onnx"))
        ok &= verify_onnx(onnx)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()