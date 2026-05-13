"""
pose.py
-------
Stage-2: RTMPose keypoint estimator.
Takes BGR person crops from the YOLO detector and returns 17 COCO keypoints
per person, projected back into full-frame pixel coordinates.

RTMPose is loaded via MMPose / rtmlib for Jetson compatibility.
Falls back to ONNX Runtime if MMPose is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.detector import Detection

logger = logging.getLogger(__name__)

# COCO 17-keypoint indices — used throughout the project by name
KEYPOINT_NAMES = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]

KP = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}  # name → index lookup


@dataclass
class PoseResult:
    """Keypoint estimation result for one person."""
    track_id: int
    bbox: Tuple[int, int, int, int]           # original full-frame bbox
    confidence: float                          # YOLO detection confidence

    # Shape: (17, 2) — (x, y) in full-frame pixel coordinates
    keypoints: np.ndarray = field(repr=False)
    # Shape: (17,) — per-keypoint confidence scores from RTMPose
    kp_scores: np.ndarray = field(repr=False)

    @property
    def mean_kp_score(self) -> float:
        return float(np.mean(self.kp_scores))

    def keypoint(self, name: str) -> Tuple[float, float]:
        """Return (x, y) for a named keypoint."""
        return tuple(self.keypoints[KP[name]])

    def kp_score(self, name: str) -> float:
        """Return confidence for a named keypoint."""
        return float(self.kp_scores[KP[name]])

    def visible(self, name: str, min_score: float = 0.3) -> bool:
        """True if keypoint is detected with sufficient confidence."""
        return self.kp_score(name) >= min_score


class PoseEstimator:
    """
    RTMPose wrapper for Jetson deployment.

    Supports three backends:
      - 'tensorrt': TRT .engine (fastest — use this after trtexec export)
      - 'rtmlib'  : rtmlib ONNX (good alternative)
      - 'mmpose'  : MMPose full framework (.pth, no export needed)

    Parameters
    ----------
    model_path : str | Path
        Path to RTMPose .pth or .onnx weight file.
    input_size : tuple
        Model input (width, height). RTMPose-s = (192, 256).
    device : str
        'cuda' or 'cpu'.
    backend : str
        'rtmlib' or 'mmpose'.
    min_kp_score : float
        Keypoints below this score are treated as invisible.
    """

    # RTMPose-s default input size (width, height)
    DEFAULT_INPUT_SIZE = (192, 256)

    def __init__(
        self,
        model_path: str | Path,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        device: str = "cuda",
        backend: str = "rtmlib",
        min_kp_score: float = 0.3,
        mmpose_config: Optional[str] = None,   # path to .py config (mmpose backend only)
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size          # (W, H)
        self.device = device
        self.backend = backend
        self.min_kp_score = min_kp_score
        self.mmpose_config = mmpose_config    # e.g. "models/pose/rtmpose-m_...py"
        self._model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load RTMPose weights. Call once before inference loop."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"RTMPose weights not found: {self.model_path}")

        logger.info(
            "Loading RTMPose via backend='%s' from %s", self.backend, self.model_path
        )

        if self.backend == "tensorrt":
            self._load_tensorrt()
        elif self.backend == "rtmlib":
            self._load_rtmlib()
        elif self.backend == "mmpose":
            self._load_mmpose()
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'tensorrt', 'rtmlib', or 'mmpose'.")

        logger.info("RTMPose loaded successfully.")

    def release(self) -> None:
        self._model = None
        if hasattr(self, '_trt') and self._trt is not None:
            self._trt.release()
            self._trt = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def estimate(self, frame: np.ndarray, detections: List[Detection]) -> List[PoseResult]:
        """
        Run RTMPose on all person detections.

        Parameters
        ----------
        frame : np.ndarray
            Full-resolution BGR frame (used for coordinate projection).
        detections : List[Detection]
            Output of PersonDetector.detect().

        Returns
        -------
        List[PoseResult]
            One PoseResult per detection, in the same order.
        """
        if not detections:
            return []

        results: List[PoseResult] = []
        for det in detections:
            kps, scores = self._infer_single(det.crop, det.bbox, frame.shape)
            results.append(
                PoseResult(
                    track_id=det.track_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    keypoints=kps,
                    kp_scores=scores,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Private: backends
    # ------------------------------------------------------------------

    def _load_tensorrt(self) -> None:
        """Load RTMPose TensorRT .engine — fastest backend on Jetson/desktop GPU."""
        from core.pose_trt import RTMPoseTRT  # type: ignore
        self._trt = RTMPoseTRT(
            engine_path=self.model_path,
            input_size=self.input_size,
        )
        self._trt.load()
        self._infer_fn = self._infer_tensorrt
        logger.info("TRT backend loaded successfully.")

    def _infer_tensorrt(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Delegate to RTMPoseTRT — returns (keypoints, scores) in crop coords."""
        return self._trt.infer(crop)

    def _load_rtmlib(self) -> None:
        try:
            from rtmlib import RTMPose as _RTMPose  # type: ignore

            # rtmlib device: "cpu" or "cuda" — maps to ORT providers internally
            # Use "cuda" to get CUDAExecutionProvider; falls back to CPU if unavailable
            device = self.device if self.device in ("cpu", "cuda") else "cpu"
            self._model = _RTMPose(
                str(self.model_path),
                model_input_size=self.input_size,
                backend="onnxruntime",
                device=device,
            )
            self._infer_fn = self._infer_rtmlib
        except ImportError:
            logger.warning(
                "rtmlib not found, falling back to direct ONNX Runtime inference."
            )
            self._load_onnx_fallback()

    def _load_mmpose(self) -> None:
        try:
            from mmpose.apis import init_model  # type: ignore

            if self.mmpose_config:
                cfg_path = Path(self.mmpose_config)
            else:
                candidates = list(self.model_path.parent.glob("*.py"))
                if not candidates:
                    raise FileNotFoundError(
                        f"No MMPose config (.py) found beside {self.model_path}. "
                        "Set mmpose_config in config.yaml pose section."
                    )
                cfg_path = candidates[0]
                logger.info("Auto-detected MMPose config: %s", cfg_path)

            if not cfg_path.exists():
                raise FileNotFoundError(
                    f"MMPose config not found: {cfg_path}. "
                    "Download the matching .py from the mmpose configs repo."
                )

            device_str = "cuda:0" if self.device == "cuda" else "cpu"
            logger.info("Loading MMPose config=%s weights=%s device=%s",
                        cfg_path, self.model_path, device_str)

            self._model = init_model(str(cfg_path), str(self.model_path), device=device_str)
            self._infer_fn = self._infer_mmpose

        except ImportError as e:
            raise ImportError(
                "MMPose not installed. Install: pip install mmpose mmcv. "
                "Or export .pth to .onnx and set backend: rtmlib in config.yaml"
            ) from e

    def _load_onnx_fallback(self) -> None:
        """Direct ONNX Runtime inference — works on Jetson without extra libs."""
        import onnxruntime as ort  # type: ignore

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        onnx_path = self.model_path.with_suffix(".onnx")
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                "Export your .pth to ONNX or use rtmlib backend."
            )
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._infer_fn = self._infer_onnx

    # ------------------------------------------------------------------
    # Private: inference per backend
    # ------------------------------------------------------------------

    def _infer_single(
        self,
        crop: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch to the loaded backend and project keypoints."""
        kps_crop, scores = self._infer_fn(crop)        # (17,2) in crop coords
        kps_full = self._project_to_frame(kps_crop, bbox, crop.shape, frame_shape)
        return kps_full, scores

    def _infer_rtmlib(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """rtmlib returns (keypoints, scores) both shape (1, 17, ...)."""
        result = self._model(crop)
        # rtmlib returns list of (kps, scores) — take first person
        kps = np.array(result[0][0], dtype=np.float32)    # (17, 2)
        scores = np.array(result[1][0], dtype=np.float32) # (17,)
        return kps, scores

    def _infer_mmpose(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from mmpose.apis import inference_topdown  # type: ignore

        # MMPose expects a fake bbox in full-image space
        h, w = crop.shape[:2]
        fake_bbox = np.array([[0, 0, w, h, 1.0]])
        data_sample = inference_topdown(self._model, crop, fake_bbox)[0]
        kps = data_sample.pred_instances.keypoints[0].astype(np.float32)
        scores = data_sample.pred_instances.keypoint_scores[0].astype(np.float32)
        return kps, scores

    def _infer_onnx(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Minimal ONNX Runtime inference with standard RTMPose pre/post-processing."""
        W, H = self.input_size
        blob = cv2.resize(crop, (W, H)).astype(np.float32)
        # ImageNet normalisation
        blob = (blob / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob.astype(np.float32)})

        # RTMPose ONNX outputs: heatmaps (1, 17, H/4, W/4)
        heatmaps = outputs[0][0]  # (17, H', W')
        kps, scores = self._decode_heatmaps(heatmaps, crop.shape)
        return kps, scores

    # ------------------------------------------------------------------
    # Private: geometry helpers
    # ------------------------------------------------------------------

    def _decode_heatmaps(
        self, heatmaps: np.ndarray, crop_shape: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode RTMPose SimCC or heatmap output to keypoints in crop coords.
        Uses argmax for speed (Jetson constraint).
        """
        num_kp, hh, hw = heatmaps.shape
        crop_h, crop_w = crop_shape[:2]

        keypoints = np.zeros((num_kp, 2), dtype=np.float32)
        scores = np.zeros(num_kp, dtype=np.float32)

        for k in range(num_kp):
            flat_idx = np.argmax(heatmaps[k])
            hy, hx = divmod(int(flat_idx), hw)
            scores[k] = float(heatmaps[k, hy, hx])
            # Scale back to crop pixel coordinates
            keypoints[k, 0] = hx * crop_w / hw
            keypoints[k, 1] = hy * crop_h / hh

        return keypoints, scores

    def _project_to_frame(
        self,
        kps_crop: np.ndarray,
        bbox: Tuple[int, int, int, int],
        crop_shape: Tuple[int, ...],
        frame_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """
        Map keypoints from crop-local coords to full-frame pixel coords.
        Accounts for the padding added by PersonDetector._extract_crop().
        """
        x1, y1, x2, y2 = bbox
        crop_h, crop_w = crop_shape[:2]

        scale_x = (x2 - x1) / crop_w
        scale_y = (y2 - y1) / crop_h

        kps_full = kps_crop.copy()
        kps_full[:, 0] = kps_crop[:, 0] * scale_x + x1
        kps_full[:, 1] = kps_crop[:, 1] * scale_y + y1

        # Clamp to frame boundaries
        kps_full[:, 0] = np.clip(kps_full[:, 0], 0, frame_shape[1] - 1)
        kps_full[:, 1] = np.clip(kps_full[:, 1], 0, frame_shape[0] - 1)

        return kps_full