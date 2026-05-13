
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ImageNet mean/std for RTMPose preprocessing
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# SimCC split ratio used during RTMPose-m export
_SIMCC_SPLIT_RATIO = 2.0


class RTMPoseTRT:
    """
    TensorRT inference wrapper for RTMPose SimCC engine.

    Parameters
    ----------
    engine_path : str | Path
        Path to the .engine file built by trtexec.
    input_size  : tuple (W, H)
        Must match the engine's fixed input shape — (192, 256) for RTMPose-m.
    device_id   : int
        CUDA device index (0 on single-GPU Jetson/desktop).
    """

    # Fixed from trtexec output
    INPUT_W  = 192
    INPUT_H  = 256
    N_KP     = 17
    SIMCC_X  = 384   # 192 * split_ratio
    SIMCC_Y  = 512   # 256 * split_ratio

    def __init__(
        self,
        engine_path: str | Path,
        input_size:  Tuple[int, int] = (192, 256),
        device_id:   int = 0,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.input_w, self.input_h = input_size
        self.device_id   = device_id
        self._context    = None
        self._engine     = None

        # Host/device buffers (allocated once at load time)
        self._h_input:   Optional[np.ndarray] = None
        self._h_out_x:   Optional[np.ndarray] = None
        self._h_out_y:   Optional[np.ndarray] = None
        self._d_input    = None
        self._d_out_x    = None
        self._d_out_y    = None
        self._stream     = None

        # Output names from trtexec
        self._input_name  = "input"
        self._out_x_name  = "output"   # [1, 17, 384] SimCC-X
        self._out_y_name  = "700"      # [1, 17, 512] SimCC-Y

    # ── Lifecycle ─────────────────────────────────────────────────────

    def load(self) -> None:
        """Deserialise the TRT engine and allocate I/O buffers."""
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {self.engine_path}")

        import tensorrt as trt  # type: ignore
        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore  # initialises CUDA context

        logger.info("Loading TRT engine: %s", self.engine_path)
        trt_logger = trt.Logger(trt.Logger.WARNING)

        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"Failed to deserialise TRT engine: {self.engine_path}")

        self._context = self._engine.create_execution_context()
        self._stream  = cuda.Stream()

        # Allocate pinned host buffers + device buffers
        self._h_input  = cuda.pagelocked_empty((1, 3, self.input_h, self.input_w), np.float32)
        self._h_out_x  = cuda.pagelocked_empty((1, self.N_KP, self.SIMCC_X),       np.float32)
        self._h_out_y  = cuda.pagelocked_empty((1, self.N_KP, self.SIMCC_Y),       np.float32)

        self._d_input  = cuda.mem_alloc(self._h_input.nbytes)
        self._d_out_x  = cuda.mem_alloc(self._h_out_x.nbytes)
        self._d_out_y  = cuda.mem_alloc(self._h_out_y.nbytes)

        # Warm-up pass to avoid first-frame latency spike
        dummy = np.zeros((1, 3, self.input_h, self.input_w), dtype=np.float32)
        self._run_trt(dummy)

        logger.info("TRT engine loaded and warmed up — RTMPose SimCC ready.")

    def release(self) -> None:
        """Free GPU resources."""
        self._context = None
        self._engine  = None
        self._stream  = None
        for buf in [self._d_input, self._d_out_x, self._d_out_y]:
            if buf is not None:
                try:
                    buf.free()
                except Exception:
                    pass

    # ── Public inference ──────────────────────────────────────────────

    def infer(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run TRT inference on a single person BGR crop.

        Parameters
        ----------
        crop : np.ndarray
            BGR crop of the person (any size — resized internally).

        Returns
        -------
        keypoints : np.ndarray  (17, 2)  x,y in crop-local pixel coords
        scores    : np.ndarray  (17,)    per-keypoint confidence
        """
        blob = self._preprocess(crop)           # (1, 3, H, W) float32
        self._run_trt(blob)
        kps, scores = self._decode_simcc(
            self._h_out_x[0],    # (17, 384)
            self._h_out_y[0],    # (17, 512)
            crop.shape,
        )
        return kps, scores

    # ── Private: preprocessing ────────────────────────────────────────

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        BGR crop → float32 NCHW tensor, ImageNet normalised.
        """
        img = cv2.resize(crop, (self.input_w, self.input_h),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD                      # HWC float32
        img = img.transpose(2, 0, 1)[np.newaxis]        # 1CHW float32
        return np.ascontiguousarray(img)

    # ── Private: TRT execution ────────────────────────────────────────

    def _run_trt(self, blob: np.ndarray) -> None:
        """
        Copy input H→D, execute engine, copy outputs D→H.
        Uses pycuda async transfers on a single stream for minimal latency.
        """
        import pycuda.driver as cuda  # type: ignore

        np.copyto(self._h_input, blob)
        cuda.memcpy_htod_async(self._d_input, self._h_input, self._stream)

        # Set tensor addresses for TRT 10.x API
        self._context.set_tensor_address(self._input_name, int(self._d_input))
        self._context.set_tensor_address(self._out_x_name, int(self._d_out_x))
        self._context.set_tensor_address(self._out_y_name, int(self._d_out_y))

        self._context.execute_async_v3(stream_handle=self._stream.handle)

        cuda.memcpy_dtoh_async(self._h_out_x, self._d_out_x, self._stream)
        cuda.memcpy_dtoh_async(self._h_out_y, self._d_out_y, self._stream)
        self._stream.synchronize()

    # ── Private: SimCC decoding ───────────────────────────────────────

    def _decode_simcc(
        self,
        simcc_x:    np.ndarray,   # (17, 384)
        simcc_y:    np.ndarray,   # (17, 512)
        crop_shape: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode SimCC 1D logit distributions into keypoint coordinates + scores.

        SimCC represents each axis as a 1D categorical distribution.
        The predicted coordinate is argmax / split_ratio.
        The score is softmax_max(X) * softmax_max(Y).

        Parameters
        ----------
        simcc_x    : (17, W*split) X-axis logits
        simcc_y    : (17, H*split) Y-axis logits
        crop_shape : (H, W, C) of the original crop (before resize)

        Returns
        -------
        keypoints : (17, 2) in crop pixel coordinates
        scores    : (17,)
        """
        crop_h, crop_w = crop_shape[:2]

        # Softmax for score computation
        def _softmax_max(logits: np.ndarray) -> np.ndarray:
            e = np.exp(logits - logits.max(axis=-1, keepdims=True))
            return (e / e.sum(axis=-1, keepdims=True)).max(axis=-1)

        # Argmax coordinate
        x_idx   = np.argmax(simcc_x, axis=-1).astype(np.float32)  # (17,)
        y_idx   = np.argmax(simcc_y, axis=-1).astype(np.float32)  # (17,)

        # Convert from bin index to crop-local pixel coordinate
        # bin → model_pixel = idx / split_ratio
        # model_pixel → crop_pixel = model_pixel * (crop_size / model_input_size)
        x_model = x_idx / _SIMCC_SPLIT_RATIO                       # 0..192
        y_model = y_idx / _SIMCC_SPLIT_RATIO                       # 0..256
        x_crop  = x_model * (crop_w / self.input_w)
        y_crop  = y_model * (crop_h / self.input_h)

        keypoints = np.stack([x_crop, y_crop], axis=-1)            # (17, 2)

        # Score = product of per-axis softmax maxima
        score_x = _softmax_max(simcc_x)
        score_y = _softmax_max(simcc_y)
        scores  = (score_x * score_y).astype(np.float32)

        return keypoints, scores