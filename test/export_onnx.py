from mmpose.apis import init_model
from mmengine.registry import init_default_scope

import torch

# ==========================================================
# INIT SCOPE
# ==========================================================

init_default_scope('mmpose')

# ==========================================================
# CONFIG + CHECKPOINT
# ==========================================================
CONFIG = "/media/algosium/SSD/vivek/mandown_detection/test/rtmpose-m_8xb256-420e_coco-256x192.py"

CHECKPOINT = "/media/algosium/SSD/vivek/mandown_detection/models/pose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"

# ==========================================================
# LOAD MODEL
# ==========================================================

model = init_model(
    CONFIG,
    CHECKPOINT,
    device='cpu'
)

model.eval()

# ==========================================================
# DUMMY INPUT
# ==========================================================

dummy = torch.randn(1, 3, 256, 192)

# ==========================================================
# EXPORT
# ==========================================================

torch.onnx.export(
    model,
    dummy,
    "rtmpose-m.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11,
    dynamic_axes={
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    }
)

print("✅ ONNX export completed")