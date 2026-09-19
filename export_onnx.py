"""
One-time script: export final_model.pth to ONNX format.
Run locally:  python export_onnx.py
Then commit weights/final_model.onnx to the repo.
"""
import os
import torch
import segmentation_models_pytorch as smp

WEIGHTS_PTH = os.path.join("weights", "final_model.pth")
WEIGHTS_ONNX = os.path.join("weights", "final_model.onnx")

if not os.path.exists(WEIGHTS_PTH):
    print(f"ERROR: {WEIGHTS_PTH} not found")
    exit(1)

print("Loading PyTorch model...")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
).to("cpu")
model.eval()

state_dict = torch.load(WEIGHTS_PTH, map_location="cpu", weights_only=False)
model.load_state_dict(state_dict)

print("Exporting to ONNX...")
dummy = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy,
    WEIGHTS_ONNX,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                  "output": {0: "batch", 2: "height", 3: "width"}},
    opset_version=17,
)

pth_size = os.path.getsize(WEIGHTS_PTH) / (1024 * 1024)
onnx_size = os.path.getsize(WEIGHTS_ONNX) / (1024 * 1024)
print(f"Done: {WEIGHTS_PTH} ({pth_size:.1f} MB) -> {WEIGHTS_ONNX} ({onnx_size:.1f} MB)")
print(f"Now commit {WEIGHTS_ONNX} and push to deploy.")
