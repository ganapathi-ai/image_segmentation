"""
Flask web server for the Image Segmentation app.
Uses ONNX Runtime for lightweight CPU inference (~30 MB vs ~300 MB for PyTorch).
Routes:
  GET  /            → serve the single-page frontend
  GET  /api/health  → health check
  POST /api/upload-weights  → accept .onnx or .pth file
  POST /api/segment         → accept image, return segmentation results
"""
import os
from flask import Flask, request, jsonify, render_template

from model_utils import (
    get_session,
    load_weights,
    load_weights_from_path,
    is_weights_loaded,
    preprocess_image,
    run_inference,
    decode_original_image,
    mask_to_overlay,
    encode_jpeg,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload

# ── Lazy-load model on first inference request ──────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "final_model.onnx")
_model_load_failed = False


def _ensure_model_loaded():
    """Load ONNX model from disk on first inference call."""
    global _model_load_failed
    if _model_load_failed:
        return
    onnx_path = WEIGHTS_PATH
    # Fallback: if .onnx not found, check for .pth and tell user
    if not os.path.exists(onnx_path):
        pth_path = WEIGHTS_PATH.replace(".onnx", ".pth")
        if os.path.exists(pth_path):
            # Auto-convert .pth to .onnx using torch (only on startup)
            _convert_pth_to_onnx(pth_path, onnx_path)
        else:
            _model_load_failed = True
            return
    try:
        load_weights_from_path(onnx_path)
        print(f"Loaded ONNX model from {onnx_path}")
    except Exception as exc:
        _model_load_failed = True
        print(f"Warning: could not load model: {exc}")


def _convert_pth_to_onnx(pth_path: str, onnx_path: str) -> None:
    """Convert .pth to .onnx on first deploy if .onnx not yet committed."""
    import torch
    import segmentation_models_pytorch as smp

    print(f"Converting {pth_path} to ONNX (one-time)...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to("cpu")
    model.eval()

    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)

    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
        opset_version=18,
    )
    print(f"ONNX export complete: {onnx_path}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/status")
def status():
    """Return whether model weights are loaded."""
    return jsonify({
        "weights_loaded": is_weights_loaded(),
        "runtime": "onnxruntime",
    })


@app.route("/api/upload-weights", methods=["POST"])
def upload_weights():
    """Accept a .pth or .onnx file and load it into memory."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("pth", "onnx"):
        return jsonify({"error": "Please upload a .pth or .onnx file"}), 400

    try:
        weights_bytes = file.read()

        # If .onnx, save directly. If .pth, save as-is for potential conversion.
        global _model_load_failed
        if ext == "onnx":
            load_weights(weights_bytes)
            _model_load_failed = False
            return jsonify({"message": "ONNX model loaded successfully"})
        else:
            # Save .pth — will be converted on first inference
            pth_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "weights", "final_model.pth"
            )
            os.makedirs(os.path.dirname(pth_path), exist_ok=True)
            with open(pth_path, "wb") as f:
                f.write(weights_bytes)
            _model_load_failed = False
            return jsonify({"message": "PyTorch weights saved — will convert to ONNX on first inference"})

    except Exception as exc:
        return jsonify({"error": f"Failed to load weights: {exc}"}), 500


@app.route("/api/segment", methods=["POST"])
def segment():
    """Accept an image file, run ONNX inference, return results."""
    # Lazy-load model on first inference request
    _ensure_model_loaded()

    if not is_weights_loaded():
        return jsonify({"error": "Model weights not loaded. Upload .pth or .onnx first."}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()

        original_rgb = decode_original_image(image_bytes)
        h_orig, w_orig = original_rgb.shape[:2]

        tensor = preprocess_image(image_bytes)
        session = get_session()
        pred_sigmoid, pred_binary = run_inference(session, tensor)

        # Resize mask back to original image size
        pred_binary_full = cv2.resize(pred_binary, (w_orig, h_orig),
                                      interpolation=cv2.INTER_NEAREST)

        overlay_rgb = mask_to_overlay(original_rgb, pred_binary_full)

        original_b64 = encode_jpeg(original_rgb)
        mask_b64 = encode_jpeg(cv2.cvtColor(pred_binary_full, cv2.COLOR_GRAY2RGB))
        overlay_b64 = encode_jpeg(overlay_rgb)

        coverage_pct = float(pred_binary_full.sum() / 255 / (h_orig * w_orig) * 100)

        return jsonify({
            "original": original_b64,
            "mask": mask_b64,
            "overlay": overlay_b64,
            "coverage_pct": round(coverage_pct, 2),
            "image_size": {"width": w_orig, "height": h_orig},
        })

    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
