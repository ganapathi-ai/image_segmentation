"""
Flask web server for the Image Segmentation app.
Uses ONNX Runtime for lightweight CPU inference (~30 MB vs ~300 MB for PyTorch).
"""
import os
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from model_utils import (
    get_session,
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
CORS(app)  # Allow same-origin and cross-origin requests

# ── Load model at startup from repo weights ─────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "final_model.onnx")
_model_load_failed = False

if os.path.exists(WEIGHTS_PATH):
    try:
        load_weights_from_path(WEIGHTS_PATH)
        print(f"Loaded ONNX model from {WEIGHTS_PATH}")
    except Exception as exc:
        _model_load_failed = True
        print(f"Warning: could not load model at startup: {exc}")
else:
    print(f"No weights file found at {WEIGHTS_PATH}")
    _model_load_failed = True


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


@app.route("/api/segment", methods=["POST"])
def segment():
    """Accept an image file, run ONNX inference, return results."""
    if not is_weights_loaded():
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

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
