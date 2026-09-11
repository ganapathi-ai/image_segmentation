"""
Flask web server for the Image Segmentation app.
Routes:
  GET  /            → serve the single-page frontend
  GET  /api/health  → health check
  POST /api/upload-weights  → accept .pth file, store in memory
  POST /api/segment         → accept image, return segmentation results
"""
import gc
import os
from flask import Flask, request, jsonify, render_template

from model_utils import (
    get_model,
    load_weights,
    is_weights_loaded,
    preprocess_image,
    run_inference,
    decode_original_image,
    mask_to_overlay,
    encode_jpeg,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload

# ── Auto-load weights on startup ───────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "final_model.pth")
_weights_preloaded = False

if os.path.exists(WEIGHTS_PATH):
    try:
        with open(WEIGHTS_PATH, "rb") as f:
            load_weights(f.read())
        _weights_preloaded = True
        print(f"Loaded weights from {WEIGHTS_PATH}")
    except Exception as exc:
        print(f"Warning: could not pre-load weights: {exc}")


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
        "weights_preloaded": _weights_preloaded,
    })


@app.route("/api/upload-weights", methods=["POST"])
def upload_weights():
    """Accept a .pth file and load it into the model."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".pth"):
        return jsonify({"error": "Please upload a .pth file"}), 400

    try:
        weights_bytes = file.read()
        load_weights(weights_bytes)
        return jsonify({"message": "Weights loaded successfully"})
    except Exception as exc:
        return jsonify({"error": f"Failed to load weights: {exc}"}), 500


@app.route("/api/segment", methods=["POST"])
def segment():
    """Accept an image file, run inference, return results as base64 images + metrics."""
    # Check weights
    if not is_weights_loaded():
        return jsonify({"error": "Model weights not loaded. Upload .pth first."}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()

        # Decode original at full resolution for display
        original_rgb = decode_original_image(image_bytes)
        h_orig, w_orig = original_rgb.shape[:2]

        # Preprocess for model (256×256)
        tensor = preprocess_image(image_bytes)

        # Inference
        model = get_model()
        pred_sigmoid, pred_binary = run_inference(model, tensor)

        # Resize mask back to original image size
        pred_binary_full = cv2.resize(pred_binary, (w_orig, h_orig),
                                      interpolation=cv2.INTER_NEAREST)
        pred_sigmoid_full = cv2.resize(pred_sigmoid, (w_orig, h_orig),
                                       interpolation=cv2.INTER_LINEAR)

        # Overlay
        overlay_rgb = mask_to_overlay(original_rgb, pred_binary_full)

        # Encode results as JPEG base64
        original_b64 = encode_jpeg(original_rgb)
        mask_b64 = encode_jpeg(
            cv2.cvtColor(pred_binary_full, cv2.COLOR_GRAY2RGB)
        )
        overlay_b64 = encode_jpeg(overlay_rgb)

        # Compute pixel coverage
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
    finally:
        # Free per-request memory on the 512 MB free tier
        gc.collect()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
