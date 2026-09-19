"""
Model utilities for U-Net image segmentation.
Uses ONNX Runtime for lightweight CPU inference (~30 MB vs ~300 MB for PyTorch).
Preprocessing/postprocessing mirrors the Colab notebook exactly.
"""
import io
import cv2
import numpy as np
import onnxruntime as ort

# ImageNet normalization (same as notebook)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 256

# Singleton — load ONNX session once
_session = None
_weights_loaded = False


def get_session():
    """Lazy-load and return the ONNX Runtime inference session."""
    global _session
    if _session is None:
        model_path = _get_model_path()
        _session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    return _session


def _get_model_path() -> str:
    """Return path to the ONNX model file."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "weights", "final_model.onnx")


def load_weights(weights_bytes: bytes) -> None:
    """Load ONNX model from raw bytes into memory."""
    global _weights_loaded, _session
    path = _get_model_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(weights_bytes)
    # Force session reload on next request
    _session = None
    _weights_loaded = True
    # Free the bytes buffer
    del weights_bytes
    import gc
    gc.collect()


def load_weights_from_path(path: str) -> None:
    """Load ONNX model directly from a file path on disk."""
    global _weights_loaded, _session
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    # Validate it's a real ONNX file
    with open(path, "rb") as f:
        header = f.read(8)
    if header[:4] != b"ONNX":
        raise ValueError(f"Not a valid ONNX file: {path}")
    _weights_loaded = True
    _session = None  # Force reload on next get_session()


def is_weights_loaded() -> bool:
    global _weights_loaded
    if not _weights_loaded:
        # Check if model file exists on disk
        import os
        if os.path.exists(_get_model_path()):
            _weights_loaded = True
    return _weights_loaded


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image from bytes, resize to IMAGE_SIZE×IMAGE_SIZE,
    normalize with ImageNet mean/std, return (1, 3, H, W) float32 array.
    """
    img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD

    # HWC -> CHW -> (1, C, H, W)
    img = np.transpose(img, (2, 0, 1))
    return img[np.newaxis, ...].astype(np.float32)


def run_inference(session, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run forward pass and return:
      - pred_sigmoid: float mask in [0, 1], shape (H, W)
      - pred_binary:  uint8 binary mask {0, 255}, shape (H, W)
    """
    output = session.run(None, {"input": tensor})[0]
    logits = output[0, 0]  # (H, W) — raw logits
    pred_sigmoid = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    pred_binary = (pred_sigmoid > 0.5).astype(np.uint8) * 255
    return pred_sigmoid, pred_binary


def mask_to_overlay(original_rgb: np.ndarray, mask_255: np.ndarray,
                    alpha: float = 0.35) -> np.ndarray:
    """Overlay a red mask on the original RGB image."""
    overlay = original_rgb.copy().astype(np.float32) / 255.0
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    mask_bool = mask_255 > 127
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * red
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def encode_jpeg(img_rgb: np.ndarray, quality: int = 90) -> str:
    """Encode an RGB numpy array as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                          [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    import base64
    return base64.b64encode(buf).decode("utf-8")


def decode_original_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image to RGB numpy array (original size, for display)."""
    img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
