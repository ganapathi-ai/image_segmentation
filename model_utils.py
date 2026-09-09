"""
Model utilities for U-Net image segmentation.
Mirrors the preprocessing/postprocessing from the Colab notebook exactly.
"""
import io
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

# ImageNet normalization (same as notebook)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 256

# Singleton pattern — load model once, reuse across requests
_model = None
_weights_loaded = False


def get_model():
    """Lazy-load and return the U-Net model with ResNet34 encoder."""
    global _model
    if _model is None:
        _model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to("cpu")
        _model.eval()
    return _model


def load_weights(weights_bytes: bytes) -> None:
    """Load trained weights from raw bytes into the model."""
    global _weights_loaded
    model = get_model()
    state_dict = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    _weights_loaded = True


def is_weights_loaded() -> bool:
    return _weights_loaded


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Decode image from bytes, resize to IMAGE_SIZE×IMAGE_SIZE,
    normalize with ImageNet mean/std, return (1, 3, H, W) float tensor.
    """
    # Decode image
    img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    # Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD

    # HWC -> CHW -> (1, C, H, W)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).float()
    return tensor


def run_inference(model, tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Run forward pass and return:
      - pred_sigmoid: float mask in [0, 1], shape (H, W)
      - pred_binary:  uint8 binary mask {0, 255}, shape (H, W)
    """
    with torch.no_grad():
        output = model(tensor)
        pred_sigmoid = torch.sigmoid(output).squeeze().cpu().numpy()
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


def encode_png(img_bgr_or_rgb: np.ndarray) -> str:
    """Encode an RGB or BGR numpy array as a base64 PNG string."""
    _, buf = cv2.imencode(".png", cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR))
    import base64
    return base64.b64encode(buf).decode("utf-8")


def encode_jpeg(img_rgb: np.ndarray, quality: int = 90) -> str:
    """Encode an RGB numpy array as a base64 JPEG string (smaller than PNG)."""
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
