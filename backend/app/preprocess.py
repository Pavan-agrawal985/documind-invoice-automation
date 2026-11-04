"""Image cleanup before OCR. Kept small and fast on purpose."""
import cv2
import numpy as np


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Grayscale -> upscale small images -> light denoise -> adaptive threshold."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # OCR works best when text is roughly 30+ px tall; upscale small scans.
    longest = max(gray.shape)
    if longest < 1600:
        scale = 1600 / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
