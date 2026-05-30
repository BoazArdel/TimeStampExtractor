"""
Preprocess a cropped timestamp region into a clean black-on-white image for OCR.

This replaces v1's slow per-pixel Python loop with vectorised NumPy/OpenCV ops.
"""

from __future__ import annotations

import cv2
import numpy as np

from .region_detector import red_mask


def isolate_red_text(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Return a binarised image: black digits on white background.

    Steps:
      1. Build a red mask (vectorised).
      2. Slightly dilate so thin digit strokes don't fragment under OCR.
      3. Invert so digits are black and background white (Tesseract prefers this).
      4. Upscale to ~3x — small timestamps OCR poorly at native size.
    """
    mask = red_mask(bgr_crop)

    # Dilate slightly to thicken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thick = cv2.dilate(mask, kernel, iterations=1)

    # Invert: digits black (0), background white (255)
    inverted = cv2.bitwise_not(thick)

    # Upscale for better OCR accuracy. Tesseract works best when characters are
    # roughly 30-40px tall.
    h = inverted.shape[0]
    if h < 60:
        scale = 60 / h
        inverted = cv2.resize(
            inverted,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    # A median blur cleans up speckle without destroying digit shapes
    inverted = cv2.medianBlur(inverted, 3)
    return inverted


def add_border(img: np.ndarray, px: int = 20) -> np.ndarray:
    """Add a white border around the image. Tesseract often misreads text right at the edge."""
    return cv2.copyMakeBorder(
        img, px, px, px, px, cv2.BORDER_CONSTANT, value=255
    )
