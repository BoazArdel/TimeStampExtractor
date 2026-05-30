"""
Detect the region of an image that contains the red timestamp.

Strategy:
    The original v1 hardcoded a crop at (78%-92% width, 83%-94% height). That fails
    when the scanner orientation changes or the timestamp sits in a different corner.

    Instead, we:
      1. Convert to HSV and build a mask of "red" pixels (red wraps around hue 0).
      2. Apply a small morphological close so adjacent digit pixels become one blob.
      3. Find connected components; filter out tiny noise and giant artifacts.
      4. Score each surviving component by:
            - distance from the nearest corner (timestamps are near a corner)
            - aspect ratio (timestamps are wider than tall)
            - pixel density of red within the bbox
      5. Return the highest-scoring bbox plus a small padding margin.

    If nothing scores above threshold, fall back to the legacy v1 region so the
    pipeline still has *something* to OCR.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Region:
    """Bounding box of the detected timestamp area in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    confidence: float  # 0.0 to 1.0, how sure we are this is a timestamp
    source: str  # "detected" / "fallback" / "user" / "full"

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y : self.y + self.h, self.x : self.x + self.w]


# Named-corner shortcuts. Tuples are (x0%, y0%, x1%, y1%) of image dimensions.
# These are slightly generous so we don't slice through a digit.
NAMED_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "bottom-right": (0.70, 0.80, 0.99, 0.99),
    "bottom-left": (0.01, 0.80, 0.30, 0.99),
    "top-right": (0.70, 0.01, 0.99, 0.20),
    "top-left": (0.01, 0.01, 0.30, 0.20),
    "bottom": (0.01, 0.80, 0.99, 0.99),  # whole bottom strip
    "top": (0.01, 0.01, 0.99, 0.20),
    "full": (0.0, 0.0, 1.0, 1.0),
}


def parse_region_spec(spec: str) -> tuple[float, float, float, float]:
    """
    Parse a user-supplied region spec into a (x0%, y0%, x1%, y1%) tuple.

    Accepted forms:
        "bottom-right"               -> NAMED_REGIONS["bottom-right"]
        "78,83,92,94"                -> (0.78, 0.83, 0.92, 0.94)  (percentages 0-100)
        "0.78,0.83,0.92,0.94"        -> taken as fractions if any value <= 1.0
    """
    key = spec.strip().lower()
    if key in NAMED_REGIONS:
        return NAMED_REGIONS[key]

    parts = [p.strip() for p in key.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Region spec {spec!r} must be a named corner "
            f"({', '.join(NAMED_REGIONS)}) or four comma-separated numbers."
        )
    try:
        nums = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"Region spec {spec!r}: non-numeric value.") from e

    # If any value > 1, assume the user supplied percentages (0-100).
    if max(nums) > 1.0:
        nums = [n / 100.0 for n in nums]

    x0, y0, x1, y1 = nums
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(
            f"Region spec {spec!r}: bounds must satisfy 0 ≤ x0 < x1 ≤ 1 and 0 ≤ y0 < y1 ≤ 1."
        )
    return x0, y0, x1, y1


def fixed_region(img: np.ndarray, spec: str) -> Region:
    """Build a Region from a user spec, in pixel coordinates for the given image."""
    x0p, y0p, x1p, y1p = parse_region_spec(spec)
    img_h, img_w = img.shape[:2]
    x = int(img_w * x0p)
    y = int(img_h * y0p)
    w = int(img_w * (x1p - x0p))
    h = int(img_h * (y1p - y0p))
    return Region(x=x, y=y, w=w, h=h, confidence=1.0, source="user")


# HSV ranges for "red" — red wraps around the hue axis so we need two ranges
_RED_LOWER_1 = np.array([0, 70, 50])
_RED_UPPER_1 = np.array([10, 255, 255])
_RED_LOWER_2 = np.array([170, 70, 50])
_RED_UPPER_2 = np.array([180, 255, 255])


def red_mask(bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 mask where red pixels are 255, else 0."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, _RED_LOWER_1, _RED_UPPER_1)
    m2 = cv2.inRange(hsv, _RED_LOWER_2, _RED_UPPER_2)
    return cv2.bitwise_or(m1, m2)


def _distance_to_nearest_corner(
    cx: float, cy: float, img_w: int, img_h: int
) -> float:
    corners = [(0, 0), (img_w, 0), (0, img_h), (img_w, img_h)]
    return min(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 for x, y in corners)


def detect_timestamp_region(
    bgr: np.ndarray,
    *,
    pad_ratio: float = 0.01,
    min_confidence: float = 0.25,
) -> Region:
    """Return the best candidate region or a fallback Region if detection fails."""
    img_h, img_w = bgr.shape[:2]
    mask = red_mask(bgr)

    # Close small gaps between digits so a date like "'97 8 14" becomes one blob.
    # Kernel scaled by image size so it works on both 1MP and 24MP scans.
    kernel_size = max(3, min(img_w, img_h) // 200)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size * 5, kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    best_score = -1.0
    best_region: Region | None = None
    img_diag = (img_w**2 + img_h**2) ** 0.5

    for label in range(1, num_labels):  # 0 is background
        x, y, w, h, area = stats[label]
        if area < (img_w * img_h) * 0.00005:  # too tiny — noise
            continue
        if area > (img_w * img_h) * 0.05:  # too big — probably not a timestamp
            continue
        if h < 5 or w < 10:
            continue

        aspect = w / h
        if not (1.5 < aspect < 12):  # timestamps are wider than tall but not lines
            continue

        # Score components
        cx, cy = x + w / 2, y + h / 2
        corner_dist = _distance_to_nearest_corner(cx, cy, img_w, img_h)
        corner_score = 1.0 - min(1.0, corner_dist / (img_diag * 0.3))
        # Penalize being too close to the center
        center_dist = ((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2) ** 0.5
        center_score = min(1.0, center_dist / (img_diag * 0.4))
        aspect_score = 1.0 if 2.5 <= aspect <= 7 else 0.5
        density = area / (w * h) if w * h > 0 else 0
        density_score = min(1.0, density * 2)

        score = (
            0.4 * corner_score
            + 0.2 * center_score
            + 0.2 * aspect_score
            + 0.2 * density_score
        )
        if score > best_score:
            best_score = score
            best_region = Region(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                confidence=float(score),
                source="detected",
            )

    if best_region is None or best_region.confidence < min_confidence:
        # Fallback to the v1 hardcoded region (bottom-right)
        return Region(
            x=int(img_w * 0.78),
            y=int(img_h * 0.83),
            w=int(img_w * 0.14),
            h=int(img_h * 0.11),
            confidence=0.0,
            source="fallback",
        )

    # Add padding so we don't slice through a digit
    pad_x = int(img_w * pad_ratio)
    pad_y = int(img_h * pad_ratio)
    return Region(
        x=max(0, best_region.x - pad_x),
        y=max(0, best_region.y - pad_y),
        w=min(img_w - max(0, best_region.x - pad_x), best_region.w + 2 * pad_x),
        h=min(img_h - max(0, best_region.y - pad_y), best_region.h + 2 * pad_y),
        confidence=best_region.confidence,
        source=best_region.source,
    )
