"""
Two OCR backends:

  1. Tesseract — fast, free, runs locally. Configured for digits-only with a
     tight whitelist and PSM 7 (single text line).
  2. Claude vision — sent ONLY when Tesseract output fails to parse as a date
     or is below a confidence threshold. This keeps API costs minimal.

The AI fallback is the key upgrade over v1, which had no graceful failure mode.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OcrResult:
    text: str
    backend: str  # "tesseract" or "claude"
    confidence: float


# ---- Tesseract ----------------------------------------------------------

_TESSERACT_CONFIG = (
    "--psm 7 "  # treat the image as a single text line
    "-c tessedit_char_whitelist=0123456789 '-/.: "
)


def ocr_tesseract(img: np.ndarray) -> OcrResult:
    """Run Tesseract on a preprocessed (binarised) image."""
    import pytesseract

    # If a custom binary path is set, respect it
    bin_path = os.getenv("TESSERACT_CMD")
    if bin_path:
        pytesseract.pytesseract.tesseract_cmd = bin_path

    try:
        data = pytesseract.image_to_data(
            img,
            config=_TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract binary not found. Install it (apt: tesseract-ocr, "
            "brew: tesseract, win: UB Mannheim build) or set TESSERACT_CMD."
        ) from e

    words = [w for w in data["text"] if w.strip()]
    confs = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
    avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    text = " ".join(words)
    return OcrResult(text=text, backend="tesseract", confidence=avg_conf)


# ---- Claude vision ------------------------------------------------------

_CLAUDE_PROMPT = (
    "This is a small cropped image showing a date stamp imprinted by a film camera. "
    "The digits are typically red/orange and may include a 2- or 4-digit year, month, "
    "day, and sometimes hours and minutes. Camera date stamps often look like "
    "\"'97 8 14\" or \"97 8 14\" or \"8 14 '97\" or \"1997-08-14 16:42\".\n\n"
    "Read the date and time visible in the image. Respond with ONLY one of:\n"
    "  - The date in ISO 8601 format: YYYY-MM-DD or YYYY-MM-DDTHH:MM\n"
    "  - The literal string UNREADABLE if you cannot make it out\n"
    "Do not add any other text, explanation, or markdown."
)


def _encode_image(img: np.ndarray) -> tuple[str, str]:
    """Encode a numpy image as base64 PNG."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG for Claude API.")
    return base64.b64encode(buf.tobytes()).decode("ascii"), "image/png"


def ocr_claude(
    img: np.ndarray,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> OcrResult:
    """Send the cropped region to Claude and ask for the date."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from e

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file or environment."
        )

    model = model or os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
    client = anthropic.Anthropic(api_key=api_key)
    b64, media_type = _encode_image(img)

    resp = client.messages.create(
        model=model,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": _CLAUDE_PROMPT},
                ],
            }
        ],
    )

    text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    text = "".join(text_parts).strip()
    if text == "UNREADABLE" or not text:
        return OcrResult(text="", backend="claude", confidence=0.0)
    return OcrResult(text=text, backend="claude", confidence=0.9)
