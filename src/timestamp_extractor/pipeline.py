"""
End-to-end pipeline for a single image file.

Flow:
    load → detect region → preprocess → tesseract OCR → parse
        ↓ (if parse fails or confidence low)
        claude vision → parse
        ↓ (if still no date)
        give up, record failure
    → write EXIF + mtime (unless --dry-run)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .date_parser import ParsedDate, parse_date
from .exif_writer import write_exif_datetime
from .ocr import ocr_claude, ocr_tesseract
from .preprocessing import add_border, isolate_red_text
from .region_detector import Region, detect_timestamp_region, fixed_region

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    path: Path
    success: bool
    parsed_date: ParsedDate | None = None
    region: Region | None = None
    backend: str | None = None  # "tesseract" / "claude" / None
    error: str | None = None
    debug_crops: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    use_ai_fallback: bool = True
    ai_only: bool = False  # Skip Tesseract entirely and go straight to Claude
    region_spec: str | None = None  # User-fixed region; skips auto-detection
    tesseract_min_confidence: float = 0.35
    region_min_confidence: float = 0.25
    min_year: int = 1950
    max_year: int = field(default_factory=lambda: datetime.now().year)
    dry_run: bool = False
    save_debug: bool = False
    debug_dir: Path | None = None

    def __post_init__(self) -> None:
        # ai_only implies AI is in play; force-enable the fallback path so it actually runs
        if self.ai_only:
            self.use_ai_fallback = True


def process_image(path: Path, config: PipelineConfig) -> ProcessResult:
    """Run the full extraction pipeline on a single image file."""
    result = ProcessResult(path=path, success=False)

    bgr = cv2.imread(str(path))
    if bgr is None:
        result.error = "Could not read image (corrupt or unsupported format)"
        return result

    # --- Resolve region: user-fixed OR auto-detected --------------------
    if config.region_spec:
        region = fixed_region(bgr, config.region_spec)
        logger.debug(
            "%s: user-fixed region=%s (spec=%r)",
            path.name, (region.x, region.y, region.w, region.h), config.region_spec,
        )
    else:
        region = detect_timestamp_region(bgr, min_confidence=config.region_min_confidence)
        logger.debug(
            "%s: region=%s conf=%.2f source=%s",
            path.name, (region.x, region.y, region.w, region.h),
            region.confidence, region.source,
        )
    result.region = region

    crop = region.crop(bgr)
    if crop.size == 0:
        result.error = "Region was empty (check --region spec)"
        return result

    # --- Preprocess (skip Tesseract prep work entirely when AI-only) ----
    parsed: ParsedDate | None = None
    backend_used: str | None = None

    if not config.ai_only:
        binary = isolate_red_text(crop)
        bordered = add_border(binary, px=20)
        if config.save_debug:
            result.debug_crops["00_crop"] = crop
            result.debug_crops["01_binary"] = binary
            result.debug_crops["02_bordered"] = bordered

        # --- Tesseract OCR ----------------------------------------------
        try:
            tess = ocr_tesseract(bordered)
            logger.debug(
                "%s: tesseract='%s' conf=%.2f", path.name, tess.text, tess.confidence
            )
            if tess.confidence >= config.tesseract_min_confidence:
                parsed = parse_date(
                    tess.text,
                    min_year=config.min_year,
                    max_year=config.max_year,
                )
                if parsed is not None:
                    backend_used = "tesseract"
        except RuntimeError as e:
            logger.warning("Tesseract unavailable: %s", e)
    else:
        # In ai-only mode we still save the crop for debug, but skip binarisation
        if config.save_debug:
            result.debug_crops["00_crop"] = crop

    # --- Claude vision: fallback OR primary (when ai_only) --------------
    if parsed is None and config.use_ai_fallback:
        try:
            cl = ocr_claude(crop)
            logger.debug("%s: claude='%s'", path.name, cl.text)
            if cl.text:
                parsed = parse_date(
                    cl.text,
                    min_year=config.min_year,
                    max_year=config.max_year,
                )
                if parsed is not None:
                    backend_used = "claude"
        except RuntimeError as e:
            logger.warning("Claude call failed: %s", e)

    if parsed is None:
        result.error = "Could not extract a valid date from the image"
        return result

    result.parsed_date = parsed
    result.backend = backend_used
    result.success = True

    # --- Write EXIF ------------------------------------------------------
    if not config.dry_run:
        ok = write_exif_datetime(path, parsed.dt)
        if not ok:
            result.success = False
            result.error = "Date parsed but EXIF write failed"

    return result
