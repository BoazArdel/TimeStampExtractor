"""
Write a parsed datetime into a JPEG's EXIF tags AND update file mtime.

EXIF tags we set:
    DateTimeOriginal       (0x9003) — when the photo was taken
    DateTimeDigitized      (0x9004) — when the photo was digitised (we use same)
    DateTime               (0x0132) — last modified

Format per EXIF spec: "YYYY:MM:DD HH:MM:SS"

For non-JPEG inputs (PNG, TIFF, etc.) we still set the file mtime so the OS sees
the date, and skip EXIF if the format doesn't support it.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


_EXIF_DATETIME_FMT = "%Y:%m:%d %H:%M:%S"


def write_exif_datetime(path: Path, dt: datetime) -> bool:
    """
    Write the datetime into JPEG EXIF tags. Returns True on success.

    Falls back to setting only mtime for non-JPEG formats.
    """
    suffix = path.suffix.lower()
    formatted = dt.strftime(_EXIF_DATETIME_FMT)

    if suffix in {".jpg", ".jpeg"}:
        try:
            import piexif
        except ImportError as e:
            raise RuntimeError(
                "piexif not installed. Run: pip install piexif"
            ) from e

        try:
            exif_dict = piexif.load(str(path))
        except Exception:
            # File may have no existing EXIF — start a blank dict
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # ImageDateTime (0x0132) lives in IFD0 ("0th"); the originals in Exif IFD
        exif_dict.setdefault("0th", {})[piexif.ImageIFD.DateTime] = formatted.encode("ascii")
        exif_dict.setdefault("Exif", {})[piexif.ExifIFD.DateTimeOriginal] = formatted.encode("ascii")
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = formatted.encode("ascii")

        try:
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(path))
        except Exception as e:
            logger.exception("Failed to write EXIF to %s: %s", path, e)
            return False
    else:
        logger.info(
            "File %s is not a JPEG (%s) — EXIF will be skipped, only mtime is set.",
            path.name,
            suffix,
        )

    # Always also set the file's mtime so the OS/file browser shows the date
    ts = dt.timestamp()
    try:
        os.utime(path, (ts, ts))
    except OSError as e:
        logger.warning("Could not update mtime on %s: %s", path, e)
        return False

    return True
