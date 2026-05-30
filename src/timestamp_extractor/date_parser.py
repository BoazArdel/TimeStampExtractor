"""
Parse OCR output (or Claude vision output) into a real datetime.

Old film cameras imprinted dates in many styles depending on brand/year/region:
    "'97 8 14"          (apostrophe year + month + day, Olympus/Canon)
    "97 8 14"
    "1997 8 14"
    "8 14 '97"          (US order)
    "14 8 '97"          (European order)
    "8.14.97" / "8/14/97"
    "97.08.14"
    "14-08-1997"
    "1997-08-14"

Some imprint just the date, some also imprint hh:mm. The user's input may also
contain stray OCR garbage like 'S' for '5' or 'O' for '0'.

Strategy: try several regex patterns from most-specific to most-permissive, then
sanity-check the parsed date against a configurable plausible range.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class ParsedDate:
    dt: datetime
    raw: str
    pattern: str  # name of the pattern that matched
    confidence: float  # 0-1, our confidence the parse is correct


# Map common OCR confusions back to digits
_OCR_FIXUPS = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "|": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8",
    "g": "9", "q": "9",
})


def _normalize(raw: str) -> str:
    """Lightly clean OCR output before regex matching."""
    s = raw.strip()
    # Normalise separators
    s = s.replace("'", " ").replace(".", " ").replace("/", " ")
    s = s.replace("-", " ").replace(":", " ").replace(",", " ")
    # OCR fixups, but only if the surrounding chars are digit-like — keep simple
    s = s.translate(_OCR_FIXUPS)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Ordered: more-specific first
# Each tuple: (regex, year_group, month_group, day_group, name, base_confidence)
_PATTERNS: list[tuple[re.Pattern, str, str, str, str, float]] = [
    # 1997 08 14  /  1997-08-14
    (
        re.compile(r"\b(?P<y>(?:19|20)\d{2})\s+(?P<m>\d{1,2})\s+(?P<d>\d{1,2})\b"),
        "y", "m", "d", "YYYY MM DD", 0.95,
    ),
    # 14 08 1997  (European)
    (
        re.compile(r"\b(?P<d>\d{1,2})\s+(?P<m>\d{1,2})\s+(?P<y>(?:19|20)\d{2})\b"),
        "y", "m", "d", "DD MM YYYY", 0.85,
    ),
    # '97 8 14  /  97 8 14   (apostrophe-year, common on film cameras)
    (
        re.compile(r"\b(?P<y>\d{2})\s+(?P<m>\d{1,2})\s+(?P<d>\d{1,2})\b"),
        "y", "m", "d", "YY MM DD", 0.80,
    ),
    # 8 14 97  (US order, 2-digit year)
    (
        re.compile(r"\b(?P<m>\d{1,2})\s+(?P<d>\d{1,2})\s+(?P<y>\d{2})\b"),
        "y", "m", "d", "MM DD YY", 0.70,
    ),
    # 14 8 97  (European, 2-digit year) — same shape as MM DD YY; disambiguated by validation
    (
        re.compile(r"\b(?P<d>\d{1,2})\s+(?P<m>\d{1,2})\s+(?P<y>\d{2})\b"),
        "y", "m", "d", "DD MM YY", 0.65,
    ),
]


def _expand_year(y: int) -> int:
    """Two-digit years: <50 → 20xx, >=50 → 19xx (typical photo-era assumption)."""
    if y >= 100:
        return y
    return 2000 + y if y < 50 else 1900 + y


def _try_with_time(s: str, base_dt: datetime) -> datetime:
    """If the OCR'd string has a trailing hh mm time, attach it."""
    m = re.search(r"\b(?P<h>\d{1,2})\s+(?P<min>\d{2})\b\s*$", s)
    if not m:
        return base_dt
    try:
        h, mn = int(m.group("h")), int(m.group("min"))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return base_dt.replace(hour=h, minute=mn)
    except ValueError:
        pass
    return base_dt


def parse_date(
    raw: str,
    *,
    min_year: int = 1950,
    max_year: int | None = None,
) -> ParsedDate | None:
    """
    Try to parse a date string. Returns None if no pattern matches plausibly.

    `min_year` / `max_year` clamp accepted dates to a reasonable range. Default
    max is today's year (configured at call site for testability).
    """
    if max_year is None:
        max_year = datetime.now().year

    norm = _normalize(raw)
    if not norm:
        return None

    candidates: list[ParsedDate] = []
    for pattern, yg, mg, dg, name, base_conf in _PATTERNS:
        for match in pattern.finditer(norm):
            try:
                y = _expand_year(int(match.group(yg)))
                m = int(match.group(mg))
                d = int(match.group(dg))
                if not (min_year <= y <= max_year):
                    continue
                if not (1 <= m <= 12):
                    continue
                if not (1 <= d <= 31):
                    continue
                dt = datetime(y, m, d, 12, 0, 0)  # noon default
                dt = _try_with_time(norm[match.end():], dt)
                # Boost confidence if month and day are both <=12 (ambiguous);
                # penalise if either is obviously out of range for the other order.
                conf = base_conf
                candidates.append(ParsedDate(dt=dt, raw=raw, pattern=name, confidence=conf))
            except ValueError:
                continue

    if not candidates:
        return None

    # Prefer the highest confidence; break ties by which one appeared first.
    candidates.sort(key=lambda c: -c.confidence)
    return candidates[0]


def parse_first_valid(strings: Iterable[str], **kwargs) -> ParsedDate | None:
    """Try parsing each string and return the first successful parse."""
    for s in strings:
        result = parse_date(s, **kwargs)
        if result is not None:
            return result
    return None
