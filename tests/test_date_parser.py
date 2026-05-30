"""Tests for the date parsing logic — covering many real-world camera formats."""

from datetime import datetime

import pytest

from timestamp_extractor.date_parser import parse_date


@pytest.mark.parametrize(
    "raw, expected",
    [
        # 4-digit year, ISO-ish
        ("1997 08 14", datetime(1997, 8, 14, 12, 0)),
        ("1997-08-14", datetime(1997, 8, 14, 12, 0)),
        ("2001/12/03", datetime(2001, 12, 3, 12, 0)),
        # 2-digit year with apostrophe (very common on film cameras)
        ("'97 8 14", datetime(1997, 8, 14, 12, 0)),
        ("97 8 14", datetime(1997, 8, 14, 12, 0)),
        # 2-digit year, 20xx
        ("'03 11 21", datetime(2003, 11, 21, 12, 0)),
        # With surrounding garbage from OCR
        ("garbage '97 8 14 more", datetime(1997, 8, 14, 12, 0)),
        # OCR confusions: O -> 0, I -> 1, S -> 5
        ("I997 O8 I4", datetime(1997, 8, 14, 12, 0)),
        # With time appended
        ("1997 08 14 16 42", datetime(1997, 8, 14, 16, 42)),
    ],
)
def test_parse_various_formats(raw: str, expected: datetime) -> None:
    result = parse_date(raw, min_year=1950, max_year=2026)
    assert result is not None, f"Failed to parse: {raw!r}"
    assert result.dt == expected


def test_rejects_out_of_range_year() -> None:
    # 1492 is before any photographic camera existed
    assert parse_date("1492 06 12", min_year=1950) is None


def test_rejects_invalid_month() -> None:
    # 13 isn't a valid month — and 20 isn't a valid day-as-month either
    # so neither YYYY MM DD nor DD MM YYYY parses
    assert parse_date("1997 13 20", min_year=1950) is None


def test_rejects_pure_garbage() -> None:
    assert parse_date("hello world", min_year=1950) is None
    assert parse_date("", min_year=1950) is None
    assert parse_date("xxx", min_year=1950) is None


def test_returns_confidence() -> None:
    result = parse_date("1997-08-14", min_year=1950, max_year=2026)
    assert result is not None
    assert 0.0 < result.confidence <= 1.0
    assert "YYYY MM DD" in result.pattern
