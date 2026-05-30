"""Tests for region spec parsing."""

import pytest

from timestamp_extractor.region_detector import NAMED_REGIONS, parse_region_spec


@pytest.mark.parametrize("name", list(NAMED_REGIONS.keys()))
def test_all_named_corners_parse(name: str) -> None:
    result = parse_region_spec(name)
    assert result == NAMED_REGIONS[name]


def test_named_is_case_insensitive() -> None:
    assert parse_region_spec("Bottom-Right") == NAMED_REGIONS["bottom-right"]
    assert parse_region_spec("FULL") == NAMED_REGIONS["full"]


def test_percentages() -> None:
    assert parse_region_spec("78,83,92,94") == (0.78, 0.83, 0.92, 0.94)


def test_fractions() -> None:
    assert parse_region_spec("0.78,0.83,0.92,0.94") == (0.78, 0.83, 0.92, 0.94)


def test_whitespace_tolerated() -> None:
    assert parse_region_spec("  78 , 83 , 92 , 94  ") == (0.78, 0.83, 0.92, 0.94)


@pytest.mark.parametrize(
    "bad",
    [
        "nope",
        "1,2,3",                # not 4 values
        "50,60,40,80",          # x1 < x0
        "10,20,30,200",         # 200 → 2.0, out of [0,1]
        "a,b,c,d",              # not numeric
        "",
    ],
)
def test_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_region_spec(bad)
