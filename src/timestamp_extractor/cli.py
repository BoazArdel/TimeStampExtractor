"""
Command-line interface for batch processing scanned photos.

Usage:
    timestamp-extractor /path/to/photos --pattern '*.jpg'
    timestamp-extractor ./scans --dry-run --no-ai
    timestamp-extractor ./scans --report results.csv --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from . import __version__
from .pipeline import PipelineConfig, ProcessResult, process_image


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="timestamp-extractor",
        description=(
            "Extract red date stamps from scanned photos and write them to EXIF "
            "DateTimeOriginal. Uses Tesseract OCR with Claude vision as a smart "
            "fallback."
        ),
    )
    p.add_argument("input", help="Folder containing image files (or a single file).")
    p.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern for files inside the folder (default: *.jpg). "
        "Use ** for recursive (e.g. '**/*.jpg').",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write a CSV report of the run to this path.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default 1; AI calls are network-bound "
        "so 4-8 is fine if you have an API key).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction but do NOT modify any files.",
    )
    p.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable the Claude AI fallback. Tesseract-only.",
    )
    p.add_argument(
        "--ai-only",
        action="store_true",
        help="Skip Tesseract entirely and use Claude vision for every image. "
        "Implies --no-ai is OFF. Costs more but most reliable on faded stamps.",
    )
    p.add_argument(
        "--region",
        default=None,
        help=(
            "Skip auto-detection and use a fixed region. Either a named corner "
            "(bottom-right, bottom-left, top-right, top-left, bottom, top, full) "
            "or four comma-separated numbers x0,y0,x1,y1 as percentages or fractions "
            "(e.g. '78,83,92,94' or '0.78,0.83,0.92,0.94')."
        ),
    )
    p.add_argument(
        "--min-year",
        type=int,
        default=1950,
        help="Reject parsed dates earlier than this year (default 1950).",
    )
    p.add_argument(
        "--max-year",
        type=int,
        default=datetime.now().year,
        help="Reject parsed dates later than this year (default: current year).",
    )
    p.add_argument(
        "--save-debug",
        type=Path,
        default=None,
        help="Save intermediate crops to this folder for debugging.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v info, -vv debug).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _discover_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")
    return sorted(p for p in input_path.glob(pattern) if p.is_file())


def _write_report(report_path: Path, results: list[ProcessResult]) -> None:
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "success",
                "parsed_date",
                "backend",
                "pattern_used",
                "date_confidence",
                "region_source",
                "region_confidence",
                "error",
            ]
        )
        for r in results:
            w.writerow(
                [
                    str(r.path),
                    r.success,
                    r.parsed_date.dt.isoformat() if r.parsed_date else "",
                    r.backend or "",
                    r.parsed_date.pattern if r.parsed_date else "",
                    f"{r.parsed_date.confidence:.2f}" if r.parsed_date else "",
                    r.region.source if r.region else "",
                    f"{r.region.confidence:.2f}" if r.region else "",
                    r.error or "",
                ]
            )


def _save_debug_crops(result: ProcessResult, debug_dir: Path) -> None:
    import cv2  # local import keeps debug code optional at runtime

    stem = result.path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, img in result.debug_crops.items():
        out = debug_dir / f"{stem}_{name}.png"
        cv2.imwrite(str(out), img)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # pick up ANTHROPIC_API_KEY etc.
    args = _build_parser().parse_args(argv)

    log_level = (
        logging.WARNING if args.verbose == 0
        else logging.INFO if args.verbose == 1
        else logging.DEBUG
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    files = _discover_files(Path(args.input), args.pattern)
    if not files:
        print(f"No files matched pattern {args.pattern!r} under {args.input}", file=sys.stderr)
        return 1

    if args.no_ai and args.ai_only:
        print("--no-ai and --ai-only are mutually exclusive.", file=sys.stderr)
        return 1

    # Validate region spec early so a typo fails before we touch any files
    if args.region:
        try:
            from .region_detector import parse_region_spec
            parse_region_spec(args.region)
        except ValueError as e:
            print(f"Invalid --region: {e}", file=sys.stderr)
            return 1

    config = PipelineConfig(
        use_ai_fallback=not args.no_ai,
        ai_only=args.ai_only,
        region_spec=args.region,
        min_year=args.min_year,
        max_year=args.max_year,
        dry_run=args.dry_run,
        save_debug=args.save_debug is not None,
        debug_dir=args.save_debug,
    )

    mode = "ai-only" if args.ai_only else ("tesseract-only" if args.no_ai else "hybrid")
    region_msg = f"region={args.region!r}" if args.region else "auto-detect region"
    print(
        f"Processing {len(files)} files "
        f"(mode={mode}, {region_msg}, workers={args.workers}, dry_run={args.dry_run})"
    )

    results: list[ProcessResult] = []
    if args.workers <= 1:
        for f in tqdm(files, desc="Photos", unit="img"):
            r = process_image(f, config)
            if args.save_debug:
                _save_debug_crops(r, args.save_debug)
            results.append(r)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_image, f, config): f for f in files}
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Photos",
                unit="img",
            ):
                r = fut.result()
                if args.save_debug:
                    _save_debug_crops(r, args.save_debug)
                results.append(r)

    # Summary
    successes = sum(1 for r in results if r.success)
    by_backend: dict[str, int] = {}
    for r in results:
        if r.success and r.backend:
            by_backend[r.backend] = by_backend.get(r.backend, 0) + 1

    print()
    print(f"Done: {successes}/{len(results)} succeeded.")
    if by_backend:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_backend.items()))
        print(f"By backend: {breakdown}")
    if args.dry_run:
        print("(dry-run — no files were modified)")

    if args.report:
        _write_report(args.report, results)
        print(f"Report written to {args.report}")

    # Exit non-zero only if NOTHING succeeded
    return 0 if successes > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
