# TimeStampExtractor v2

Read the red date stamp imprinted on a scanned photo, then write it back into the file's EXIF `DateTimeOriginal` so photo libraries (Apple Photos, Google Photos, Lightroom, your file browser) sort the scan to its real date instead of the day you scanned it.

This is a ground-up rewrite of [BoazArdel/TimeStampExtractor](https://github.com/BoazArdel/TimeStampExtractor) from 2019. The original idea is intact; the implementation is modernised and the missing pieces (EXIF write, batch CLI, AI fallback, robust region detection) are added.

---

## What's new vs the original

| Area | v1 (2019) | v2 (this repo) |
|---|---|---|
| Region detection | Hardcoded `(0.78–0.92, 0.83–0.94)` of the image | Auto-detected via HSV red mask + connected components, with v1's region as fallback |
| Red-pixel filter | Per-pixel Python loop, very slow | Vectorised NumPy/OpenCV, ~100× faster |
| OCR | Tesseract OR a custom seven-segment decoder, never unified | Tesseract first; **Claude vision API** as fallback when OCR fails or is low-confidence |
| Date parsing | None — just printed raw OCR | Multi-format regex (`'YY MM DD`, `MM DD 'YY`, `YYYY-MM-DD`, with-time variants…) plus OCR-confusion fix-ups (`O→0`, `I→1`, `S→5`) |
| EXIF write | Never implemented — the actual goal of the project | `piexif` writes `DateTimeOriginal`, `DateTimeDigitized`, `DateTime`; also sets file mtime |
| Batch handling | Windows-only hardcoded path | Cross-platform CLI with glob patterns, parallel workers, progress bar, CSV report, dry-run |
| Packaging | Loose scripts | `pyproject.toml`, installable as `timestamp-extractor` command, type hints, tests |

### Why AI as a fallback (and not always)

Red stamps from 1990s/2000s film cameras are inconsistent: faded ink, weird seven-segment fonts, partial occlusion by image content, unusual orderings (`DD MM 'YY` vs `MM DD 'YY`). Tesseract handles the easy 70–80% of cases fast and free. The remainder — where v1 just gave up — is what AI is good at: a multimodal model can read a smudged `'97` even with one digit half-faded.

By gating Claude behind a Tesseract-confidence check, you only pay for the photos that actually need it. Default model is `claude-haiku-4-5` (cheap, fast). You can switch to `claude-sonnet-4-6` via `.env` for harder collections.

### Modes

| Flag | Tesseract | Claude | When to use |
|---|---|---|---|
| *(default)* | yes | fallback | General-purpose. Pays for AI only when OCR fails. |
| `--no-ai` | yes | never | Fully offline / no API key. Stamps must be readable. |
| `--ai-only` | no | every image | Faded / unusual stamps across the whole batch, or when you don't have Tesseract installed at all. |

### Skipping region detection

If you already know where the timestamp sits on every scan (because they all came from the same scanner and camera), pass `--region` to skip the auto-detector. It's faster, deterministic, and removes a class of failures where the detector picks up something red in the photo content itself.

```bash
--region bottom-right            # named corner
--region 78,83,92,94             # percentages, exactly the v1 region
--region 0.78,0.83,0.92,0.94     # fractions, same thing
--region full                    # send the entire image (useful with --ai-only)
```

---

## Install

```bash
# 1. Clone / unzip and enter the folder
cd timestamp_extractor_v2

# 2. Create a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install
pip install -e .

# 4. System dependency: Tesseract OCR
#    macOS:   brew install tesseract
#    Ubuntu:  sudo apt install tesseract-ocr
#    Windows: https://github.com/UB-Mannheim/tesseract/wiki  then set TESSERACT_CMD in .env

# 5. Config: copy the env template and add your Anthropic key (only needed for AI fallback)
cp .env.example .env
# edit .env and paste your key from https://console.anthropic.com/
```

---

## Command examples

```bash
# Dry-run on a folder — extract dates but don't touch any files. Always do this first.
timestamp-extractor ./scans --dry-run --report results.csv -v

# Real run: process every .jpg in a folder
timestamp-extractor ./scans

# Recursive
timestamp-extractor ./scans --pattern '**/*.jpg'

# Skip the AI fallback (Tesseract-only, no API calls)
timestamp-extractor ./scans --no-ai --dry-run

# Use Claude vision for EVERY photo (skip Tesseract entirely).
# Best when stamps are faded across the whole batch.
timestamp-extractor ./scans --ai-only

# You already know where the stamp lives — skip auto-detection.
# Named corner shortcuts: bottom-right, bottom-left, top-right, top-left,
#                         bottom, top, full
timestamp-extractor ./scans --region bottom-right

# Or pin a custom region with percentages (x0,y0,x1,y1).
# This matches the v1 hardcoded region exactly:
timestamp-extractor ./scans --region 78,83,92,94

# Combine: fixed region + AI-only is the most reliable (and most expensive) mode
timestamp-extractor ./scans --region bottom-right --ai-only --report results.csv

# 4 parallel workers (good when AI is in play)
timestamp-extractor ./scans --workers 4 --report results.csv

# Single file
timestamp-extractor ./scans/IMG_0042.jpg -v

# Save intermediate crops to debug a failure
timestamp-extractor ./tricky --save-debug ./debug_out -vv

# Restrict accepted year range (rejects parses outside the window)
timestamp-extractor ./family-photos --min-year 1985 --max-year 2010
```

After a real run, open one of the modified JPEGs in your OS file properties — the "Date taken" / "Photo taken on" field should now match the imprinted stamp.

---

## How it works

```
                              ┌──────────────────┐
   image file  ───►  load ───►│ detect_region    │  HSV red mask, connected components,
                              │ (region_detector)│  score by aspect / corner proximity
                              └────────┬─────────┘
                                       ▼
                              ┌──────────────────┐
                              │ isolate_red_text │  Vectorised mask, dilate,
                              │ (preprocessing)  │  invert, upscale, denoise
                              └────────┬─────────┘
                                       ▼
                              ┌──────────────────┐
                              │ Tesseract OCR    │  digits-only whitelist, PSM 7
                              │ (ocr.py)         │
                              └────────┬─────────┘
                                       ▼
                              ┌──────────────────┐    parse_date() — regex over many
                              │ date_parser      │    formats; OCR fixups; range check
                              └────────┬─────────┘
                                       │
                  ╔════════════════════╪═══════════════════════╗
                  ║   parse failed?    │     parse succeeded?  ║
                  ▼                                            ▼
        ┌──────────────────┐                          ┌──────────────────┐
        │ Claude vision    │                          │ write_exif       │
        │ fallback         │  ───►  parse_date  ───►  │ (piexif + mtime) │
        └──────────────────┘                          └──────────────────┘
```

### Supported timestamp formats out of the box

* `1997 08 14` / `1997-08-14` / `1997/08/14`
* `'97 8 14` / `97 8 14`  (apostrophe-year, classic film camera)
* `8 14 '97`  (US order)
* `14 8 '97`  (European order)
* `8.14.97` / `14-08-1997`
* Any of the above with trailing `HH MM` time
* OCR mistakes auto-corrected: `O→0`, `I/l/|→1`, `S→5`, `Z→2`, `B→8`, `g/q→9`

Add your own in `src/timestamp_extractor/date_parser.py` (`_PATTERNS` list).

---

## Project layout

```
timestamp_extractor_v2/
├── src/timestamp_extractor/
│   ├── __init__.py
│   ├── cli.py                # argparse, batch loop, CSV report
│   ├── pipeline.py           # orchestrates one image end-to-end
│   ├── region_detector.py    # find where the timestamp sits
│   ├── preprocessing.py      # isolate red text, prep for OCR
│   ├── ocr.py                # Tesseract + Claude backends
│   ├── date_parser.py        # multi-format date regex
│   └── exif_writer.py        # piexif + mtime
├── tests/
│   └── test_date_parser.py
├── examples/                 # drop scanned photos here for manual testing
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```

The parser has the most interesting logic and is covered. Region detection and EXIF writing are best tested with real scans — drop a few into `examples/` and run `--dry-run --save-debug ./debug_out -vv` to see what the pipeline sees.

---

## Troubleshooting

* **"Tesseract binary not found"** — install Tesseract for your OS (see Install step 4) or set `TESSERACT_CMD` in `.env`.
* **"ANTHROPIC_API_KEY is not set"** — either add the key to `.env` or run with `--no-ai`.
* **A photo's date is parsed but obviously wrong** — run with `-vv --save-debug ./debug_out` to see the crop the OCR saw. Often the answer is that the timestamp is in a different position than expected and the detector picked up something else; you may need to add a stricter `min_year` / `max_year` window for your collection.
* **A photo with a clear stamp still fails** — try with `--workers 1 -vv`. If Tesseract returns garbage but the AI fallback also says UNREADABLE, the stamp is probably too faded; consider rescanning the original.

---

## License

MIT.
