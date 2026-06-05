"""
Master Figure Generation Script — kwavers Ultrasound Physics Book
=================================================================

Runs all chapter figure generation scripts in order. The chapter roster
is read from ``chapters.toml`` in the same directory (single source of truth).
Each chapter script is independent and can also be run standalone.

Usage::

    python generate_all_figures.py [--chapter N]

With no arguments, generates all chapters. With ``--chapter N``, generates
only chapter N (number as listed in chapters.toml).

Output: docs/book/figures/chNN/ for each chapter NN.

Requires: numpy, matplotlib, scipy
Optional: pykwavers (Chapter 1 fig06 and Chapter 3 fig06 only)

Chapter manifest: chapters.toml (co-located with this script).
Do NOT hardcode chapter metadata here; edit chapters.toml instead.
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback

# ── TOML parsing: stdlib in Python 3.11+; fall back to tomli for 3.10 ────────
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli
    except ImportError as exc:
        raise SystemExit(
            "TOML parser not found. Use Python ≥ 3.11 or install tomli:\n"
            "    pip install tomli"
        ) from exc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BOOK_SCRIPTS_DIR = os.path.dirname(__file__)
CHAPTERS_TOML = os.path.join(BOOK_SCRIPTS_DIR, "chapters.toml")


def load_chapters() -> list[tuple[int, str, str]]:
    """Parse chapters.toml and return a list of (number, script, title) tuples.

    Validates that chapter numbers are unique and monotonically increasing.

    Raises
    ------
    SystemExit
        If chapters.toml is missing, malformed, or fails schema validation.
    """
    if not os.path.exists(CHAPTERS_TOML):
        raise SystemExit(
            f"Chapter manifest not found: {CHAPTERS_TOML}\n"
            "Ensure chapters.toml is present in the same directory as this script."
        )

    with open(CHAPTERS_TOML, "rb") as fh:
        data = tomllib.load(fh)

    raw = data.get("chapter", [])
    if not raw:
        raise SystemExit("chapters.toml contains no [[chapter]] entries.")

    chapters: list[tuple[int, str, str]] = []
    seen_numbers: set[int] = set()
    prev_number = 0

    for entry in raw:
        try:
            number = int(entry["number"])
            script = str(entry["script"])
            title = str(entry["title"])
        except KeyError as exc:
            raise SystemExit(
                f"chapters.toml entry missing required field: {exc}\n"
                "Each [[chapter]] must have: number, script, title."
            ) from exc

        if number in seen_numbers:
            raise SystemExit(
                f"chapters.toml: duplicate chapter number {number}."
            )
        if number <= prev_number:
            raise SystemExit(
                f"chapters.toml: chapter numbers must be strictly increasing; "
                f"got {number} after {prev_number}."
            )

        seen_numbers.add(number)
        prev_number = number
        chapters.append((number, script, title))

    return chapters


def run_chapter(ch_num: int, script_name: str, ch_title: str) -> bool:
    """Run a chapter figure script and return success status."""
    script_path = os.path.join(BOOK_SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"  [SKIP] Chapter {ch_num}: {script_name} not found")
        return False

    print(f"\n{'='*60}")
    print(f"Chapter {ch_num}: {ch_title}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    try:
        module_name = f"ch{ch_num:02d}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        elapsed = time.perf_counter() - t0
        print(f"  [OK] Chapter {ch_num} completed in {elapsed:.1f}s")
        return True
    except Exception:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        print(f"  [ERROR] Chapter {ch_num} failed after {elapsed:.1f}s:")
        traceback.print_exc()
        return False


def main() -> None:
    chapter_scripts = load_chapters()
    chapter_numbers = [n for n, _, _ in chapter_scripts]
    ch_min = chapter_numbers[0]
    ch_max = chapter_numbers[-1]

    parser = argparse.ArgumentParser(
        description=(
            "Generate all book figures for the kwavers ultrasound physics book. "
            f"Chapter manifest: {os.path.relpath(CHAPTERS_TOML)}."
        )
    )
    parser.add_argument(
        "--chapter", type=int, default=None,
        help=(
            f"Generate only this chapter number ({ch_min}–{ch_max}). "
            "Default: all chapters."
        ),
    )
    args = parser.parse_args()

    sys.path.insert(0, BOOK_SCRIPTS_DIR)

    chapters_to_run = chapter_scripts
    if args.chapter is not None:
        chapters_to_run = [
            (n, s, t) for n, s, t in chapter_scripts if n == args.chapter
        ]
        if not chapters_to_run:
            valid = ", ".join(str(n) for n in chapter_numbers)
            print(
                f"ERROR: Chapter {args.chapter} not found in chapters.toml. "
                f"Valid numbers: {valid}."
            )
            sys.exit(1)

    t_total_start = time.perf_counter()
    results = []
    for ch_num, script_name, ch_title in chapters_to_run:
        ok = run_chapter(ch_num, script_name, ch_title)
        results.append((ch_num, ch_title, ok))

    t_total = time.perf_counter() - t_total_start

    # Summary
    print(f"\n{'='*60}")
    print("FIGURE GENERATION SUMMARY")
    print(f"{'='*60}")
    n_ok = sum(1 for _, _, ok in results if ok)
    n_fail = len(results) - n_ok
    for ch_num, ch_title, ok in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] Chapter {ch_num}: {ch_title}")
    print(f"\nTotal: {n_ok}/{len(results)} chapters completed in {t_total:.1f}s")
    if n_fail > 0:
        print(f"WARNING: {n_fail} chapter(s) failed — check error output above.")
        sys.exit(1)

    # List output directories
    print("\nOutput directories:")
    for ch_num, _, ok in results:
        if ok:
            out_dir = os.path.join(
                REPO_ROOT, "docs", "book", "figures", f"ch{ch_num:02d}"
            )
            if os.path.isdir(out_dir):
                n_files = len([f for f in os.listdir(out_dir) if f.endswith(".pdf")])
                print(
                    f"  docs/book/figures/ch{ch_num:02d}/  ({n_files} PDF files)"
                )


if __name__ == "__main__":
    main()
