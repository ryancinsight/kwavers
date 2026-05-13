from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np

BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_manifest_includes_chapter29_with_script():
    manifest = tomllib.loads((BOOK_DIR / "chapters.toml").read_text(encoding="utf-8"))
    chapters = {int(entry["number"]): entry for entry in manifest["chapter"]}

    assert chapters[29]["script"] == "ch29_theranostic_fwi_platforms.py"
    assert "Same-Device" in chapters[29]["title"]
    assert (BOOK_DIR / chapters[29]["script"]).is_file()


def test_book_readme_links_chapter29_markdown():
    readme = (DOCS_DIR / "README.md").read_text(encoding="utf-8")

    assert "(theranostic_fwi_platforms.md)" in readme
    assert (DOCS_DIR / "theranostic_fwi_platforms.md").is_file()


def test_chapter29_layout_helpers_report_skin_and_helmet_clearance():
    import ch29_theranostic_fwi_platforms as ch29

    kidney = {
        "anatomy": "kidney",
        "placement_metrics": {
            "skin_contact_to_nearest_aperture_m": 0.003,
            "min_body_clearance_m": 0.003,
        },
    }
    brain = {
        "anatomy": "brain",
        "placement_metrics": {
            "skin_contact_to_nearest_aperture_m": 0.016,
            "min_body_clearance_m": 0.015,
        },
    }

    x_limits = ch29.axis_limits(
        [-0.04, 0.04],
        np.asarray([-0.11, 0.09]),
        np.asarray([-0.043]),
    )
    y_limits = ch29.axis_limits([-0.05, 0.05], np.asarray([]), np.asarray([]))

    assert "skin gap 3.0 mm" == ch29.placement_label(kidney)
    assert "helmet clearance 15.0 mm" == ch29.placement_label(brain)
    assert x_limits[0] < -0.11
    assert x_limits[1] > 0.09
    assert y_limits[0] < -0.05
    assert y_limits[1] > 0.05
