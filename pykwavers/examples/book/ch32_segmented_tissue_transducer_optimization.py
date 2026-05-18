"""Chapter 32: segmentation-driven transducer optimization.

This chapter example treats normal tissue, tumor, air, bone, fat, and protected
structures as planning constraints.  It uses the local LiTS17 liver CT sample by
default, with liver and tumor from native segmentation labels and acoustic
hazards derived from CT Hounsfield units.  A hybrid optimizer screens discrete
candidate apertures by segmented ray paths, then solves complex least-squares
drive weights that shape the focal spot inside the tumor while nulling protected
anatomy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


BOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BOOK_DIR.parents[2]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch32"
LIVER_CT = REPO_ROOT / "data" / "lits17_sample" / "volume-0.nii"
LIVER_SEG = REPO_ROOT / "data" / "lits17_sample" / "segmentation-0.nii"

if str(BOOK_DIR) not in sys.path:
    sys.path.insert(0, str(BOOK_DIR))

from segmented_lesion_planning.figures import render_plan, write_metrics  # noqa: E402
from segmented_lesion_planning.liver_dataset import load_lits_liver_planning_grid  # noqa: E402
from segmented_lesion_planning.phantom import build_segmented_therapy_phantom  # noqa: E402
from segmented_lesion_planning.solver import optimize_transducer_layout  # noqa: E402
from segmented_lesion_planning.types import HybridPlanConfig  # noqa: E402


def load_planning_grid() -> tuple[object, dict[str, object]]:
    source = os.environ.get("KWAVERS_CH32_SOURCE", "liver").strip().lower()
    if source == "phantom":
        return build_segmented_therapy_phantom(), {"source": "analytic segmented phantom"}
    try:
        return load_lits_liver_planning_grid(LIVER_CT, LIVER_SEG)
    except (FileNotFoundError, ValueError, OSError) as exc:
        grid = build_segmented_therapy_phantom()
        return grid, {
            "source": "analytic segmented phantom",
            "fallback_reason": str(exc),
            "requested_source": "LiTS17 sample liver CT",
        }


def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid, dataset = load_planning_grid()
    config = HybridPlanConfig()
    result = optimize_transducer_layout(grid, config)
    result["dataset"] = dataset
    figures = render_plan(grid, result, OUT_DIR)
    metrics = write_metrics(grid, result, figures, OUT_DIR / "metrics.json")
    payload = {"figures": [str(path) for path in figures], "metrics": str(metrics)}
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return payload


if __name__ == "__main__" or __name__ == "ch32":
    run()
