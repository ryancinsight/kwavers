"""Chapter 32: segmentation-driven HistoSonics liver histotripsy.

Demonstrates:
  1. 3-D focused-bowl placement on the liver skin surface derived from CT
     segmentation (plan_abdominal_array_placement_from_ritk_ct).
  2. Full-wave theranostic inversion using Westervelt nonlinear PSTD
     propagation, multi-frequency HADAMARD-encoded receive, acoustic-speed /
     nonlinearity FWI with Charbonnier misfit, elastic-shear RTM, and
     harmonic / subharmonic / ultraharmonic cavitation reconstruction
     (run_theranostic_inverse_from_ritk).

Physics: kwavers Rust library (pykwavers PyO3 binding).
Visualization: matplotlib only.
Data: LiTS17 liver CT + segmentation.  Missing files trigger the built-in
      synthetic abdominal liver phantom automatically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pykwavers

BOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BOOK_DIR.parents[2]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch32"
LIVER_CT = str(REPO_ROOT / "data" / "lits17_sample" / "volume-0.nii")
LIVER_SEG = str(REPO_ROOT / "data" / "lits17_sample" / "segmentation-0.nii")

if str(BOOK_DIR) not in sys.path:
    sys.path.insert(0, str(BOOK_DIR))

from segmented_lesion_planning.figures import (  # noqa: E402
    plot_3d_placement,
    plot_exposure_slice,
    plot_fwi_convergence,
    plot_reconstructions,
    write_metrics,
)


def run() -> dict[str, object]:
    """Run ch32 end-to-end.

    Returns a payload dict containing figure paths and the metrics JSON path.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: 3-D bowl placement ──────────────────────────────────────────
    # plan_abdominal_array_placement_from_ritk_ct places a HistoSonics-like
    # 256-element focused bowl on the anterior abdominal skin surface targeting
    # the liver organ centroid via Fibonacci golden-spiral element distribution.
    # Falls back to a built-in synthetic abdominal liver phantom automatically
    # when the LiTS17 NIfTI files are absent on disk.
    placement = pykwavers.plan_abdominal_array_placement_from_ritk_ct(
        LIVER_CT,
        LIVER_SEG,
        anatomy_label="liver",
        element_count=256,
        surface_stride=6,
    )

    # ── Phase 2: theranostic inverse (FWI + exposure) ─────────────────────────
    # run_theranostic_inverse_from_ritk executes the following pipeline in Rust:
    #   1. Westervelt nonlinear PSTD forward exposure (source_pressure_pa = 2.5 MPa,
    #      650 kHz fundamental with 2nd and 3rd harmonics).
    #   2. HADAMARD-coded multi-frequency receive using 3 encoding rows per code.
    #   3. Acoustic-speed and nonlinearity-parameter FWI
    #      (Charbonnier waveform misfit, 12 outer iterations).
    #   4. Elastic-shear ElasticPSTD RTM channel (3 elastic FWI iterations).
    #   5. Harmonic / subharmonic / ultraharmonic cavitation reconstruction.
    #   6. Multi-modal fusion image.
    # Falls back to the built-in synthetic liver phantom when NIfTI is absent.
    result = pykwavers.run_theranostic_inverse_from_ritk(
        LIVER_CT,
        LIVER_SEG,
        anatomy="liver",
        grid_size=64,
        iterations=12,
        frequencies_hz=[650_000.0, 1_300_000.0, 1_950_000.0],
        source_pressure_pa=2.5e6,
        waveform_misfit="charbonnier",
        waveform_misfit_scale_fraction=0.012,
        elastic_fwi_iterations=3,
        transmit_schedule_strategy="full",
    )

    # ── Figures ───────────────────────────────────────────────────────────────
    figures = [
        plot_3d_placement(placement, OUT_DIR / "fig01_bowl_placement_3d.png"),
        plot_exposure_slice(result, OUT_DIR / "fig02_exposure_and_segmentation.png"),
        plot_reconstructions(result, OUT_DIR / "fig03_multimodal_reconstruction.png"),
        plot_fwi_convergence(result, OUT_DIR / "fig04_fwi_convergence.png"),
    ]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_path = write_metrics(placement, result, figures, OUT_DIR / "metrics.json")

    summary = {
        "anatomy": result["anatomy"],
        "device_model": result["device_model"],
        "geometry_model": result["geometry_model"],
        "is_full_wave_inversion": result["is_full_wave_inversion"],
        "uses_nonlinear_wave_propagation": result["uses_nonlinear_wave_propagation"],
        "iterative_elastic_fwi": result["iterative_elastic_fwi"],
        "element_count": placement["element_count"],
        "fwi_final_objective": float(result["objective_history"][-1]),
        "fwi_iterations": int(len(result["objective_history"])),
        "synthetic_phantom": placement["synthetic_phantom"],
        "exposure_model": result["exposure_model"],
        "operator_model": result["operator_model"],
        "waveform_model": result["waveform_model"],
        "waveform_misfit": result["waveform_misfit"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    return {
        "figures": [str(f) for f in figures],
        "metrics": str(metrics_path),
        "summary": summary,
    }


if __name__ == "__main__" or __name__ == "ch32":
    run()
