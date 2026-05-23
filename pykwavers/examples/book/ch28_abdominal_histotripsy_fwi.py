"""Chapter 28: CT-derived abdominal FWI for histotripsy analysis.

This script calls pykwavers.run_theranostic_inverse_from_ritk for kidney
and liver abdominal histotripsy targets. When real CT/segmentation NIfTI
files are provided (via environment variables), the simulation runs on
actual CT-derived tissue properties. When paths are absent, kwavers uses
its built-in synthetic abdominal phantom.

All acoustic wave physics — FDTD/PSTD forward propagation, waveform-misfit
FWI with Charbonnier functional, RTM imaging, and iterative elastic shear
FWI — executes in Rust via the pykwavers PyO3 binding. No physics is
implemented in this file.

Transducer placement invariant
-------------------------------
Element positions satisfy placement_context_skin_gap_m >= 0: the transducer
array is positioned on the patient skin surface (hip/abdominal region), not
inside the patient.

Figures produced
----------------
fig01  CT anatomy, sound speed, anatomy FWI reconstruction, lesion target
fig02  Multi-modal reconstructions (subharmonic, harmonic, ultraharmonic, fused)
fig03  FWI convergence (anatomy objective, elastic shear objective)

Environment variables
---------------------
KWAVERS_CH28_KIDNEY_CT_NIFTI   path to kidney CT NIfTI (synthetic if absent)
KWAVERS_CH28_KIDNEY_SEG_NIFTI  path to kidney segmentation NIfTI
KWAVERS_CH28_LIVER_CT_NIFTI    path to liver CT NIfTI (synthetic if absent)
KWAVERS_CH28_LIVER_SEG_NIFTI   path to liver segmentation NIfTI
KWAVERS_CH28_GRID_SIZE         inversion grid size (default 64)
KWAVERS_CH28_ITERATIONS        FWI iterations (default 12)
KWAVERS_CH28_ELASTIC_ITER      elastic shear FWI iterations (default 3)

References
----------
Hall et al. (2007) Ultrasound Med. Biol. 33(9):1417
Parsons et al. (2006) Ultrasound Med. Biol. 32(1):115
Lin et al. (2014) IEEE Trans. UFFC 61(1):41
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pykwavers  # All physics: Rust FDTD/PSTD/FWI via PyO3

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch28"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "lines.linewidth": 1.5,
})

CT_KIDNEY_NIFTI = os.environ.get("KWAVERS_CH28_KIDNEY_CT_NIFTI", "")
CT_LIVER_NIFTI = os.environ.get("KWAVERS_CH28_LIVER_CT_NIFTI", "")
SEG_KIDNEY_NIFTI = os.environ.get("KWAVERS_CH28_KIDNEY_SEG_NIFTI", "")
SEG_LIVER_NIFTI = os.environ.get("KWAVERS_CH28_LIVER_SEG_NIFTI", "")
GRID_SIZE = int(os.environ.get("KWAVERS_CH28_GRID_SIZE", "64"))
ITERATIONS = int(os.environ.get("KWAVERS_CH28_ITERATIONS", "12"))
ELASTIC_ITERATIONS = int(os.environ.get("KWAVERS_CH28_ELASTIC_ITER", "3"))


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(OUT_DIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  saved: figures/ch28/{name}.{{pdf,png}}")


def _run_case(anatomy: str, ct_nifti: str, seg_nifti: str | None) -> dict:
    """
    Run kwavers theranostic FWI/RTM for one abdominal organ.

    Falls back to the kwavers built-in synthetic abdominal phantom when
    the CT path is absent or does not exist. All acoustic wave physics
    (FDTD/PSTD propagation, waveform-misfit FWI, RTM, elastic shear FWI)
    executes in Rust.
    """
    kwargs: dict = dict(
        ct_nifti_path=ct_nifti if ct_nifti and Path(ct_nifti).exists() else "__synthetic__",
        anatomy=anatomy,
        grid_size=GRID_SIZE,
        iterations=ITERATIONS,
        waveform_misfit="charbonnier",
        elastic_fwi_iterations=ELASTIC_ITERATIONS,
        transmit_schedule_strategy="full",
    )
    if seg_nifti and Path(seg_nifti).exists():
        kwargs["segmentation_nifti_path"] = seg_nifti
    return pykwavers.run_theranostic_inverse_from_ritk(**kwargs)


def _norm(arr: np.ndarray) -> np.ndarray:
    """Clip to non-negative and normalise to [0, 1]."""
    a = np.clip(np.asarray(arr, dtype=float), 0.0, None)
    mx = float(a.max())
    return a / mx if mx > 0.0 else a


# ── Figure 01: Anatomy + FWI reconstruction ───────────────────────────────────
def fig01_anatomy_fwi(results: dict[str, dict]) -> None:
    """CT HU, sound speed, anatomy FWI reconstruction, and lesion target."""
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 4,
                             figsize=(16, 5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    for row, (name, r) in enumerate(results.items()):
        dx = float(r["spacing_m"]) * 1e3
        ct = np.asarray(r["ct_hu"])
        cs = np.asarray(r["sound_speed_m_s"])
        anat = _norm(r["anatomy_reconstruction"])
        les = _norm(r["lesion_target"])
        nx, ny = ct.shape
        ext = [0, nx * dx, 0, ny * dx]
        foc = r.get("focus_m", (0.0, 0.0))
        m = r.get("metrics", {})
        anat_dice = m.get("anatomy", {}).get("dice_equal_area", float("nan"))
        anat_cnr = m.get("anatomy", {}).get("cnr", float("nan"))

        im0 = axes[row][0].imshow(ct.T, origin="lower", cmap="gray",
                                   vmin=-300, vmax=300, extent=ext, aspect="equal")
        axes[row][0].set_title(f"{name.capitalize()} — CT HU")
        axes[row][0].set_xlabel("x [mm]"); axes[row][0].set_ylabel("y [mm]")
        fig.colorbar(im0, ax=axes[row][0], fraction=0.046, label="HU")

        im1 = axes[row][1].imshow(cs.T, origin="lower", cmap="plasma",
                                   vmin=1450, vmax=1700, extent=ext, aspect="equal")
        axes[row][1].set_title(f"{name.capitalize()} — Sound speed [m/s]")
        axes[row][1].set_xlabel("x [mm]")
        fig.colorbar(im1, ax=axes[row][1], fraction=0.046, label="c [m/s]")

        anat_title = f"{name.capitalize()} — Anatomy FWI"
        if not (anat_dice != anat_dice):  # NaN check
            anat_title += f"\nDice={anat_dice:.3f} CNR={anat_cnr:.2f}"
        im2 = axes[row][2].imshow(anat.T, origin="lower", cmap="inferno",
                                   extent=ext, aspect="equal")
        axes[row][2].set_title(anat_title)
        axes[row][2].set_xlabel("x [mm]")
        axes[row][2].plot(foc[0] * 1e3, foc[1] * 1e3, "c+", ms=10, mew=1.5,
                          label="Focus")
        axes[row][2].legend(fontsize=7)
        fig.colorbar(im2, ax=axes[row][2], fraction=0.046, label="score")

        im3 = axes[row][3].imshow(les.T, origin="lower", cmap="hot",
                                   extent=ext, aspect="equal")
        axes[row][3].set_title(f"{name.capitalize()} — Lesion target")
        axes[row][3].set_xlabel("x [mm]")
        axes[row][3].plot(foc[0] * 1e3, foc[1] * 1e3, "c+", ms=10, mew=1.5)
        fig.colorbar(im3, ax=axes[row][3], fraction=0.046, label="norm.")

    fig.suptitle(
        "Figure 01 — CT anatomy and anatomy FWI reconstruction (kwavers Rust solver)\n"
        "(synthetic phantom when CT not provided; transducer on skin surface)",
        fontsize=11)
    savefig("fig01_anatomy_fwi")
    plt.close(fig)


# ── Figure 02: Multi-modal reconstructions ────────────────────────────────────
def fig02_multimodal(results: dict[str, dict]) -> None:
    """Subharmonic, harmonic, ultraharmonic, and fused FWI reconstructions."""
    keys = [
        "subharmonic_reconstruction",
        "harmonic_reconstruction",
        "ultraharmonic_reconstruction",
        "fused_reconstruction",
    ]
    labels = ["Subharmonic (f₀/2)", "Harmonic (2f₀)", "Ultraharmonic (3f₀/2)", "Fused"]
    metric_keys = ["subharmonic", "harmonic", "ultraharmonic", "fusion"]
    cmaps = ["Blues", "Reds", "Greens", "Purples"]

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 4,
                             figsize=(16, 5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    for row, (name, r) in enumerate(results.items()):
        dx = float(r["spacing_m"]) * 1e3
        foc = r.get("focus_m", (0.0, 0.0))
        m = r.get("metrics", {})

        for col, (key, lbl, cm, mk) in enumerate(zip(keys, labels, cmaps, metric_keys)):
            img = _norm(r[key])
            nx, ny = img.shape
            ext = [0, nx * dx, 0, ny * dx]
            dice = m.get(mk, {}).get("dice_equal_area", float("nan"))
            cnr = m.get(mk, {}).get("cnr", float("nan"))
            title = f"{name.capitalize()} — {lbl}"
            if not (dice != dice):  # not NaN
                title += f"\nDice={dice:.3f} CNR={cnr:.2f}"
            im = axes[row][col].imshow(img.T, origin="lower", cmap=cm,
                                       extent=ext, aspect="equal")
            axes[row][col].set_title(title)
            axes[row][col].set_xlabel("x [mm]")
            if col == 0:
                axes[row][col].set_ylabel("y [mm]")
            axes[row][col].plot(foc[0] * 1e3, foc[1] * 1e3, "k+", ms=8, mew=1.2)
            fig.colorbar(im, ax=axes[row][col], fraction=0.046, label="score")

    fig.suptitle(
        "Figure 02 — Multi-modal cavitation emission reconstructions (kwavers Rust FWI)\n"
        "Subharmonic + harmonic + ultraharmonic + fused reconstruction",
        fontsize=11)
    savefig("fig02_multimodal")
    plt.close(fig)


# ── Figure 03: FWI convergence ────────────────────────────────────────────────
def fig03_convergence(results: dict[str, dict]) -> None:
    """Anatomy FWI and elastic shear FWI objective history vs iteration."""
    fig, (ax_anat, ax_shear) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = ["#1f77b4", "#d62728"]

    for col_idx, (name, r) in enumerate(results.items()):
        color = colors[col_idx % 2]

        obj = np.asarray(r.get("objective_history", []), dtype=float)
        if obj.size > 0:
            j0 = max(float(obj[0]), 1.0e-30)
            ax_anat.semilogy(np.arange(len(obj)), obj / j0, "o-", color=color,
                             ms=5, lw=1.5, label=name.capitalize())
            reduction = (1.0 - float(obj[-1]) / j0) * 100.0
            print(f"  {name}: anatomy FWI objective reduction = {reduction:.1f}%")

        shear = np.asarray(r.get("elastic_shear_objective_history", []), dtype=float)
        if shear.size > 0:
            s0 = max(float(shear[0]), 1.0e-30)
            accepted = int(r.get("elastic_shear_accepted_step_count", 0))
            n_iter = int(r.get("elastic_shear_iteration_count", len(shear)))
            ax_shear.semilogy(np.arange(len(shear)), shear / s0, "s-", color=color,
                              ms=5, lw=1.5,
                              label=f"{name.capitalize()} ({accepted}/{n_iter} accepted)")

    for ax, title in [
        (ax_anat, "Anatomy FWI (waveform-misfit Charbonnier)"),
        (ax_shear, "Elastic shear FWI (Armijo line search)"),
    ]:
        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel("Normalized objective $\\mathcal{J}/\\mathcal{J}_0$")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Figure 03 — FWI convergence (kwavers Rust solver)", fontsize=11)
    savefig("fig03_convergence")
    plt.close(fig)


# ── Metrics export ────────────────────────────────────────────────────────────
def write_metrics(results: dict[str, dict]) -> Path:
    """Write reproducible scalar diagnostics for the generated figures."""
    payload = {
        "chapter": 28,
        "grid_size": GRID_SIZE,
        "iterations": ITERATIONS,
        "elastic_fwi_iterations": ELASTIC_ITERATIONS,
        "cases": {
            name: {
                "anatomy": r.get("metrics", {}).get("anatomy", {}),
                "active_lesion": r.get("metrics", {}).get("active_lesion", {}),
                "subharmonic": r.get("metrics", {}).get("subharmonic", {}),
                "harmonic": r.get("metrics", {}).get("harmonic", {}),
                "ultraharmonic": r.get("metrics", {}).get("ultraharmonic", {}),
                "fused": r.get("metrics", {}).get("fusion", {}),
                "waveform_objective": float(r.get("waveform_objective", float("nan"))),
                "elastic_shear_accepted_step_count": int(
                    r.get("elastic_shear_accepted_step_count", 0)
                ),
                "is_full_wave_inversion": bool(r.get("is_full_wave_inversion", False)),
                "uses_nonlinear_wave_propagation": bool(
                    r.get("uses_nonlinear_wave_propagation", False)
                ),
                "placement_context_skin_gap_m": float(
                    r.get("placement_context_skin_gap_m", 0.0)
                ),
            }
            for name, r in results.items()
        },
    }
    output = OUT_DIR / "metrics.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


# ── Entry point ───────────────────────────────────────────────────────────────
def run() -> dict[str, object]:
    """Generate Chapter 28 abdominal FWI figures and metrics."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("  [1/2] Running kidney FWI (kwavers Rust solver)...")
    results: dict[str, dict] = {}
    results["kidney"] = _run_case("kidney", CT_KIDNEY_NIFTI, SEG_KIDNEY_NIFTI)
    km = results["kidney"].get("placement_metrics", {})
    print(f"    skin→focus: {km.get('skin_to_focus_m', 0) * 100:.1f} cm")

    print("  [2/2] Running liver FWI (kwavers Rust solver)...")
    results["liver"] = _run_case("liver", CT_LIVER_NIFTI, SEG_LIVER_NIFTI)
    lm = results["liver"].get("placement_metrics", {})
    print(f"    skin→focus: {lm.get('skin_to_focus_m', 0) * 100:.1f} cm")

    # Verify transducer placement invariant: all elements outside body.
    for name, r in results.items():
        gap = float(r.get("placement_context_skin_gap_m", 0.0))
        assert gap >= 0.0, (
            f"{name}: skin_gap_m={gap:.4f} < 0 — element inside body! "
            "Transducer placement invariant violated."
        )
        print(f"  {name}: skin coupling gap = {gap * 1e3:.2f} mm (transducer outside body ✓)")

    print("  Generating figures...")
    print("  fig01...", end=" ", flush=True)
    fig01_anatomy_fwi(results)
    print("done")

    print("  fig02...", end=" ", flush=True)
    fig02_multimodal(results)
    print("done")

    print("  fig03...", end=" ", flush=True)
    fig03_convergence(results)
    print("done")

    metrics_path = write_metrics(results)
    print(f"  metrics → {metrics_path}")

    figure_names = ["fig01_anatomy_fwi.png", "fig02_multimodal.png", "fig03_convergence.png"]
    return {
        "figures": [str(OUT_DIR / n) for n in figure_names],
        "metrics": str(metrics_path),
    }


if __name__ == "__main__" or __name__ == "ch28":
    print("Chapter 28: CT-derived abdominal FWI for histotripsy analysis")
    print("=" * 60)
    print(f"  Grid size     : {GRID_SIZE}")
    print(f"  Iterations    : {ITERATIONS}")
    print(f"  Elastic iter  : {ELASTIC_ITERATIONS}")
    print(f"  Kidney CT     : {CT_KIDNEY_NIFTI or '(synthetic phantom)'}")
    print(f"  Liver CT      : {CT_LIVER_NIFTI or '(synthetic phantom)'}")
    run()
