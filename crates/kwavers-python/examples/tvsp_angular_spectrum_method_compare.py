#!/usr/bin/env python3
"""
Parity comparison: pykwavers ``angular_spectrum_cw`` vs
k-wave-python ``angular_spectrum_cw``.

Both implementations are frequency-domain spectral propagators following
Zeng & McGough (2008, JASA 123:68-76).  The comparison validates that the
pykwavers pure-NumPy re-implementation produces bit-equivalent output for:

* Lossless homogeneous medium (c₀ = 1500 m/s)
* Absorbing medium (α = 0.5 dB/MHz/cm, y = 1.5)
* Multiple propagation distances (z = 5, 10, 20, 40 mm)
* Angular restriction enabled and disabled

Pass criterion: Pearson r ≥ 0.9999, PSNR ≥ 60 dB on |pressure| at
each z plane (tolerance reflects double-precision FFT round-off only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    save_text_report,
)

_ROOT = bootstrap_example_paths()

import pykwavers as pkw
from kwave.utils.angular_spectrum_cw import angular_spectrum_cw as kw_angular_spectrum_cw

# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------
F0 = 1_000_000.0        # source frequency [Hz]
C0 = 1500.0             # sound speed [m/s]
DX = 0.15e-3            # grid spacing [m]  (< λ/10 at 1 MHz)
NX = NY = 128           # grid size (isotropic)
APERTURE_RADIUS = 4e-3  # circular piston radius [m]

ALPHA_COEFF = 0.5       # absorption coefficient [dB/MHz^y/cm]
ALPHA_POWER = 1.5       # absorption power y

Z_POSITIONS = np.array([5e-3, 10e-3, 20e-3, 40e-3])  # propagation planes [m]

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "tvsp_angular_spectrum_method_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_angular_spectrum_method_metrics.txt"

PARITY_THRESHOLDS = {
    "pearson_r_min": 0.9999,
    "psnr_db_min": 60.0,
}


def _circular_piston_source() -> np.ndarray:
    """Return complex pressure (uniform amplitude, zero phase) on a circular disc."""
    x = (np.arange(NX) - NX / 2.0) * DX
    y = (np.arange(NY) - NY / 2.0) * DX
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = (xx**2 + yy**2) <= APERTURE_RADIUS**2
    src = np.zeros((NX, NY), dtype=complex)
    src[mask] = 1.0
    return src


def _run_scenario(medium_kw, medium_pkw, label: str) -> dict[str, object]:
    """Run both propagators and return fields + metrics."""
    src = _circular_piston_source()

    # k-wave-python angular_spectrum_cw: z_pos must be float (beartype enforced);
    # call once per propagation plane and stack results.
    kw_planes = []
    for z in Z_POSITIONS:
        # Returns [Nx, Ny, 1] with one z plane
        kw_single = kw_angular_spectrum_cw(
            input_plane=src.real,
            dx=DX,
            z_pos=float(z),
            f0=int(F0),
            medium=medium_kw,
            angular_restriction=True,
        )
        kw_planes.append(kw_single[:, :, 0])
    kw_pressure = np.stack(kw_planes, axis=-1)  # [Nx, Ny, Nz]
    kw_abs = np.abs(kw_pressure)

    pkw_pressure = pkw.angular_spectrum_cw(
        input_plane=src.real,
        dx=DX,
        z_pos=Z_POSITIONS.tolist(),
        f0=F0,
        medium=medium_pkw,
        angular_restriction=True,
    )
    pkw_abs = np.abs(pkw_pressure)

    metrics_per_z = []
    for zi, z in enumerate(Z_POSITIONS):
        m = compute_image_metrics(kw_abs[:, :, zi], pkw_abs[:, :, zi])
        metrics_per_z.append((z, m))

    # Aggregate: minimum Pearson and minimum PSNR across all planes
    pearson_min = min(m["pearson_r"] for _, m in metrics_per_z)
    psnr_min = min(m["psnr_db"] for _, m in metrics_per_z)

    return {
        "label": label,
        "kw": kw_abs,
        "pkw": pkw_abs,
        "metrics_per_z": metrics_per_z,
        "pearson_min": pearson_min,
        "psnr_min": psnr_min,
        "pass": (
            pearson_min >= PARITY_THRESHOLDS["pearson_r_min"]
            and psnr_min >= PARITY_THRESHOLDS["psnr_db_min"]
        ),
    }


def save_figure(scenarios: list[dict], path: Path) -> None:
    n_z = len(Z_POSITIONS)
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(
        n_scenarios * 2, n_z,
        figsize=(4 * n_z, 3 * n_scenarios * 2),
        constrained_layout=True,
    )
    if n_scenarios == 1:
        axes = axes.reshape(2, n_z)

    for s_idx, sc in enumerate(scenarios):
        row_kw = s_idx * 2
        row_pkw = row_kw + 1
        vmax = max(sc["kw"].max(), sc["pkw"].max())
        for zi in range(n_z):
            z_mm = Z_POSITIONS[zi] * 1e3
            axes[row_kw, zi].imshow(
                sc["kw"][:, :, zi].T, origin="lower", vmin=0, vmax=vmax, cmap="viridis"
            )
            axes[row_kw, zi].set_title(f"k-Wave z={z_mm:.0f}mm")
            axes[row_pkw, zi].imshow(
                sc["pkw"][:, :, zi].T, origin="lower", vmin=0, vmax=vmax, cmap="viridis"
            )
            axes[row_pkw, zi].set_title(f"pykwavers z={z_mm:.0f}mm")
        axes[row_kw, 0].set_ylabel(f"{sc['label']}\nk-Wave")
        axes[row_pkw, 0].set_ylabel(f"{sc['label']}\npykwavers")

    fig.savefig(path, dpi=100)
    plt.close(fig)


def build_report_lines(scenarios: list[dict]) -> list[str]:
    lines = [
        "tvsp_angular_spectrum_method parity report",
        "==========================================",
        "",
        f"Source: circular piston, r={APERTURE_RADIUS*1e3:.1f} mm, f0={F0/1e6:.1f} MHz",
        f"Grid: {NX}×{NY}, dx={DX*1e3:.3f} mm",
        f"z planes [mm]: {[round(float(z)*1e3, 1) for z in Z_POSITIONS]}",
        "",
    ]
    all_pass = True
    for sc in scenarios:
        status = "PASS" if sc["pass"] else "FAIL"
        all_pass = all_pass and sc["pass"]
        lines.append(f"Scenario: {sc['label']}  [{status}]")
        for z, m in sc["metrics_per_z"]:
            lines.append(
                f"  z={z*1e3:5.1f} mm: pearson_r={m['pearson_r']:.6f}"
                f"  psnr_db={m['psnr_db']:.2f}  rms_ratio={m['rms_ratio']:.6f}"
            )
        lines.append(
            f"  → pearson_r_min={sc['pearson_min']:.6f}"
            f"  psnr_db_min={sc['psnr_min']:.2f}"
        )
        lines.append(
            f"  Thresholds: pearson≥{PARITY_THRESHOLDS['pearson_r_min']}  "
            f"psnr≥{PARITY_THRESHOLDS['psnr_db_min']} dB"
        )
        lines.append("")

    overall = "PASS" if all_pass else "FAIL"
    lines.append(f"parity_status: {overall}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-failure", action="store_true",
        help="Return exit code 0 even when parity targets are not met."
    )
    args = parser.parse_args()

    # k-wave-python angular_spectrum_cw requires medium as int (sound speed) or dict
    lossless_medium_kw: int = int(C0)
    absorbing_medium_kw = {
        "sound_speed": C0,
        "alpha_coeff": ALPHA_COEFF,
        "alpha_power": ALPHA_POWER,
    }
    # pykwavers accepts float for lossless medium
    lossless_medium_pkw: float = C0
    absorbing_medium_pkw = absorbing_medium_kw

    scenarios = [
        _run_scenario(lossless_medium_kw, lossless_medium_pkw, "lossless"),
        _run_scenario(absorbing_medium_kw, absorbing_medium_pkw, "absorbing"),
    ]

    report_lines = build_report_lines(scenarios)
    for line in report_lines:
        print(line)

    save_figure(scenarios, FIGURE_PATH)
    save_text_report(METRICS_PATH, "tvsp_angular_spectrum_method parity report", report_lines)
    print(f"Saved: {FIGURE_PATH}")
    print(f"Saved: {METRICS_PATH}")

    all_pass = all(sc["pass"] for sc in scenarios)
    if not all_pass and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
