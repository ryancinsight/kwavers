#!/usr/bin/env python3
"""Chapter 11 (fig07) — elastic shear-wave FWI: stiff-lesion reconstruction.

Plots the **real** shear-modulus maps produced by the Rust example
``elastic_shear_fwi_lesion`` (run it first):

    cargo run -p kwavers-solver --release --example elastic_shear_fwi_lesion

The adjoint-state elastic full-waveform inversion
(`kwavers_solver::inverse::elastography::elastic_fwi::ElasticFwi`, ADR 033)
reconstructs the Lamé shear modulus mu(x) from crossed four-side transmission
shear-wave data. No physics is recomputed in Python — it only plots the CSV grids
the Rust inversion wrote (the "physics in Rust, plotting in Python" contract).
"""
from __future__ import annotations

import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[4]
DATA = REPO / "target" / "book_data" / "elastic_fwi"
OUT = REPO / "docs" / "book" / "figures" / "ch10"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> np.ndarray:
    return np.loadtxt(DATA / f"{name}.csv", delimiter=",")


def meta() -> dict[str, float]:
    d: dict[str, float] = {}
    for line in (DATA / "meta.csv").read_text().splitlines():
        if "," in line:
            k, v = line.split(",", 1)
            try:
                d[k] = float(v)
            except ValueError:
                pass
    return d


def main() -> None:
    if not (DATA / "mu_recovered.csv").exists():
        raise SystemExit(
            f"missing {DATA}; run: "
            "cargo run -p kwavers-solver --release --example elastic_shear_fwi_lesion"
        )
    mu_true, mu_init, mu_rec = (load(n) for n in ("mu_true", "mu_initial", "mu_recovered"))
    m = meta()
    dx_mm = 1.0e3 * m.get("dx_m", 1.0e-3)
    mu_bg_kpa = m.get("mu_bg_pa", 4000.0) / 1.0e3
    mu_incl_kpa = m.get("mu_incl_pa", 12000.0) / 1.0e3

    # Crop to the imaged interior — the region the inversion targets — excluding the
    # PML and the source/receiver ring (FWI maps conventionally omit the absorbing
    # boundary, which is not reconstructed).
    lo, hi = 8, 28
    mu_true, mu_init, mu_rec = (a[lo:hi, lo:hi] for a in (mu_true, mu_init, mu_rec))
    ny, nx = mu_true.shape
    extent = [0, nx * dx_mm, 0, ny * dx_mm]
    norm = Normalize(vmin=mu_bg_kpa * 0.9, vmax=mu_incl_kpa * 1.05)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)

    for ax, field, title in (
        (axes[0], mu_true, "True shear modulus mu (3x stiff lesion)"),
        (axes[1], mu_rec, "Elastic-FWI reconstruction"),
    ):
        # 'gaussian' display interpolation renders the discrete grid smoothly
        # (a standard field-visualization choice); the quantitative recovery is
        # shown raw in the profile panel.
        im = ax.imshow(
            field / 1.0e3,
            origin="lower",
            extent=extent,
            cmap="inferno",
            norm=norm,
            aspect="equal",
            interpolation="gaussian",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mu [kPa]")

    # Horizontal profile through the lesion centre: true vs initial vs recovered.
    row = ny // 2
    x_mm = np.arange(nx) * dx_mm
    axes[2].plot(x_mm, mu_true[row] / 1.0e3, "k-", lw=2.0, label="true")
    axes[2].plot(x_mm, mu_init[row] / 1.0e3, color="0.6", ls=":", lw=1.5, label="initial (background)")
    axes[2].plot(x_mm, mu_rec[row] / 1.0e3, "C3-", lw=1.8, label="elastic FWI")
    axes[2].set_title("Profile through lesion centre", fontsize=11)
    axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("mu [kPa]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig07_elastic_fwi_lesion.{ext}", dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch10/fig07_elastic_fwi_lesion.{{pdf,png}}")


if __name__ == "__main__":
    main()
