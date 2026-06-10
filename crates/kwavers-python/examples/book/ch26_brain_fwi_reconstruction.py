#!/usr/bin/env python3
"""Chapter 26 — transcranial brain-FWI reconstruction, plain vs Plug-and-Play (PnP).

Plots the **real** reconstructions produced by the Rust example
``transcranial_brain_fwi`` (run it first):

    cargo run -p kwavers-solver --example transcranial_brain_fwi

Two reconstructions are compared: plain masked FWI, and FWI with a Plug-and-Play
total-variation prior (the canonical compressed-sensing-MRI / CT-MBIR
regulariser; `kwavers_math::inverse_problems::tv_denoise_chambolle`). No physics
is recomputed in Python — it only plots the CSV grids the Rust inversion wrote.
"""
from __future__ import annotations

import pathlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[4]
DATA = REPO / "target" / "book_data" / "brain_fwi"
OUT = REPO / "docs" / "book" / "figures" / "ch27"
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
    if not (DATA / "recon_pnp.csv").exists():
        raise SystemExit(
            f"missing {DATA}; run: cargo run -p kwavers-solver --example transcranial_brain_fwi"
        )
    truth, plain, pnp = (load(n) for n in ("truth", "recon", "recon_pnp"))
    m = meta()
    dx = m.get("dx_mm", 2.2)
    ny, nx = truth.shape
    extent = [0, nx * dx, 0, ny * dx]
    norm = Normalize(vmin=1500.0, vmax=m.get("c_anomaly", 1700.0))
    removed = plain - pnp  # what the PnP prior changed (the cleaned artefacts)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.8), constrained_layout=True)
    panels = [
        (axes[0, 0], truth, "Ground truth (skull + brain + lesion)", "turbo", norm, "sound speed [m/s]"),
        (axes[0, 1], plain, f"Plain FWI  (lesion {100*m.get('lesion_peak_recovered_frac',0):.0f}%, "
                            f"RMS {m.get('brain_rms_err_m_s',0):.0f} m/s)", "turbo", norm, "sound speed [m/s]"),
        (axes[1, 0], pnp, f"FWI + PnP-TV prior  (lesion {100*m.get('pnp_lesion_peak_recovered_frac',0):.0f}%, "
                          f"RMS {m.get('pnp_rms_err_m_s',0):.0f} m/s)", "turbo", norm, "sound speed [m/s]"),
        (axes[1, 1], removed, "Plain − PnP  (artefacts removed by the prior)", "seismic",
         TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40), "Δc [m/s]"),
    ]
    for ax, field, title, cm, nm, cl in panels:
        im = ax.imshow(field, origin="lower", extent=extent, cmap=cm, norm=nm, aspect="equal")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cl)

    sup = (
        "Transcranial brain FWI with a Plug-and-Play TV prior (compressed-sensing-MRI / CT-MBIR)\n"
        f"{int(m.get('shots',0))} shots, {m.get('noise_pct',0):.0f}% noise — the prior cleans "
        "high-frequency artefacts but the residual is dominated by coherent limited-aperture streaks"
    )
    fig.suptitle(sup, fontsize=11)
    out = OUT / "fig09_brain_fwi_reconstruction.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
