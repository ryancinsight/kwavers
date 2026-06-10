#!/usr/bin/env python3
"""Chapter 26 — guidance-free transcranial-UST skull/brain reconstruction figure.

Renders the MOFI exact-adjoint alignment produced by the Rust example
``transcranial_ust_reconstruction`` (run it first):

    cargo run -p kwavers-solver --example transcranial_ust_reconstruction

The Rust example writes real reconstruction CSV grids to
``target/book_data/transcranial/``; this script only loads and plots them — no
physics is recomputed in Python (the MOFI/self-adjoint engine lives in Rust).
"""
from __future__ import annotations

import pathlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[4]  # …/crates/kwavers-python/examples/book/<file> → repo root
DATA = REPO / "target" / "book_data" / "transcranial"
OUT = REPO / "docs" / "book" / "figures" / "ch27"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> np.ndarray:
    return np.loadtxt(DATA / f"{name}.csv", delimiter=",")


def load_meta() -> dict[str, float]:
    meta: dict[str, float] = {}
    for line in (DATA / "meta.csv").read_text().splitlines():
        if "," in line:
            k, v = line.split(",", 1)
            try:
                meta[k] = float(v)
            except ValueError:
                pass
    return meta


def main() -> None:
    if not (DATA / "patient.csv").exists():
        raise SystemExit(
            f"missing {DATA}; run: cargo run -p kwavers-solver "
            "--example transcranial_ust_reconstruction"
        )
    patient = load("patient")
    template = load("template_initial")
    aligned = load("mofi_aligned")
    error = load("error")
    meta = load_meta()
    dx_mm = meta.get("dx_mm", 2.0)
    ny, nx = patient.shape
    extent = [0, nx * dx_mm, 0, ny * dx_mm]  # mm

    # Sound-speed colour scale spanning water → brain → skull.
    norm = Normalize(vmin=1480.0, vmax=meta.get("c_skull", 2600.0))
    cmap = "turbo"

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.6), constrained_layout=True)
    panels = [
        (axes[0, 0], patient, "Patient (ground truth)\nskull + brain at unknown pose", cmap, norm),
        (axes[0, 1], template, "CT template — initial (misaligned)", cmap, norm),
        (axes[1, 0], aligned, "MOFI reconstruction (guidance-free)", cmap, norm),
        (axes[1, 1], error, "|reconstruction − patient|  [m/s]", "magma", Normalize(0, 200)),
    ]
    for ax, field, title, cm, nm in panels:
        im = ax.imshow(field, origin="lower", extent=extent, cmap=cm, norm=nm, aspect="equal")
        # Skull outline (template) for visual registration reference.
        ax.contour(
            np.linspace(0, nx * dx_mm, nx),
            np.linspace(0, ny * dx_mm, ny),
            patient if title.startswith("Patient") else aligned if "MOFI" in title else template,
            levels=[2000.0],
            colors="white",
            linewidths=0.6,
            alpha=0.6,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="sound speed [m/s]" if cm == cmap else "|Δc| [m/s]")

    sup = (
        "Guidance-free transcranial-UST alignment (MOFI, exact-adjoint engine)\n"
        f"recovered θ = {meta.get('theta_rec_deg', float('nan')):.2f}° "
        f"(true {meta.get('theta_true_deg', float('nan')):.2f}°), "
        f"|Δθ| = {meta.get('dtheta_deg', float('nan')):.3f}°, "
        f"|Δδ| = {meta.get('dtrans_mm', float('nan')):.3f} mm"
    )
    fig.suptitle(sup, fontsize=12)

    out = OUT / "fig08_mofi_skull_brain_reconstruction.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
