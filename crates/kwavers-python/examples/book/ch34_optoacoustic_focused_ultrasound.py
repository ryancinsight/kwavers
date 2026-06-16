"""
Chapter 34 — Optically-Generated Focused Ultrasound (OFUS / SOAP).

Replicates the design relations of Li et al., Light: Sci. Appl. 11, 321 (2022)
from the kwavers Rust kernels (single source of truth):

fig01  (a) lateral resolution R_L = 0.71 ν/(NA·f) vs numerical aperture, with the
           paper's empirical 71.5/NA fit and the conventional-PZT NA band;
       (b) focal gain G = (2πf/c0)·r·(1 − √(1 − 1/4f_N²)) vs f-number, with the
           G_max ≈ 280 device point.

All physics comes from the Rust core via pykwavers — no equations are
reimplemented here (pykwavers is a PyO3 wrapper only):
  kw.acoustic_resolution_lateral, kw.soap_focal_gain,
  kw.numerical_aperture_from_geometry, kw.f_number_from_na.

References
----------
Li et al. (2022) Light: Sci. Appl. 11:321
O'Neil (1949) J. Acoust. Soc. Am. 21:516
Yao & Wang (2013) Laser Photonics Rev. 7:758
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pykwavers as kw
    _HAS_PYKWAVERS = True
except ImportError:
    kw = None
    _HAS_PYKWAVERS = False

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch34")
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch34/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.6,
})

C0 = 1500.0        # ambient sound speed [m/s]
F0 = 15.0e6        # CS-PDMS centre frequency [Hz]
R_CURV = 6.35e-3   # radius of curvature [m]
D_TRANS = 12.1e-3  # transverse aperture diameter [m]


def fig01_soap_resolution_gain() -> None:
    """Lateral resolution vs NA and focal gain vs f-number (Eqs. 34.4–34.5)."""
    if not _HAS_PYKWAVERS:
        raise ImportError("pykwavers is required for ch34 fig01 (OFUS design relations)")

    fig, (ax_r, ax_g) = plt.subplots(1, 2, figsize=(11, 4.3))

    # (a) Lateral resolution vs NA — Eq. (34.5), kw.acoustic_resolution_lateral.
    na = np.linspace(0.15, 1.0, 200)
    r_l_um = np.array([kw.acoustic_resolution_lateral(C0, n, F0) for n in na]) * 1e6
    ax_r.plot(na, r_l_um, color="#1f77b4", label="Eq. (34.5): 0.71 ν/(NA·f)")
    # empirical fit R_L[µm] = 71.5/NA reproduced by sampled markers
    na_fit = np.array([0.2, 0.3, 0.4, 0.6, 0.95, 1.0])
    ax_r.plot(na_fit, 71.5 / na_fit, "ks", ms=5, label="paper fit 71.5/NA")
    ax_r.axvspan(0.2, 0.6, color="orange", alpha=0.2, label="conventional PZT NA")
    na_dev = kw.numerical_aperture_from_geometry(R_CURV, D_TRANS)
    r_dev = kw.acoustic_resolution_lateral(C0, na_dev, F0) * 1e6
    ax_r.plot([na_dev], [r_dev], "r*", ms=14, label=f"CS-PDMS NA={na_dev:.2f} → {r_dev:.0f} µm")
    ax_r.set_xlabel("Numerical aperture NA")
    ax_r.set_ylabel("Lateral resolution $R_L$ (µm)")
    ax_r.set_title("(a) Lateral resolution vs NA (15 MHz)")
    ax_r.set_ylim(0, 360)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend()

    # (b) Focal gain vs f-number — Eq. (34.4), kw.soap_focal_gain.
    f_n = np.linspace(0.5, 2.0, 200)
    g = np.array([kw.soap_focal_gain(F0, C0, R_CURV, fn) for fn in f_n])
    ax_g.plot(f_n, g, color="#d62728")
    f_n_dev = kw.f_number_from_na(na_dev)
    g_dev = kw.soap_focal_gain(F0, C0, R_CURV, f_n_dev)
    ax_g.plot([f_n_dev], [g_dev], "r*", ms=14,
              label=f"CS-PDMS $f_N$={f_n_dev:.2f} → G={g_dev:.0f}")
    ax_g.axhline(280, color="k", ls="--", lw=1, label="paper $G_{max}$ ≈ 280")
    ax_g.set_xlabel("f-number $f_N = r/D_t$")
    ax_g.set_ylabel("Focal gain $G$")
    ax_g.set_title("(b) Focal gain vs f-number")
    ax_g.grid(True, alpha=0.3)
    ax_g.legend()

    fig.tight_layout()
    savefig("fig01_soap_resolution_gain")


if __name__ == "__main__":
    fig01_soap_resolution_gain()
