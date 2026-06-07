"""Chapter 33: CMUT vs PMUT — Figure Generation Script.

All physics is computed by the Rust core (`kwavers_transducer::mems`) through the
pykwavers bindings; this script only plots. Rebuild pykwavers (`maturin develop
--release`) after changing the Rust models.

  fig01  Clamped-plate resonance vs geometry (vacuum + in-blood)
  fig02  Electrical: CMUT collapse voltage / bias coupling, PMUT coupling (AlN vs PZT)
  fig03  Self-heating: CMUT vs PMUT-AlN vs PMUT-PZT
  fig04  Fractional bandwidth vs fluid-loading ratio (axial resolution)
  fig05  IVUS figure-of-merit comparison
"""

import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import pykwavers as kw
except ImportError as exc:  # pragma: no cover
    raise ImportError("pykwavers is required (maturin develop --release)") from exc

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch33")
os.makedirs(OUT_DIR, exist_ok=True)

BLOOD_RHO = 1060.0  # kg/m^3
# Silicon membrane constants (match the Rust CmutCell::silicon preset)
SI_E, SI_NU, SI_RHO = 169e9, 0.22, 2330.0


def savefig(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch33/{name}.{{png,pdf}}")
    plt.close()


def fig01_resonance_geometry() -> None:
    radii_um = np.linspace(8, 40, 60)
    fig, ax = plt.subplots(figsize=(6, 4))
    for h_um, style in [(0.4, "-"), (0.8, "--"), (1.2, ":")]:
        f_vac = np.array([
            kw.mems_clamped_plate_resonance(SI_E, h_um * 1e-6, SI_NU, SI_RHO, a * 1e-6) / 1e6
            for a in radii_um
        ])
        f_imm = np.array([
            kw.mems_immersion_resonance(fv * 1e6, SI_RHO, h_um * 1e-6, BLOOD_RHO, a * 1e-6) / 1e6
            for fv, a in zip(f_vac, radii_um)
        ])
        ax.plot(radii_um, f_vac, style, color="tab:blue", label=f"vacuum, h={h_um} µm")
        ax.plot(radii_um, f_imm, style, color="tab:red", label=f"blood, h={h_um} µm")
    ax.axhspan(20, 60, color="green", alpha=0.08, label="IVUS band")
    ax.set_xlabel("membrane radius a (µm)")
    ax.set_ylabel("fundamental f₀ (MHz)")
    ax.set_title("Clamped Si plate resonance (f₀ ∝ h/a²)")
    ax.set_ylim(0, 120)
    ax.legend(fontsize=7, ncol=2)
    savefig("fig01_resonance_geometry")


def fig02_electrical() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 4))
    # CMUT collapse voltage vs gap + bias coupling
    gaps_nm = np.linspace(80, 400, 40)
    vc = np.array([kw.cmut_collapse_voltage(14e-6, 0.4e-6, g * 1e-9) for g in gaps_nm])
    axl.plot(gaps_nm, vc, "tab:blue")
    axl.set_xlabel("vacuum gap g₀ (nm)")
    axl.set_ylabel("collapse voltage V_c (V)", color="tab:blue")
    axl.set_title("CMUT: collapse voltage (∝ g^1.5) & bias coupling")
    ax2 = axl.twinx()
    vc25 = kw.cmut_collapse_voltage(14e-6, 0.4e-6, 0.25e-6)
    bias = np.linspace(0, vc25, 40)
    k2 = np.array([kw.cmut_coupling_k2(14e-6, 0.4e-6, 0.25e-6, v) for v in bias])
    ax2.plot(bias / vc25, k2, "tab:red")
    ax2.set_ylabel("k² (bias, gap=250 nm)", color="tab:red")
    # PMUT coupling AlN vs PZT
    for film, col in [("aln", "tab:green"), ("pzt", "tab:purple")]:
        k = kw.pmut_coupling_k2(film, 20e-6, 1e-6, 2e-6)
        axr.bar(film.upper(), k, color=col)
    axr.set_ylabel("effective coupling k²")
    axr.set_title("PMUT coupling: AlN vs PZT (fixed by film)")
    savefig("fig02_electrical")


def fig03_heating() -> None:
    v_ac = np.linspace(0, 20, 40)
    f0 = 40e6
    fig, ax = plt.subplots(figsize=(6, 4))
    p_cmut = np.array([kw.cmut_self_heating(14e-6, 0.4e-6, 0.25e-6, v, f0) for v in v_ac])
    p_aln = np.array([kw.pmut_self_heating("aln", 20e-6, 1e-6, 2e-6, v, f0) for v in v_ac])
    p_pzt = np.array([kw.pmut_self_heating("pzt", 20e-6, 1e-6, 2e-6, v, f0) for v in v_ac])
    ax.plot(v_ac, p_cmut * 1e3, label="CMUT (tanδ≈1e-3)")
    ax.plot(v_ac, p_aln * 1e3, label="PMUT-AlN (tanδ≈3e-3)")
    ax.plot(v_ac, p_pzt * 1e3, label="PMUT-PZT (tanδ≈2e-2)")
    ax.set_xlabel("AC drive voltage (V)")
    ax.set_ylabel("dielectric self-heating (mW)")
    ax.set_title("Self-heating at 40 MHz (P = π f C V² tanδ)")
    ax.legend()
    savefig("fig03_heating")


def fig04_bandwidth() -> None:
    radii_um = np.linspace(8, 40, 60)
    fig, ax = plt.subplots(figsize=(6, 4))
    fbw_cmut = np.array([kw.cmut_fractional_bandwidth(a * 1e-6, 0.4e-6, BLOOD_RHO) for a in radii_um])
    fbw_pmut = np.array([kw.pmut_fractional_bandwidth("pzt", a * 1e-6, 1e-6, 2e-6, BLOOD_RHO) for a in radii_um])
    ax.plot(radii_um, fbw_cmut * 100, label="CMUT (light membrane)")
    ax.plot(radii_um, fbw_pmut * 100, label="PMUT-PZT (composite plate)")
    ax.set_xlabel("radius a (µm)")
    ax.set_ylabel("fractional bandwidth (%)")
    ax.set_title("Bandwidth from fluid loading → axial resolution")
    ax.legend()
    savefig("fig04_bandwidth")


def fig05_ivus_fom() -> None:
    v = kw.ivus_figure_of_merit(14e-6, 0.4e-6, 0.25e-6, "pzt", 20e-6, 1e-6, 2e-6, BLOOD_RHO, 5.0)
    cmut_fbw, pmut_fbw, cmut_heat, pmut_heat, cmut_v, pmut_v, cmut_fom, pmut_fom, rec = v
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 4))
    # normalized per-criterion scores (higher = better)
    crits = ["bandwidth", "thermal", "drive V", "integration"]
    cmut_scores = [cmut_fbw / max(cmut_fbw, pmut_fbw),
                   min(cmut_heat, pmut_heat) / cmut_heat,
                   min(cmut_v, pmut_v) / cmut_v, 1.0]
    pmut_scores = [pmut_fbw / max(cmut_fbw, pmut_fbw),
                   min(cmut_heat, pmut_heat) / pmut_heat,
                   min(cmut_v, pmut_v) / pmut_v, 0.7]
    x = np.arange(len(crits))
    axl.bar(x - 0.2, cmut_scores, 0.4, label="CMUT")
    axl.bar(x + 0.2, pmut_scores, 0.4, label="PMUT-PZT")
    axl.set_xticks(x)
    axl.set_xticklabels(crits, rotation=20)
    axl.set_ylabel("per-criterion score")
    axl.set_title("IVUS criteria (higher = better)")
    axl.legend()
    winner = "CMUT" if rec == 0.0 else "PMUT"
    axr.bar(["CMUT", "PMUT-PZT"], [cmut_fom, pmut_fom], color=["tab:blue", "tab:purple"])
    axr.set_ylabel("weighted IVUS figure of merit")
    axr.set_title(f"IVUS verdict: {winner} preferred")
    savefig("fig05_ivus_fom")


def fig06_therapy_output() -> None:
    # Therapy-scale designs (~2-5 MHz) in water.
    rho, c = 1000.0, 1500.0
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 4))
    # output pressure vs drive: CMUT (gap-limited ceiling) vs PMUT (scales)
    drives = np.linspace(0, 60, 40)
    p_cmut = np.array([
        kw.cmut_max_output_pressure(60e-6, 2e-6, 0.2e-6, rho, c, 1.0 / 3.0) for _ in drives
    ])  # gap-limited: independent of drive
    p_pmut = np.array([
        kw.pmut_max_output_pressure("pzt", 60e-6, 2e-6, 4e-6, v, rho, c) for v in drives
    ])
    axl.plot(drives, p_cmut / 1e6, label="CMUT (gap-limited ceiling)")
    axl.plot(drives, p_pmut / 1e6, label="PMUT-PZT (∝ drive)")
    axl.set_xlabel("drive voltage (V)")
    axl.set_ylabel("peak output pressure (MPa)")
    axl.set_title("Therapy output: CMUT saturates, PMUT scales")
    axl.legend()
    # CMUT flex penalty vs curvature for several gaps
    kappa = np.linspace(0, 1.0 / 1.0e-3, 40)  # up to 1 mm radius of curvature
    for g_nm, col in [(100, "tab:red"), (200, "tab:orange"), (400, "tab:green")]:
        d = np.array([kw.cmut_flex_gap_derating(60e-6, 2e-6, g_nm * 1e-9, k) for k in kappa])
        axr.plot(1.0 / np.maximum(kappa, 1e-9) * 1e3, d, col, label=f"gap={g_nm} nm")
    axr.set_xlabel("radius of curvature (mm)")
    axr.set_ylabel("output derating η_flex")
    axr.set_title("Flexing a CMUT cuts output (tighter gap → worse)")
    axr.set_xlim(0.5, 10)
    axr.legend()
    savefig("fig06_therapy_output")


def main() -> None:
    print("Generating Chapter 33 figures (CMUT vs PMUT)...")
    fig01_resonance_geometry()
    fig02_electrical()
    fig03_heating()
    fig04_bandwidth()
    fig05_ivus_fom()
    fig06_therapy_output()
    print("Done. Output: docs/book/figures/ch33/")


if __name__ == "__main__":
    main()
