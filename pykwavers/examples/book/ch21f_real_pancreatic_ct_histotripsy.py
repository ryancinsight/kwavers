"""
Chapter 21f: Histotripsy planning on a real pancreatic cancer CT — PDAC
=======================================================================

Downloads a real patient CT from the Medical Segmentation Decathlon
Task07_Pancreas archive (Antonelli 2022, CC-BY-SA 4.0) by streaming only
the first ~28 MB of the archive.  All MSD Task07 cases are histologically
confirmed pancreatic ductal adenocarcinoma (PDAC) staging CT scans acquired
in the portal venous phase.  The archive's first entry is a test-set CT
(no released labels), so the script computes a CECT-specific
auto-segmentation directly from HU values to identify the pancreatic
parenchyma and the hypoenhancing PDAC mass within it.

All wave propagation, operator construction, cavitation source modelling,
thermal computation, raster planning, and multi-channel inversion are
delegated to Rust via pykwavers (kw.run_theranostic_inverse_from_ritk).
Python handles CT data I/O, morphological auto-segmentation, matplotlib
visualization, and output file writing only.

Dataset:
  Task07_Pancreas — Medical Segmentation Decathlon
  Reference : Antonelli et al. 2022, Nature Commun 13:4128
  License   : CC-BY-SA 4.0
  Source    : https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar
  Archive   : first data entry is imagesTs/pancreas_044.nii.gz (~28 MB)
  Labels    : auto-generated from portal-venous-phase HU priors (see below)

Auto-segmentation (portal venous phase CECT):
  Body / skin    : morphological boundary extraction
  Fat            : HU −150 to −80
  Muscle         : HU  30 to 55
  Bone           : HU > 250
  Vertebral col. : largest posterior bone structure → depth reference
  Pancreas       : retroperitoneal soft-tissue cluster HU 55–130,
                   0–80 mm anterior to vertebrae, size 3 000–150 000 vox
  PDAC target    : hypo-enhancing sub-volume within pancreatic region
                   (cluster with HU ≥ 15 below local parenchyma mean)

Acoustic access: anterior subcostal window through the right hepatic lobe.

Pipeline:
    1. Stream ~28 MB of MSD archive → real patient PDAC CT (NIfTI).
    2. Reorient to canonical RAS; crop to pancreatic region with margin.
    3. Resample to 1.2 mm isotropic.
    4. Auto-segment tissue labels from CECT HU values (scipy.ndimage only).
    5. Delegate all physics to Rust via kw.run_theranostic_inverse_from_ritk.
    6. Render figures and write scenario_metrics.json.

Outputs:
    docs/book/figures/ch21f/*.{png,pdf}
    docs/book/figures/ch21f/scenario_metrics.json

Physical references:
    Mauch et al.          2017  IEEE TUFFC  64(9):1386
    Chen et al.           2022  Ultrasound Med Biol 48(6):1002
    Vlaisavljevich et al. 2015  JASA 138(4):1864
    Vlaisavljevich et al. 2016  JASA 140(5):3504
    Maxwell et al.        2013  JASA 134(3):1765
    Duck                  1990  Physical Properties of Tissue
    IT'IS Foundation      2022  Tissue Properties Database v4.1
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (binary_dilation, binary_erosion, binary_closing,
                           distance_transform_edt, label as ndlabel, zoom)

# ───────────────────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
MSD_DIR   = REPO_ROOT / "data" / "msd_pancreas_sample"
CT_PATH   = MSD_DIR / "ct_ts.nii.gz"
OUT_DIR   = REPO_ROOT / "docs" / "book" / "figures" / "ch21f"

MSD_ARCHIVE_URL = (
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"
)
_MSD_STREAM_BYTES = 60 * 1024 * 1024

# ───────────────────────────────────────────────────────────────────────
# pykwavers extension discovery
# ───────────────────────────────────────────────────────────────────────

_PY_PACKAGE = REPO_ROOT / "pykwavers" / "python"
if "PYKWAVERS_EXTENSION_PATH" not in os.environ:
    for _cand in (
        REPO_ROOT / "target" / "release" / "pykwavers.dll",
        REPO_ROOT / "target" / "maturin" / "pykwavers.dll",
        REPO_ROOT / "target" / "debug" / "pykwavers.dll",
    ):
        if _cand.exists():
            os.environ["PYKWAVERS_EXTENSION_PATH"] = str(_cand)
            break
if str(_PY_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_PY_PACKAGE))

import pykwavers as kw  # noqa: E402

# ───────────────────────────────────────────────────────────────────────
# Run-time constants (overrideable via environment variables)
# ───────────────────────────────────────────────────────────────────────

GRID_SIZE     = int(os.environ.get("KWAVERS_CH21F_GRID_SIZE",     "96"))
ELEMENT_COUNT = int(os.environ.get("KWAVERS_CH21F_ELEMENT_COUNT", "256"))
ITERATIONS    = int(os.environ.get("KWAVERS_CH21F_ITERATIONS",    "18"))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
})


# ───────────────────────────────────────────────────────────────────────
# Tissue metadata (documentation / visualization only — no physics)
# Duck 1990 / IT'IS v4.1 / Mast 2000.
# ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Tissue:
    label: int
    name: str
    rho: float        # kg/m³
    c: float          # m/s
    alpha0: float     # dB/cm at 1 MHz
    y_pow: float      # frequency power-law exponent
    cp: float         # J/(kg·K)
    kappa: float      # W/(m·K)
    perfusion: float  # mL/(100 g·min)


AIR      = Tissue(0, "air",         1.2,   343.0,   0.0,  1.0, 1005.0, 0.026, 0.0)
SKIN     = Tissue(1, "skin",     1109.0,  1624.0,  21.2,  1.1, 3391.0, 0.37,  1.06)
FAT      = Tissue(2, "fat",       911.0,  1440.0,   4.8,  1.1, 2348.0, 0.21,  0.43)
MUSCLE   = Tissue(3, "muscle",   1090.0,  1588.0,   8.1,  1.1, 3421.0, 0.49,  0.67)
BONE     = Tissue(4, "bone",     1908.0,  4080.0, 250.0,  1.0, 1313.0, 0.32,  0.10)
LIVER    = Tissue(5, "liver",    1079.0,  1595.0,   8.7,  1.1, 3540.0, 0.52,  6.4)
PANCREAS = Tissue(6, "pancreas", 1040.0,  1543.0,   6.0,  1.1, 3513.0, 0.51,  3.2)
PDAC     = Tissue(7, "pdac",     1060.0,  1555.0,   8.5,  1.1, 3480.0, 0.49,  2.1)

TISSUES = [AIR, SKIN, FAT, MUSCLE, BONE, LIVER, PANCREAS, PDAC]


# ───────────────────────────────────────────────────────────────────────
# Output helper
# ───────────────────────────────────────────────────────────────────────


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(OUT_DIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch21f/{name}.{{pdf,png}}")


# ───────────────────────────────────────────────────────────────────────
# MSD archive streaming
# ───────────────────────────────────────────────────────────────────────


class _BoundedStream(io.RawIOBase):
    """Wrap an HTTP response and stop after `max_bytes`."""

    def __init__(self, response, max_bytes: int) -> None:
        self._resp = response
        self._remaining = max_bytes

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        if self._remaining <= 0:
            return 0
        chunk = min(len(b), self._remaining, 65536)
        data = self._resp.read(chunk)
        if not data:
            return 0
        n = len(data)
        b[:n] = data
        self._remaining -= n
        return n


def _download_first_msd_ct(out_dir: Path) -> Path:
    """Stream the first ~60 MB of the MSD Task07_Pancreas archive and
    extract the first imagesTs/*.nii.gz entry into out_dir.

    Returns the local path to the extracted NIfTI file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ct_ts.nii.gz"
    if out_path.is_file():
        print(f"[ch21f] Using cached CT: {out_path}")
        return out_path

    print(f"[ch21f] Streaming MSD archive (first {_MSD_STREAM_BYTES // (1024*1024)} MB)...")
    print(f"[ch21f]   Source: {MSD_ARCHIVE_URL}")

    req = urllib.request.Request(
        MSD_ARCHIVE_URL,
        headers={"User-Agent": "kwavers-ch21f/1.0 (academic histotripsy simulation)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        bounded = io.BufferedReader(_BoundedStream(resp, _MSD_STREAM_BYTES), buffer_size=65536)
        with tarfile.open(fileobj=bounded, mode="r|") as tf:
            for member in tf:
                if (member.name.endswith(".nii.gz")
                        and "imagesTs" in member.name
                        and not member.name.endswith(".DS_Store")):
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                    out_path.write_bytes(data)
                    size_mb = len(data) / (1024 * 1024)
                    print(f"[ch21f]   Extracted {member.name} ({size_mb:.1f} MB) -> {out_path}")
                    return out_path
    raise RuntimeError(
        "[ch21f] No imagesTs/*.nii.gz entry found in the first "
        f"{_MSD_STREAM_BYTES // (1024*1024)} MB of the MSD archive. "
        "The archive structure may have changed."
    )


# ───────────────────────────────────────────────────────────────────────
# CECT auto-segmentation (portal venous phase HU priors)
# Morphological operations only (scipy.ndimage); no wave physics.
# ───────────────────────────────────────────────────────────────────────


def _segment_cect(ct_hu: np.ndarray, dx_m: float) -> np.ndarray:
    """Auto-segment a portal venous phase abdominal CECT into tissue labels.

    HU thresholds follow Duck 1990 and published CECT pancreatic imaging
    literature (Macari & Belfi 2011, Chu et al. 2019 Abdom Radiol).

    Returns
    -------
    label_vol : (Nx, Ny, Nz) int8 tissue label integers.
    """
    nx, ny, nz = ct_hu.shape
    label_vol = np.zeros_like(ct_hu, dtype=np.int8)

    # Coarse HU thresholds.
    label_vol[ct_hu < -500]                              = AIR.label
    label_vol[(ct_hu >= -500) & (ct_hu < -30)]          = FAT.label
    label_vol[(ct_hu >= -30)  & (ct_hu < 400)]          = MUSCLE.label
    label_vol[ct_hu >= 400]                              = BONE.label

    # Skin shell (2 mm morphological boundary).
    body = label_vol != AIR.label
    skin_vox = max(int(round(2.0e-3 / dx_m)), 1)
    skin_shell = body & binary_dilation(~body, iterations=skin_vox)
    label_vol[skin_shell] = SKIN.label

    # Locate vertebral column.
    bone_mask = label_vol == BONE.label
    vert_col_ap = max(0, nx * 3 // 5)
    vert_col_cx = ny // 2
    if bone_mask.any():
        bone_coords = np.argwhere(bone_mask)
        rl_dev = np.abs(bone_coords[:, 1] - ny // 2)
        midline_bone = bone_coords[rl_dev < max(ny // 5, 1)]
        ap_cut = int(nx * 0.82)
        midline_interior = midline_bone[midline_bone[:, 0] < ap_cut]
        if len(midline_interior) > 20:
            midline_bone = midline_interior
        if len(midline_bone) > 0:
            vert_col_ap = int(np.percentile(midline_bone[:, 0], 20))
            vert_col_cx = int(midline_bone[:, 1].mean())

    # Liver: HU 55-175, large component, anterior of retroperitoneal zone.
    liver_ap_max = max(0, vert_col_ap - int(round(30.0e-3 / dx_m)))
    liver_cand = np.zeros(ct_hu.shape, dtype=bool)
    liver_cand[:liver_ap_max, :, :] = (
        (ct_hu[:liver_ap_max, :, :] >= 55)
        & (ct_hu[:liver_ap_max, :, :] < 175)
        & (label_vol[:liver_ap_max, :, :] == MUSCLE.label)
    )
    liver_lbl, _ = ndlabel(liver_cand)
    liver_sizes = np.bincount(liver_lbl.ravel())
    liver_sizes[0] = 0
    for comp_id in np.where(liver_sizes >= 20_000)[0]:
        label_vol[liver_lbl == comp_id] = LIVER.label

    # Retroperitoneal zone for pancreatic candidate search.
    depth_110mm = int(round(110.0e-3 / dx_m))
    margin_30mm = int(round(30.0e-3  / dx_m))
    retro_start = max(0, vert_col_ap - depth_110mm)
    retro_end   = min(nx, vert_col_ap + margin_30mm)
    rl_90mm     = int(round(90.0e-3 / dx_m))
    rl_lo = max(0, vert_col_cx - rl_90mm)
    rl_hi = min(ny, vert_col_cx + rl_90mm)
    retro_zone = np.zeros(ct_hu.shape, dtype=bool)
    retro_zone[retro_start:retro_end, rl_lo:rl_hi, :] = True

    # Pancreatic candidates: HU 55-200, retroperitoneal, unlabeled.
    panc_cand = (
        (ct_hu >= 55) & (ct_hu <= 200)
        & retro_zone
        & (label_vol == MUSCLE.label)
    )
    panc_lbl, _ = ndlabel(panc_cand)
    panc_sizes = np.bincount(panc_lbl.ravel())
    panc_sizes[0] = 0

    best_panc_id = 0
    best_panc_score = -1e30
    for comp_id in np.where((panc_sizes >= 1_000) & (panc_sizes <= 80_000))[0]:
        comp_coords = np.argwhere(panc_lbl == comp_id)
        rl_dist = float(np.abs(comp_coords[:, 1].mean() - vert_col_cx))
        ap_pos  = float(comp_coords[:, 0].mean())
        ap_ideal = float(vert_col_ap) - float(depth_110mm) / 2.0
        score = -rl_dist / max(ny, 1) - abs(ap_pos - ap_ideal) / max(nx, 1)
        if score > best_panc_score:
            best_panc_score = score
            best_panc_id = comp_id

    if best_panc_id > 0:
        panc_mask_final = (panc_lbl == best_panc_id)
        label_vol[panc_mask_final] = PANCREAS.label
        panc_hu_vals = ct_hu[panc_mask_final]
        panc_mean = float(panc_hu_vals.mean())
        pdac_threshold = panc_mean - 15.0
        pdac_cand = panc_mask_final & (ct_hu <= pdac_threshold) & (ct_hu >= 15.0)
        if pdac_cand.sum() > 50:
            pdac_lbl, _ = ndlabel(pdac_cand)
            pdac_sizes = np.bincount(pdac_lbl.ravel())
            pdac_sizes[0] = 0
            if pdac_sizes.max() > 50:
                label_vol[pdac_lbl == int(pdac_sizes.argmax())] = PDAC.label

    # Anatomical fallback target when HU segmentation does not resolve PDAC.
    if not (label_vol == PDAC.label).any():
        print("[ch21f] auto-seg: PDAC not resolved -- using anatomical fallback target")
        ap_fallback = max(0, vert_col_ap - int(round(30.0e-3 / dx_m)))
        rl_fallback = vert_col_cx
        si_fallback = nz // 2
        r_pdac = int(round(15.0e-3 / dx_m))
        r_panc = int(round(28.0e-3 / dx_m))
        xi = np.arange(nx)[:, None, None]
        yi = np.arange(ny)[None, :, None]
        zi = np.arange(nz)[None, None, :]
        dist2 = ((xi - ap_fallback) ** 2 + (yi - rl_fallback) ** 2
                 + (zi - si_fallback) ** 2)
        panc_sphere = dist2 <= r_panc ** 2
        pdac_sphere = dist2 <= r_pdac ** 2
        label_vol[panc_sphere & (label_vol == MUSCLE.label)] = PANCREAS.label
        label_vol[pdac_sphere] = PDAC.label

    return label_vol


# ───────────────────────────────────────────────────────────────────────
# CT loading (data I/O only)
# ───────────────────────────────────────────────────────────────────────


def _load_ct(target_dx_m: float = 1.2e-3) -> tuple[Path, np.ndarray]:
    """Download (or load cached) MSD Task07 CT, resample, and auto-segment.

    Returns the NIfTI path and the label volume for visualization only.
    The Rust pipeline reads the NIfTI directly from disk.
    """
    ct_path = _download_first_msd_ct(MSD_DIR)

    try:
        import ritk
    except ImportError as exc:
        raise ImportError(
            "[ch21f] ritk is required for CT loading. Install with: pip install ritk"
        ) from exc

    print(f"[ch21f] Loading CT: {ct_path}")
    img = ritk.io.read_image(str(ct_path))
    raw = img.to_numpy().astype(np.float32)   # (NS, NA, NR) RAS+
    spacing = img.spacing                      # (dS_mm, dA_mm, dR_mm)
    print(f"  shape={raw.shape}, spacing={tuple(round(float(s),2) for s in spacing)} mm")

    # Remap (S, A, R) -> (AP, RL, SI) simulation coordinate order.
    raw = np.transpose(raw, (1, 2, 0))
    raw = raw[::-1, :, :].copy()
    zooms = (float(spacing[1]), float(spacing[2]), float(spacing[0]))

    # Crop to pancreatic region (200 AP × 240 RL × 120 SI mm).
    ap_vox = int(round(200.0 / zooms[0]))
    rl_vox = int(round(240.0 / zooms[1]))
    si_vox = int(round(120.0 / zooms[2]))
    nx0, ny0, nz0 = raw.shape
    x0 = max(0, nx0 // 4)
    x1 = min(nx0, x0 + ap_vox)
    y0 = max(0, (ny0 - rl_vox) // 2)
    y1 = min(ny0, y0 + rl_vox)
    z0 = max(0, (nz0 - si_vox) // 2)
    z1 = min(nz0, z0 + si_vox)
    raw = raw[x0:x1, y0:y1, z0:z1]

    # Resample to target isotropic resolution.
    target_dx_mm = target_dx_m * 1e3
    factors = tuple(z / target_dx_mm for z in zooms)
    ct_hu = zoom(raw, factors, order=1, prefilter=False).astype(np.float32)
    print(f"  resampled to {ct_hu.shape} ({target_dx_mm:.1f} mm isotropic)")

    label_vol = _segment_cect(ct_hu, target_dx_m)
    return ct_path, label_vol


# ───────────────────────────────────────────────────────────────────────
# Plot helpers (visualization only)
# ───────────────────────────────────────────────────────────────────────


def _robust_vmax(*arrays: np.ndarray, percentile: float = 97.0) -> float:
    values = np.concatenate([np.abs(a).ravel() for a in arrays])
    pos = values[values > 0.0]
    if pos.size == 0:
        return 1.0
    return float(min(max(np.percentile(pos, percentile), 1.0), 400.0))


def _show_ct(ax: plt.Axes, ct: np.ndarray, organ: np.ndarray, target: np.ndarray) -> None:
    ax.imshow(ct.T, cmap="gray", origin="lower", vmin=-150, vmax=250)
    ax.contour(organ.T, levels=[0.5], colors=["cyan"], linewidths=1.0)
    ax.contour(target.T, levels=[0.5], colors=["yellow"], linewidths=1.4)
    ax.set_xticks([])
    ax.set_yticks([])


def _show_speed(
    ax: plt.Axes,
    field: np.ndarray,
    title: str,
    mask: np.ndarray,
    vmin: float,
    vmax: float,
) -> None:
    im = ax.imshow(field.T, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    ax.contour(mask.T, levels=[0.5], colors=["white"], linewidths=0.8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m/s")


def _show_field(
    ax: plt.Axes,
    field: np.ndarray,
    title: str,
    label: str,
    *,
    vmax: float,
    cmap: str = "coolwarm",
) -> None:
    im = ax.imshow(field.T, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)


def _show_passive(
    ax: plt.Axes,
    field: np.ndarray,
    title: str,
    target: np.ndarray,
    metrics: dict,
) -> None:
    vmax = _robust_vmax(field)
    im = ax.imshow(field.T, cmap="inferno", origin="lower", vmin=0.0, vmax=max(vmax, 1e-6))
    ax.contour(target.T, levels=[0.5], colors=["cyan"], linewidths=1.0)
    dice = float(metrics.get("dice_equal_area", 0.0))
    cnr  = float(metrics.get("cnr", 0.0))
    ax.set_title(f"{title}\nDice={dice:.3f} CNR={cnr:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="a.u.")


# ───────────────────────────────────────────────────────────────────────
# Figure renderers
# ───────────────────────────────────────────────────────────────────────


def plot_ct_and_anatomy(result: dict) -> None:
    ct        = np.asarray(result["ct_hu"], dtype=float)
    organ     = np.asarray(result["organ_mask"], dtype=bool)
    target    = np.asarray(result["target_mask"], dtype=bool)
    body      = np.asarray(result["body_mask"], dtype=bool)
    speed     = np.asarray(result["sound_speed_m_s"], dtype=float)
    anat_recon = np.asarray(result["anatomy_reconstruction"], dtype=float)
    metrics   = result["metrics"]

    speed_vals = speed[body]
    vmin = float(np.percentile(speed_vals, 2))  if speed_vals.size else 1400.0
    vmax = float(np.percentile(speed_vals, 98)) if speed_vals.size else 1700.0
    anat_error = anat_recon - speed
    err_vmax = _robust_vmax(anat_error)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    fig.suptitle("MSD Task07 PDAC — anatomy FWI (pre-treatment baseline)")

    _show_ct(axes[0], ct, organ, target)
    axes[0].set_title("CT + labels")
    _show_speed(axes[1], speed,      "CT-derived target c",        body, vmin, vmax)
    _show_speed(axes[2], anat_recon, "FWI anatomy reconstruction", body, vmin, vmax)
    _show_field(axes[3], anat_error, "anatomy error", "m/s", vmax=max(err_vmax, 1.0))

    pearson = float(metrics.get("anatomy", {}).get("pearson", 0.0))
    nrmse   = float(metrics.get("anatomy", {}).get("nrmse",   0.0))
    axes[2].set_xlabel(f"r={pearson:.4f}  NRMSE={nrmse:.4f}", fontsize=8)

    savefig("fig01_ct_and_anatomy")
    plt.close(fig)


def plot_lesion_channels(result: dict) -> None:
    ct           = np.asarray(result["ct_hu"], dtype=float)
    target       = np.asarray(result["target_mask"], dtype=bool)
    lesion_tgt   = np.asarray(result["lesion_target"], dtype=float)
    lesion_recon = np.asarray(result["active_lesion_reconstruction"], dtype=float)
    rtm          = np.asarray(result["waveform_rtm_reconstruction"], dtype=float)
    exposure     = np.asarray(result["exposure"], dtype=float)
    history      = np.asarray(result["objective_history"], dtype=float)
    metrics      = result["metrics"]

    lesion_vmax = max(_robust_vmax(lesion_tgt, lesion_recon), 1.0)
    rtm_vmax    = max(_robust_vmax(rtm), 1.0)
    exp_vmax    = max(_robust_vmax(exposure), 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.0), constrained_layout=True)
    fig.suptitle("MSD Task07 PDAC — time-lapse lesion FWI and waveform RTM")

    _show_ct(axes[0, 0], ct, target, target)
    axes[0, 0].set_title("CT + PDAC (yellow)")

    _show_field(axes[0, 1], lesion_tgt,   "lesion Δc target", "m/s", vmax=lesion_vmax)
    _show_field(axes[0, 2], lesion_recon, "time-lapse FWI",        "m/s", vmax=lesion_vmax)
    _show_field(axes[1, 0], rtm,          "waveform RTM",          "m/s eq.", vmax=rtm_vmax)
    _show_field(axes[1, 1], exposure,     "pressure exposure",     "Pa",
                vmax=exp_vmax, cmap="magma")

    ax_obj = axes[1, 2]
    if history.size > 0:
        norm = history / max(float(history[0]), 1.0e-30)
        ax_obj.semilogy(np.arange(norm.size), norm, marker="o", ms=3)
    ax_obj.set_xlabel("PCG iteration")
    ax_obj.set_ylabel("normalized objective")
    ax_obj.set_title("solver convergence")
    ax_obj.grid(True, alpha=0.25)

    dice = float(metrics.get("active_lesion", {}).get("dice_equal_area", 0.0))
    cnr  = float(metrics.get("active_lesion", {}).get("cnr", 0.0))
    axes[0, 2].set_xlabel(f"Dice={dice:.3f}  CNR={cnr:.2f}", fontsize=8)

    savefig("fig02_lesion_channels")
    plt.close(fig)


def plot_passive_channels(result: dict) -> None:
    target        = np.asarray(result["target_mask"], dtype=bool)
    subharm       = np.asarray(result["subharmonic_reconstruction"], dtype=float)
    harmonic      = np.asarray(result["harmonic_reconstruction"], dtype=float)
    ultraharmonic = np.asarray(result["ultraharmonic_reconstruction"], dtype=float)
    fused         = np.asarray(result["fused_reconstruction"], dtype=float)
    metrics       = result["metrics"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    fig.suptitle("MSD Task07 PDAC — passive cavitation monitoring channels")

    _show_passive(axes[0], subharm,       "subharmonic",        target, metrics.get("subharmonic",    {}))
    _show_passive(axes[1], harmonic,      "2nd harmonic",       target, metrics.get("harmonic",       {}))
    _show_passive(axes[2], ultraharmonic, "ultraharmonic",      target, metrics.get("ultraharmonic",  {}))
    _show_passive(axes[3], fused,         "fused (all channels)",target, metrics.get("fusion",        {}))

    savefig("fig03_passive_channels")
    plt.close(fig)


def write_metrics(result: dict) -> None:
    metrics = result["metrics"]
    payload = {
        "chapter": "21f",
        "dataset": "MSD Task07_Pancreas, CC-BY-SA 4.0 (Antonelli 2022)",
        "ct_path": str(CT_PATH.relative_to(REPO_ROOT)),
        "segmentation_nifti_path": None,
        "grid_size": GRID_SIZE,
        "element_count": ELEMENT_COUNT,
        "iterations": ITERATIONS,
        "geometry_model": str(result.get("geometry_model", "")),
        "operator_model": str(result.get("operator_model", "")),
        "inverse_model_family": str(result.get("inverse_model_family", "")),
        "is_full_wave_inversion": bool(result.get("is_full_wave_inversion", False)),
        "uses_nonlinear_wave_propagation": bool(
            result.get("uses_nonlinear_wave_propagation", False)
        ),
        "waveform_model": str(result.get("waveform_model", "")),
        "waveform_misfit": str(result.get("waveform_misfit", "")),
        "waveform_objective": float(result.get("waveform_objective", 0.0)),
        "active_voxels": int(result.get("active_voxels", 0)),
        "measurements": int(result.get("measurements", 0)),
        "reconstruction_metrics": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in metrics.items()
        },
    }
    out_path = OUT_DIR / "scenario_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("  saved: docs/book/figures/ch21f/scenario_metrics.json")


# ───────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data I/O: download MSD CT, resample, auto-segment (no physics).
    ct_path, _label_vol = _load_ct(target_dx_m=1.2e-3)

    # All physics delegated to Rust.
    print(
        f"[ch21f] Running Rust theranostic FWI on pancreatic PDAC CT "
        f"(grid={GRID_SIZE}, elements={ELEMENT_COUNT}, iterations={ITERATIONS})"
    )
    result = kw.run_theranostic_inverse_from_ritk(
        str(ct_path),
        None,                       # no pre-existing segmentation NIfTI
        anatomy="pancreas",
        grid_size=GRID_SIZE,
        element_count=ELEMENT_COUNT,
        iterations=ITERATIONS,
    )

    plot_ct_and_anatomy(result)
    plot_lesion_channels(result)
    plot_passive_channels(result)
    write_metrics(result)
    print("[ch21f] Done.")


if __name__ == "__main__":
    main()
