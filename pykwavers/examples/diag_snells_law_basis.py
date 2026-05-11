#!/usr/bin/env python3
"""
diag_snells_law_basis.py
========================
Isolates the root cause of the tvsp_snells_law parity failure (pearson=-0.233).

Test matrix:
  Case A: pykwavers hom (c=1500, noabs)  vs  k-wave hom (c=1500, noabs)
  Case B: pykwavers het (c1/c2, noabs)   vs  k-wave het (c1/c2, noabs)

If A passes and B fails → issue is in heterogeneous medium propagation.
If A fails → issue is in source injection or basic PSTD propagation.
"""
from __future__ import annotations
import sys, os, time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Path bootstrap ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KWAVE_EXAMPLES = os.path.join(SCRIPT_DIR, "..", "..", "external", "k-wave-python", "examples")
KWAVE_ROOT     = os.path.join(SCRIPT_DIR, "..", "..", "external", "k-wave-python")
for p in (SCRIPT_DIR, KWAVE_EXAMPLES, KWAVE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import pykwavers as pkw
from kwave.kgrid import kWaveGrid
from kwave.data import Vector
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.signals import tone_burst

# ---------------------------------------------------------------------------
# Grid / source parameters (matching tvsp_snells_law exactly)
# ---------------------------------------------------------------------------
NX = NY = 128
DX = DY = 50e-3 / NX   # ~3.906e-4 m
C1 = 1500.0
C2 = 3000.0
RHO0 = 1000.0
LAYER_SPLIT_0 = NX // 2 - 1    # row 63 (0-based) is first row with c2

NUM_ELEMENTS = 61
X_OFFSET_0   = 24               # source row (0-based)
Y_OFFSET     = 20
START_IDX_0  = NY // 2 - (NUM_ELEMENTS + 1) // 2 + 1 - 1 - Y_OFFSET   # = 13
STEERING_DEG = 35.0
BASE_OFFSET  = 200
TONE_FREQ    = 1e6
TONE_CYCLES  = 8
PML_SIZE     = 20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _kwave_run_and_extract(c2d, alpha_coeff=0.0, alpha_power=1.5):
    """Run k-wave-python with given 2D sound-speed array."""
    kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    medium = kWaveMedium(sound_speed=c2d, density=RHO0,
                         alpha_coeff=alpha_coeff, alpha_power=alpha_power)
    kgrid.makeTime(c2d)
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    # source
    source = kSource()
    source.p_mask = np.zeros((NX, NY))
    source.p_mask[X_OFFSET_0, START_IDX_0:START_IDX_0 + NUM_ELEMENTS] = 1

    sampling_freq = 1.0 / dt
    elem_idx = np.arange(-(NUM_ELEMENTS-1)/2, (NUM_ELEMENTS-1)/2 + 1)
    offsets   = BASE_OFFSET + DX * elem_idx * np.sin(np.deg2rad(STEERING_DEG)) / (C1 * dt)
    source.p  = tone_burst(sampling_freq, TONE_FREQ, TONE_CYCLES,
                           signal_offset=np.round(offsets).astype(int))

    sensor = kSensor(mask=np.ones((NX, NY), dtype=bool))
    sensor.record = ["p_final"]

    result = kspaceFirstOrder(kgrid, medium, source, sensor,
                              backend="python", device="cpu", quiet=True,
                              pml_inside=True)

    # k-wave returns inner 88×88 for pml_inside; zero-pad to (NX, NY)
    pf_flat = np.asarray(result["p_final"], dtype=np.float64).ravel()
    inner = NX - 2 * PML_SIZE
    if pf_flat.size == NX * NY:
        p2d = pf_flat.reshape(NX, NY)
    elif pf_flat.size == inner * inner:
        p_inner = pf_flat.reshape(inner, inner)
        p2d = np.zeros((NX, NY), dtype=np.float64)
        p2d[PML_SIZE:NX-PML_SIZE, PML_SIZE:NY-PML_SIZE] = p_inner
    else:
        raise ValueError(f"Unexpected p_final size: {pf_flat.size}")
    return p2d, nt, dt


def _pkw_run_and_extract(c3d, nt, dt):
    """Run pykwavers PSTD with given (NX,NY,1) sound-speed array."""
    grid   = pkw.Grid(NX, NY, 1, DX, DY, DX)
    rho3d  = np.full((NX, NY, 1), RHO0, dtype=np.float64)
    medium = pkw.Medium(sound_speed=c3d, density=rho3d)

    src_mask = np.zeros((NX, NY, 1), dtype=np.float64)
    for n in range(NUM_ELEMENTS):
        src_mask[X_OFFSET_0, START_IDX_0 + n, 0] = 1.0

    dt_f  = float(dt)
    elem_idx = np.arange(-(NUM_ELEMENTS-1)/2, (NUM_ELEMENTS-1)/2 + 1)
    offsets  = BASE_OFFSET + DX * elem_idx * np.sin(np.deg2rad(STEERING_DEG)) / (C1 * dt_f)
    offsets  = np.round(offsets).astype(int)

    from kwave.utils.signals import tone_burst as tb
    raw = tb(1.0/dt_f, TONE_FREQ, TONE_CYCLES, signal_offset=offsets)
    raw_len = raw.shape[1]
    signals = np.zeros((NUM_ELEMENTS, nt), dtype=np.float64)
    signals[:, :min(raw_len, nt)] = raw[:, :min(raw_len, nt)]

    source = pkw.Source.from_mask(src_mask, signals, TONE_FREQ)
    sens_mask = np.ones((NX, NY, 1), dtype=bool)
    sensor    = pkw.Sensor.from_mask(sens_mask)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    result = sim.run(time_steps=nt, dt=dt_f)
    sd = np.asarray(result.sensor_data, dtype=np.float64)
    p_flat = sd[:, -1]
    p2d    = p_flat.reshape(NX, NY, order='F')

    # Zero PML strip
    p2d[:PML_SIZE, :]  = 0.0
    p2d[-PML_SIZE:, :] = 0.0
    p2d[:, :PML_SIZE]  = 0.0
    p2d[:, -PML_SIZE:] = 0.0
    return p2d


def metrics(ref, cand):
    a = ref.ravel();  b = cand.ravel()
    r = float(np.corrcoef(a, b)[0, 1])
    rms_ratio = float(np.sqrt(np.mean(b**2)) / (np.sqrt(np.mean(a**2)) + 1e-30))
    return r, rms_ratio


# ---------------------------------------------------------------------------
# Case A: HOMOGENEOUS c=1500
# ---------------------------------------------------------------------------
print("=" * 60)
print("Case A: homogeneous c=1500, no absorption")
print("  Running k-wave ...")
c2d_hom = np.full((NX, NY), C1, dtype=np.float64)
c3d_hom = np.full((NX, NY, 1), C1, dtype=np.float64)
t0 = time.perf_counter()
kw_hom, nt, dt = _kwave_run_and_extract(c2d_hom, alpha_coeff=0.0)
print(f"  k-wave done in {time.perf_counter()-t0:.1f}s  "
      f"nt={nt} dt={dt:.3e}  peak={float(np.abs(kw_hom).max()):.4e}")

print("  Running pykwavers ...")
t0 = time.perf_counter()
pkw_hom = _pkw_run_and_extract(c3d_hom, nt, dt)
print(f"  pykwavers done in {time.perf_counter()-t0:.1f}s  "
      f"peak={float(np.abs(pkw_hom).max()):.4e}")

r_A, rms_A = metrics(kw_hom, pkw_hom)
print(f"  pearson={r_A:.4f}  rms_ratio={rms_A:.4f}")

# ---------------------------------------------------------------------------
# Case B: HETEROGENEOUS c1/c2, no absorption
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Case B: heterogeneous c1=1500/c2=3000, no absorption")
print("  Running k-wave ...")
c2d_het = np.full((NX, NY), C1, dtype=np.float64)
c2d_het[LAYER_SPLIT_0:, :] = C2
c3d_het = np.full((NX, NY, 1), C1, dtype=np.float64)
c3d_het[LAYER_SPLIT_0:, :, :] = C2
t0 = time.perf_counter()
kw_het, nt_h, dt_h = _kwave_run_and_extract(c2d_het, alpha_coeff=0.0)
print(f"  k-wave done in {time.perf_counter()-t0:.1f}s  "
      f"nt={nt_h} dt={dt_h:.3e}  peak={float(np.abs(kw_het).max()):.4e}")

print("  Running pykwavers ...")
t0 = time.perf_counter()
pkw_het = _pkw_run_and_extract(c3d_het, nt_h, dt_h)
print(f"  pykwavers done in {time.perf_counter()-t0:.1f}s  "
      f"peak={float(np.abs(pkw_het).max()):.4e}")

r_B, rms_B = metrics(kw_het, pkw_het)
print(f"  pearson={r_B:.4f}  rms_ratio={rms_B:.4f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("SUMMARY:")
print(f"  Case A (hom):  pearson={r_A:.4f}  rms_ratio={rms_A:.4f}  "
      f"{'PASS' if r_A > 0.8 and 0.5 < rms_A < 2.0 else 'FAIL'}")
print(f"  Case B (het):  pearson={r_B:.4f}  rms_ratio={rms_B:.4f}  "
      f"{'PASS' if r_B > 0.8 and 0.5 < rms_B < 2.0 else 'FAIL'}")

# Save side-by-side comparison plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(SCRIPT_DIR, "output"), exist_ok=True)

    for name, kw, pkwf in [("hom", kw_hom, pkw_hom), ("het", kw_het, pkw_het)]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        vmax = float(np.abs(kw).max()) * 1.05 + 1e-10
        im_kw = axes[0].imshow(kw.T, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
        axes[0].set_title(f"k-wave {name}")
        im_pk = axes[1].imshow(pkwf.T, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
        axes[1].set_title(f"pykwavers {name}")
        diff = pkwf - kw
        axes[2].imshow(diff.T, aspect="equal", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, origin="upper")
        axes[2].set_title(f"diff {name}")
        plt.colorbar(im_kw, ax=axes[0])
        plt.colorbar(im_pk, ax=axes[1])
        plt.tight_layout()
        fig.savefig(os.path.join(SCRIPT_DIR, "output", f"diag_snells_basis_{name}.png"), dpi=80)
        plt.close(fig)
        print(f"  Saved output/diag_snells_basis_{name}.png")
except Exception as e:
    print(f"  [plot failed: {e}]")
