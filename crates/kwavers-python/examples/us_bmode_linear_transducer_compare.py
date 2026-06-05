"""
us_bmode_linear_transducer_compare.py
=======================================
Side-by-side comparison of k-wave-python vs pykwavers for the canonical
B-mode linear transducer ultrasound imaging example.

Physical setup (matches k-wave-python reference exactly):
  Grid:        216 x 108 x 108 active voxels, dx=dy=dz=1.852e-4 m (0.185 mm)
  PML:         [20, 10, 10] grid points (pml_inside=False)
  Medium:      heterogeneous phantom (phantom_data.mat), BonA=6, alpha=0.75 dB/MHz^1.5/cm
  Transducer:  32-element linear array, element_width=2pts, element_length=24pts, kerf=0
  Source:      velocity ux, 1.5 MHz 4-cycle tone burst, f/2 focus @ 20 mm, Hanning TX apod
  Receive:     delay-and-sum via NotATransducer.scan_line(), Rectangular RX apod
  Scan lines:  96 (full) or 16 (--quick mode, evenly spaced)

Post-processing (identical for both engines):
  TGC -> gaussian_filter(f0, 100%) + gaussian_filter(2*f0, 30%) ->
  envelope_detection -> log_compression(ratio=3)

Output figures:
  output/us_bmode_linear_transducer_compare.png  -- 2x3 comparison (300 dpi)
  output/us_bmode_linear_transducer_metrics.txt  -- Pearson r, RMS ratio, PSNR

Usage:
  python examples/us_bmode_linear_transducer_compare.py --quick   # 16 scan lines
  python examples/us_bmode_linear_transducer_compare.py --full    # 96 scan lines

References:
  Thomenius, K.E. (1996). Evolution of ultrasound beamformers.
    Proc. IEEE Ultrasonics Symp. 43(5), 820-831.
  Jensen, J.A. (1996). Field: A program for simulating ultrasound systems.
    Ultrasonics 34(2-5), 505-515.
  Treeby, B.E. & Cox, B.T. (2010). k-Wave: MATLAB toolbox for the simulation
    and reconstruction of photoacoustic wave fields.
    J. Biomed. Opt. 15(2), 021314.
"""

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.signal

# ---------------------------------------------------------------------------
# Path setup: expose k-wave-python and pykwavers
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parents[3]
_SCRIPT_DIR = Path(__file__).resolve().parent
_KWAVE_PY = _ROOT / "external" / "k-wave-python"
_PYKWAVERS = _ROOT / "crates" / "kwavers-python" / "python"

for p in [str(_SCRIPT_DIR), str(_KWAVE_PY), str(_PYKWAVERS / "python")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# k-wave-python imports (post-processing + transducer beamformer)
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.kmedium import kWaveMedium
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.reconstruction.beamform import envelope_detection
from kwave.reconstruction.tools import log_compression
from kwave.utils.conversion import db2neper
from kwave.utils.dotdictionary import dotdict
from kwave.utils.filters import gaussian_filter
from kwave.utils.signals import get_win, tone_burst

# pykwavers imports (simulation engine)
import pykwavers as pkw
from pykwavers.parity_targets import evaluate_parity
from example_parity_utils import RunningTimingStats, advance_lateral_window_inplace

# Set to False by --cpu flag in main(); True when GPU is available (default).
# Workers are 1 for GPU (persistent session) and args.workers for CPU.
_USE_GPU = True

# ---------------------------------------------------------------------------
# Phantom data download utility (reuse from k-wave-python example)
# ---------------------------------------------------------------------------
_EXAMPLE_DIR = _KWAVE_PY / "examples" / "us_bmode_linear_transducer"
_EXAMPLE_UTILS_DIR = _KWAVE_PY / "examples" / "legacy" / "us_bmode_linear_transducer"
sys.path.insert(0, str(_EXAMPLE_UTILS_DIR))
from example_utils import download_if_does_not_exist  # noqa: E402

PHANTOM_DATA_GDRIVE_ID = "1ZfSdJPe8nufZHz0U9IuwHR4chaOGAWO4"
SENSOR_DATA_GDRIVE_ID = "1lGFTifpOrzBYT4Bl_ccLu_Kx0IDxM0Lv"
PHANTOM_DATA_PATH = str(_EXAMPLE_DIR / "phantom_data.mat")
SENSOR_DATA_PATH = str(_EXAMPLE_DIR / "sensor_data.mat")

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulation constants (match k-wave-python reference EXACTLY)
# ---------------------------------------------------------------------------
PML_SIZE = Vector([20, 10, 10])           # PML grid points [x, y, z]
GRID_SIZE_PTS = Vector([256, 128, 128]) - 2 * PML_SIZE   # active domain [216, 108, 108]
GRID_SIZE_M = 40e-3                       # [m] domain size along x
DX = GRID_SIZE_M / GRID_SIZE_PTS.x       # 1.852e-4 m = 0.185 mm
DX_VEC = Vector([DX, DX, DX])            # isotropic spacing

C0 = 1540.0       # background sound speed [m/s]
RHO0 = 1000.0     # background density [kg/m^3]
SOURCE_STRENGTH = 1e6   # [Pa]
F0 = 1.5e6        # tone burst centre frequency [Hz]
N_CYCLES = 4      # tone burst cycles
ALPHA_COEFF = 0.75   # absorption [dB/MHz^y/cm]
ALPHA_POWER = 1.5    # absorption power law exponent
BON_A = 6.0          # nonlinearity parameter

N_SCAN_LINES = 96         # total scan lines
ELEM_WIDTH = 2            # element width [grid points]
ELEM_LENGTH = 24          # element length [grid points]
ELEM_SPACING = 0          # kerf width [grid points]
N_ELEMENTS = 32           # number of elements
FOCUS_DIST = 20e-3        # transmit focus distance [m]
ELEV_FOCUS_DIST = 19e-3   # elevation focus distance [m]
STEERING_ANGLE = 0.0      # degrees



# ---------------------------------------------------------------------------
# Step 1: load phantom data
# ---------------------------------------------------------------------------
def load_phantom():
    """Download (if needed) and load the heterogeneous phantom."""
    print("  Loading phantom data...")
    download_if_does_not_exist(PHANTOM_DATA_GDRIVE_ID, PHANTOM_DATA_PATH)
    ph = scipy.io.loadmat(PHANTOM_DATA_PATH)
    return ph["sound_speed_map"], ph["density_map"]


# ---------------------------------------------------------------------------
# Step 2: build kWaveGrid and NotATransducer (k-wave-python objects)
#         These are used for:
#           - dt/Nt calculation (kgrid.makeTime)
#           - transmit delay computation (not_transducer.beamforming_delays)
#           - elevation delay computation (not_transducer.elevation_beamforming_delays)
#           - receive beamforming (not_transducer.scan_line)
# ---------------------------------------------------------------------------
def build_kwave_objects(input_signal_base=None):
    """
    Build kWaveGrid and NotATransducer using exact k-wave-python parameters.

    Returns
    -------
    kgrid : kWaveGrid
    not_transducer : NotATransducer
    input_signal : ndarray, shape (Nt,) - velocity source signal [m/s]
    """
    kgrid = kWaveGrid(GRID_SIZE_PTS, DX_VEC)
    t_end = GRID_SIZE_PTS.x * DX * 2.2 / C0  # same formula as reference
    kgrid.makeTime(C0, t_end=t_end)

    # Build input velocity signal
    sig = tone_burst(1.0 / kgrid.dt, F0, N_CYCLES)
    sig = (SOURCE_STRENGTH / (C0 * RHO0)) * sig
    if input_signal_base is not None:
        sig = input_signal_base

    # Transducer geometry
    transducer_width = N_ELEMENTS * ELEM_WIDTH + (N_ELEMENTS - 1) * ELEM_SPACING
    tr = dotdict()
    tr.number_elements = N_ELEMENTS
    tr.element_width = ELEM_WIDTH
    tr.element_length = ELEM_LENGTH
    tr.element_spacing = ELEM_SPACING
    tr.radius = float("inf")
    tr.position = np.round([
        1,
        GRID_SIZE_PTS.y / 2 - transducer_width / 2,
        GRID_SIZE_PTS.z / 2 - ELEM_LENGTH / 2,
    ])
    transducer = kWaveTransducerSimple(kgrid, **tr)

    # NotATransducer (beamformer)
    nt = dotdict()
    nt.sound_speed = C0
    nt.focus_distance = FOCUS_DIST
    nt.elevation_focus_distance = ELEV_FOCUS_DIST
    nt.steering_angle = STEERING_ANGLE
    nt.transmit_apodization = "Hanning"
    nt.receive_apodization = "Rectangular"
    nt.active_elements = np.ones((N_ELEMENTS, 1))
    nt.input_signal = sig
    not_transducer = NotATransducer(transducer, kgrid, **nt)

    return kgrid, not_transducer, sig


# ---------------------------------------------------------------------------
# Step 3: build pykwavers source signals for one scan line
#         Implements Thomenius (1996) transmit delay model exactly as
#         kWaveTransducerSimple / NotATransducer does.
# ---------------------------------------------------------------------------
def build_source_mask_and_signals(kgrid, not_transducer, input_signal):
    """
    Build the 3D source mask and the per-source-point velocity signal matrix
    for use with pykwavers Source.from_velocity_mask_2d().

    Grid layout:
        Full grid:   (256, 128, 128) = GRID_SIZE_PTS + 2*PML_SIZE
        Active region: x=[PML_x:Nx-PML_x], y=[PML_y:Ny-PML_y], z=[PML_z:Nz-PML_z]
        Source mask is derived from not_transducer.active_elements_mask and
        embedded into the full grid with the PML halo.

    Transmit delays (Thomenius 1996, IEEE UFFC 43(5) 820-831):
        element_pitch = (element_width + element_spacing) * dx
        element_index = arange(-(N-1)/2, (N+1)/2)

        For focused beam:
            d_i = (f/c) * (1 - sqrt(1 + (i*p/f)^2 - 2*(i*p/f)*sin(theta)))
            delay_samples_i = round(d_i / dt)

    The source builder uses not_transducer.delay_mask() so the transmit drive
    includes the transducer's azimuth and elevation delay contract.

    Returns
    -------
    mask : ndarray, shape (256, 128, 128), float64
        Source grid mask (1.0 at transducer positions in full grid)
    ux_signals : ndarray, shape (n_source_pts, Nt)
        Per-source-point x-velocity time series [m/s], in C-order (y-slow, z-fast)
    """
    # Full grid dimensions including PML
    FNx = int(GRID_SIZE_PTS.x) + 2 * int(PML_SIZE.x)  # 216 + 40 = 256
    FNy = int(GRID_SIZE_PTS.y) + 2 * int(PML_SIZE.y)  # 108 + 20 = 128
    FNz = int(GRID_SIZE_PTS.z) + 2 * int(PML_SIZE.z)  # 108 + 20 = 128
    Nt = kgrid.Nt

    active_mask = np.asarray(not_transducer.active_elements_mask, dtype=np.float64)
    delay_mask = np.asarray(not_transducer.delay_mask(), dtype=np.int32)
    apod_mask = np.asarray(not_transducer.transmit_apodization_mask, dtype=np.float64)

    # Build 3D source mask in full-grid coordinates from the transducer SSOT.
    mask = np.zeros((FNx, FNy, FNz), dtype=np.float64)
    px, py, pz = int(PML_SIZE.x), int(PML_SIZE.y), int(PML_SIZE.z)
    mask[
        px : px + int(GRID_SIZE_PTS.x),
        py : py + int(GRID_SIZE_PTS.y),
        pz : pz + int(GRID_SIZE_PTS.z),
    ] = active_mask

    n_source_pts = int(active_mask.sum())  # = N_ELEMENTS * ELEM_WIDTH * ELEM_LENGTH = 1536

    # kwavers applies 2*c0*dt/dx internally for additive velocity sources
    # (commit caabc640). Do NOT apply the factor here.

    delay_full = np.zeros((FNx, FNy, FNz), dtype=np.int32)
    apod_full = np.zeros((FNx, FNy, FNz), dtype=np.float64)
    delay_full[
        px : px + int(GRID_SIZE_PTS.x),
        py : py + int(GRID_SIZE_PTS.y),
        pz : pz + int(GRID_SIZE_PTS.z),
    ] = delay_mask
    apod_full[
        px : px + int(GRID_SIZE_PTS.x),
        py : py + int(GRID_SIZE_PTS.y),
        pz : pz + int(GRID_SIZE_PTS.z),
    ] = apod_mask

    # Source points follow the C-order enumeration of the full-grid mask.
    # NotATransducer.delay_mask() convention: delay_mask=d means inject
    # padded_signal[d:d+Nt], where padded_signal = [zeros(max_delay), input_signal].
    # Element with delay=max_delay fires first (signal[0] at t=0);
    # element with delay=0 fires last (signal[0] at t=max_delay).
    # This matches the k-wave C++ binary's padded-input-signal injection model.
    input_1d = np.asarray(input_signal).ravel()
    max_delay = int(delay_full.max())
    padded_signal = np.concatenate([np.zeros(max_delay), input_1d])
    ux_signals = np.zeros((n_source_pts, Nt), dtype=np.float64)

    for p, (x, y, z) in enumerate(np.argwhere(mask != 0)):
        delay = int(delay_full[x, y, z])
        weight = float(apod_full[x, y, z])
        n_inj = min(padded_signal.size - delay, Nt)
        if n_inj > 0:
            ux_signals[p, :n_inj] = padded_signal[delay : delay + n_inj] * weight

    return mask, ux_signals


# ---------------------------------------------------------------------------
# Step 4: run one pykwavers scan-line simulation
# ---------------------------------------------------------------------------

def _pool_initializer(rayon_threads):
    """Pool initializer: cap Rayon threads before pykwavers is imported."""
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)
    # Re-bootstrap sys.path in spawned process (Windows spawn loses parent path)
    _ROOT = Path(__file__).parents[3]
    for p in [str(_ROOT / "external" / "k-wave-python"), str(_ROOT / "crates" / "kwavers-python" / "python")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _scanline_worker(args):
    """Top-level worker: runs one scan-line and returns (out_idx, sensor_data_raw)."""
    out_idx, ss_slice, rho_slice, mask, ux_signals, dt, Nt = args

    class _FakeKgrid:
        pass

    kg = _FakeKgrid()
    kg.dt = dt
    kg.Nt = Nt
    sensor_data_raw = run_pykwavers_scanline(ss_slice, rho_slice, mask, ux_signals, kg)
    return out_idx, sensor_data_raw


def run_pykwavers_scanline(sound_speed_slice, density_slice, mask, ux_signals, kgrid):
    """
    Run a single scan-line simulation using pykwavers.

    Grid layout: full 256x128x128 grid with PML [20,10,10] embedded inside.
    Active region: x=[20:236], y=[10:118], z=[10:118] (matches k-wave pml_inside=False).
    The phantom (216x108x108) fills the active region; PML cells get background values.

    Parameters
    ----------
    sound_speed_slice : ndarray (216, 108, 108)  active-domain phantom slice
    density_slice     : ndarray (216, 108, 108)  active-domain density slice
    mask              : ndarray (256, 128, 128)  source+sensor mask in full grid
    ux_signals        : ndarray (n_src, Nt)      per-source-point x-velocity [m/s]
    kgrid             : kWaveGrid                for dt, Nt

    Returns
    -------
    sensor_data_raw : ndarray (n_src, Nt)  pressure at source/sensor positions
    """
    FNx = int(GRID_SIZE_PTS.x) + 2 * int(PML_SIZE.x)  # 256
    FNy = int(GRID_SIZE_PTS.y) + 2 * int(PML_SIZE.y)  # 128
    FNz = int(GRID_SIZE_PTS.z) + 2 * int(PML_SIZE.z)  # 128
    px, py, pz = int(PML_SIZE.x), int(PML_SIZE.y), int(PML_SIZE.z)

    grid = pkw.Grid(FNx, FNy, FNz, DX, DX, DX)

    # Build full 256x128x128 medium arrays:
    # - Active region [px:px+ANx, py:py+ANy, pz:pz+ANz] = phantom slice
    # - PML region = background (C0, RHO0)
    ss_full = np.full((FNx, FNy, FNz), C0, dtype=np.float64)
    rho_full = np.full((FNx, FNy, FNz), RHO0, dtype=np.float64)
    abs_full = np.full((FNx, FNy, FNz), ALPHA_COEFF, dtype=np.float64)
    nl_full = np.full((FNx, FNy, FNz), BON_A, dtype=np.float64)
    ANx = int(GRID_SIZE_PTS.x)
    ANy = int(GRID_SIZE_PTS.y)
    ANz = int(GRID_SIZE_PTS.z)
    ss_full[px:px+ANx, py:py+ANy, pz:pz+ANz] = sound_speed_slice.astype(np.float64)
    rho_full[px:px+ANx, py:py+ANy, pz:pz+ANz] = density_slice.astype(np.float64)

    medium = pkw.Medium(
        sound_speed=ss_full,
        density=rho_full,
        absorption=abs_full,
        nonlinearity=nl_full,
    )

    # Source: per-element velocity (ux_signals already incorporates TX delays)
    source = pkw.Source.from_velocity_mask_2d(
        mask,
        ux=ux_signals,
        mode="additive",
    )

    # Sensor: same mask (record pressure at all element positions)
    sensor = pkw.Sensor.from_mask(mask.astype(bool))

    # Solver selection: GPU PSTD if requested, else CPU PSTD
    solver_type = pkw.SolverType.PstdGpu if _USE_GPU else pkw.SolverType.PSTD

    # Simulation (PML embedded inside full 256×128×128 grid — pml_inside=True default)
    sim = pkw.Simulation(
        grid, medium, source, sensor,
        solver=solver_type,
    )
    sim.set_pml_size_xyz(int(PML_SIZE.x), int(PML_SIZE.y), int(PML_SIZE.z))
    # Enable nonlinear + absorption for CPU path only; GPU path uses GpuPstdSession
    if not _USE_GPU:
        sim.set_nonlinear(True)
        sim.set_alpha_coeff(ALPHA_COEFF)
        sim.set_alpha_power(ALPHA_POWER)

    result = sim.run(kgrid.Nt, dt=kgrid.dt)
    sd = result.sensor_data
    return sd[:, :kgrid.Nt]


# ---------------------------------------------------------------------------
# Step 5: run full B-mode loop over all scan lines
# ---------------------------------------------------------------------------
def _build_full_medium_arrays(ss_slice, rho_slice, px, py, pz, ANx, ANy, ANz, FNx, FNy, FNz):
    """Build 256x128x128 medium arrays with PML padding from an active-domain slice."""
    ss_full = np.full((FNx, FNy, FNz), C0, dtype=np.float64)
    rho_full = np.full((FNx, FNy, FNz), RHO0, dtype=np.float64)
    ss_full[px:px+ANx, py:py+ANy, pz:pz+ANz]  = ss_slice.astype(np.float64)
    rho_full[px:px+ANx, py:py+ANy, pz:pz+ANz] = rho_slice.astype(np.float64)
    return ss_full, rho_full


def run_pykwavers_bmode(sound_speed_map, density_map, kgrid, not_transducer,
                        input_signal, scan_line_indices, workers=1):
    """
    Loop over scan lines, running pykwavers simulation for each.

    The phantom is sliced laterally as in the k-wave-python reference:
        medium_position advances by element_width per scan line.

    When --gpu is active, a single GpuPstdSession is created before the loop
    and reused, eliminating ~500 ms wgpu pipeline-compilation overhead per scan
    line.  The CPU path is unchanged (multiprocess workers).

    Parameters
    ----------
    scan_line_indices : list[int]  indices of scan lines to compute
    workers : int  number of parallel worker processes (default 1 = sequential)

    Returns
    -------
    scan_lines : ndarray (len(scan_line_indices), Nt)  beamformed scan lines
    gpu_profiles : RunningTimingStats | None  per-line timing summary when GPU is used
    """
    # Build source mask and signals (same for all scan lines, steering_angle=0)
    print("  Building transmit source signals...")
    mask, ux_signals = build_source_mask_and_signals(kgrid, not_transducer, input_signal)

    n_lines = len(scan_line_indices)
    # Precompute lateral phantom offsets once; the loop only selects the slice.
    medium_positions = [sl_idx * ELEM_WIDTH for sl_idx in scan_line_indices]
    scan_lines = np.zeros((n_lines, kgrid.Nt))

    FNx = int(GRID_SIZE_PTS.x) + 2 * int(PML_SIZE.x)   # 256
    FNy = int(GRID_SIZE_PTS.y) + 2 * int(PML_SIZE.y)   # 128
    FNz = int(GRID_SIZE_PTS.z) + 2 * int(PML_SIZE.z)   # 128
    px, py, pz = int(PML_SIZE.x), int(PML_SIZE.y), int(PML_SIZE.z)
    ANx = int(GRID_SIZE_PTS.x)
    ANy = int(GRID_SIZE_PTS.y)
    ANz = int(GRID_SIZE_PTS.z)

    # ── GPU path: persistent GpuPstdSession (compiles pipelines once) ────────
    if _USE_GPU:
        print("  [GPU] Initialising GpuPstdSession (compiles pipelines once)...")
        t_init = time.perf_counter()
        gpu_profiles = RunningTimingStats(
            (
                "medium_upload_ns",
                "medium_variable_upload_ns",
                "medium_static_upload_ns",
                "solver_run_ns",
                "materialize_ns",
                "total_ns",
            )
        )

        grid = pkw.Grid(FNx, FNy, FNz, DX, DX, DX)
        # Reuse the full-grid medium buffers across scan lines and overwrite only
        # the active interior slice to avoid repeated full-volume allocations.
        active_region = (
            slice(px, px + ANx),
            slice(py, py + ANy),
            slice(pz, pz + ANz),
        )
        ss_full = np.full((FNx, FNy, FNz), C0, dtype=np.float64)
        rho_full = np.full((FNx, FNy, FNz), RHO0, dtype=np.float64)
        ss_active = ss_full[active_region]
        rho_active = rho_full[active_region]

        # Initial medium: use first scan-line slice
        mp0 = medium_positions[0]
        np.copyto(ss_active, sound_speed_map[:, mp0:mp0 + ANy, :])
        np.copyto(rho_active, density_map[:, mp0:mp0 + ANy, :])

        # Absorption and nonlinearity remain fixed across the scan-line sweep.
        abs_full = np.full((FNx, FNy, FNz), ALPHA_COEFF, dtype=np.float64)
        nl_full  = np.full((FNx, FNy, FNz), BON_A,       dtype=np.float64)

        session = pkw.GpuPstdSession(
            grid,
            ss_full, rho_full,
            dt=kgrid.dt, time_steps=kgrid.Nt,
            absorption=abs_full,
            nonlinearity=nl_full,
            pml_size_xyz=(int(PML_SIZE.x), int(PML_SIZE.y), int(PML_SIZE.z)),
            alpha_power=ALPHA_POWER,
        )
        # mask is float (nonzero = source/sensor point)
        session.set_source_sensor(mask.astype(np.float64), ux_signals)
        # k-wave-python NotATransducer defaults u_mode to "additive-no-correction".
        # Disable the GPU source-kappa correction so the transducer contract
        # matches the reference path before scan-line recomposition.
        session.disable_source_correction()
        print(f"  [GPU] Session ready in {time.perf_counter()-t_init:.1f}s")

        try:
            from tqdm import tqdm
            _iter = tqdm(enumerate(scan_line_indices), total=n_lines,
                         desc="pykwavers GPU scan lines")
        except ImportError:
            _iter = enumerate(scan_line_indices)

        prev_medium_position = mp0
        for out_idx, _sl_idx in _iter:
            medium_position = medium_positions[out_idx]
            if out_idx == 0:
                np.copyto(ss_active, sound_speed_map[:, medium_position:medium_position + ANy, :])
                np.copyto(rho_active, density_map[:, medium_position:medium_position + ANy, :])
            else:
                advance_lateral_window_inplace(
                    ss_active,
                    sound_speed_map,
                    prev_medium_position,
                    medium_position,
                )
                advance_lateral_window_inplace(
                    rho_active,
                    density_map,
                    prev_medium_position,
                    medium_position,
                )
            prev_medium_position = medium_position

            sensor_data_raw = session.run_scan_line(ss_full, rho_full)
            scan_lines[out_idx, :] = not_transducer.scan_line(
                not_transducer.combine_sensor_data(sensor_data_raw)
            )
            gpu_profiles.update(session.last_run_profile_ns)

        return scan_lines, gpu_profiles

    # ── CPU path (unchanged): multiprocess workers ────────────────────────────
    # Build per-scan-line phantom slices
    slices = []
    for medium_position in medium_positions:
        ss_slice = sound_speed_map[
            :, medium_position : medium_position + int(GRID_SIZE_PTS.y), :
        ].copy()
        rho_slice = density_map[
            :, medium_position : medium_position + int(GRID_SIZE_PTS.y), :
        ].copy()
        slices.append((ss_slice, rho_slice))

    # Clamp workers: each PSTD sim needs ~800 MB; cap to avoid OOM
    cpu_count = os.cpu_count() or 1
    workers = max(1, min(workers, n_lines))
    # Rayon threads per worker: share cores evenly, minimum 1
    rayon_threads = max(1, cpu_count // workers)
    print(f"  Workers={workers}, RAYON_NUM_THREADS={rayon_threads} per worker "
          f"(total threads <= {workers * rayon_threads})")

    worker_args = [
        (out_idx, slices[out_idx][0], slices[out_idx][1],
         mask, ux_signals, kgrid.dt, kgrid.Nt)
        for out_idx in range(n_lines)
    ]

    if workers > 1:
        print(f"  Dispatching {n_lines} scan lines across {workers} worker processes...")
        with multiprocessing.Pool(
            processes=workers,
            initializer=_pool_initializer,
            initargs=(rayon_threads,),
        ) as pool:
            try:
                from tqdm import tqdm
                results = list(tqdm(
                    pool.imap_unordered(_scanline_worker, worker_args),
                    total=n_lines, desc="pykwavers scan lines",
                ))
            except ImportError:
                results = pool.map(_scanline_worker, worker_args)
    else:
        # Single-process path: set Rayon threads before first pykwavers call
        os.environ.setdefault("RAYON_NUM_THREADS", str(cpu_count))
        try:
            from tqdm import tqdm
            iterator = tqdm(worker_args, desc="pykwavers scan lines")
        except ImportError:
            iterator = worker_args
        results = [_scanline_worker(a) for a in iterator]

    # Collect results (may arrive out of order with imap_unordered)
    for out_idx, sensor_data_raw in results:
        scan_lines[out_idx, :] = not_transducer.scan_line(
            not_transducer.combine_sensor_data(sensor_data_raw)
        )

    return scan_lines, None


# ---------------------------------------------------------------------------
# Step 6: load k-wave-python reference scan lines
# ---------------------------------------------------------------------------
def load_kwave_reference(scan_line_indices):
    """
    Load pre-computed k-wave-python scan lines from sensor_data.mat.
    Downloads from Google Drive if not present.

    Returns
    -------
    scan_lines : ndarray (len(scan_line_indices), Nt)
    """
    print("  Loading k-wave-python reference data...")
    download_if_does_not_exist(SENSOR_DATA_GDRIVE_ID, SENSOR_DATA_PATH)
    data = scipy.io.loadmat(SENSOR_DATA_PATH)["sensor_data_all_lines"]
    # data shape: (96, Nt)
    return data[scan_line_indices, :]


# ---------------------------------------------------------------------------
# Step 6b: run k-wave-python OMP (CPU) for N_PROBE scan lines and time it
# ---------------------------------------------------------------------------
def run_kwave_omp_timed(sound_speed_map, density_map, kgrid, not_transducer,
                        scan_line_indices):
    """
    Run k-Wave-Python OMP binary for each requested scan line, measuring wall time.

    Uses the kspaceFirstOrder-omp.exe C++ binary (multi-threaded CPU, no CUDA needed).
    Writes HDF5 input/output to a temp directory so as not to litter the working dir.

    Returns
    -------
    scan_lines : ndarray (n_lines, Nt)
    elapsed_s  : float  total wall time [seconds]
    """
    import tempfile

    n_lines = len(scan_line_indices)
    scan_lines = np.zeros((n_lines, kgrid.Nt))

    # Build a kWaveMedium (background; will be overridden per scan line)
    medium = kWaveMedium(
        sound_speed=np.full(tuple(GRID_SIZE_PTS), C0, dtype=np.float32),
        density=np.full(tuple(GRID_SIZE_PTS), RHO0, dtype=np.float32),
        alpha_coeff=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
        BonA=BON_A,
    )

    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        for out_idx, sl_idx in enumerate(scan_line_indices):
            medium_position = sl_idx * ELEM_WIDTH
            medium.sound_speed = sound_speed_map[
                :, medium_position : medium_position + int(GRID_SIZE_PTS.y), :
            ].astype(np.float32)
            medium.density = density_map[
                :, medium_position : medium_position + int(GRID_SIZE_PTS.y), :
            ].astype(np.float32)

            sim_opts = SimulationOptions(
                pml_inside=False,
                pml_size=PML_SIZE,
                data_cast="single",
                data_recast=True,
                save_to_disk=True,
                input_filename=f"kwave_input_{out_idx}.h5",
                save_to_disk_exit=False,
            )
            exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

            sensor_data = kspaceFirstOrder3D(
                medium=medium,
                kgrid=kgrid,
                source=not_transducer,
                sensor=not_transducer,
                simulation_options=sim_opts,
                execution_options=exec_opts,
            )
            scan_lines[out_idx, :] = not_transducer.scan_line(
                not_transducer.combine_sensor_data(sensor_data["p"].T)
            )
            elapsed_so_far = time.perf_counter() - t_start
            print(f"    scan line {out_idx+1}/{n_lines} done  "
                  f"({elapsed_so_far:.1f}s elapsed, "
                  f"{elapsed_so_far/(out_idx+1):.1f}s/line)")

    elapsed_s = time.perf_counter() - t_start
    return scan_lines, elapsed_s


# ---------------------------------------------------------------------------
# Step 7: post-processing (identical for both engines)
#         TGC -> Gaussian filter (fundamental + harmonic) ->
#         envelope detection -> log compression
# ---------------------------------------------------------------------------
def post_process(scan_lines, kgrid):
    """
    Apply the canonical post-processing chain from the k-wave-python example.

    Parameters
    ----------
    scan_lines : ndarray (n_lines, Nt)

    Returns
    -------
    fund : ndarray (n_lines, Nt)  fundamental image
    harm : ndarray (n_lines, Nt)  harmonic image
    """
    Nt = kgrid.Nt

    # --- Remove input signal artefact (Tukey window) ---
    tukey_win, _ = get_win(Nt * 2, "Tukey", False, 0.05)
    sig_tmp = tone_burst(1.0 / kgrid.dt, F0, N_CYCLES)
    sig_tmp = (SOURCE_STRENGTH / (C0 * RHO0)) * sig_tmp
    transmit_len = len(sig_tmp.squeeze())
    scan_line_win = np.concatenate(
        (np.zeros([1, transmit_len * 2]),
         tukey_win.T[:, : Nt - transmit_len * 2]),
        axis=1,
    )
    sl = scan_lines * scan_line_win

    # --- Time Gain Compensation ---
    # r = c0 * (t - t0) / 2  [one-way range]
    # TGC = exp(alpha_Np/m * f0^y * 2 * r)
    t0 = transmit_len * kgrid.dt / 2
    t_arr = np.arange(1, Nt + 1) * kgrid.dt
    r = C0 * (t_arr - t0) / 2
    tgc_alpha_db_cm = ALPHA_COEFF * (F0 * 1e-6) ** ALPHA_POWER
    tgc_alpha_np_m = db2neper(tgc_alpha_db_cm) * 100.0
    tgc = np.exp(tgc_alpha_np_m * 2.0 * r)
    sl = sl * tgc[np.newaxis, :]

    # --- Gaussian frequency filtering ---
    fund = gaussian_filter(sl, 1.0 / kgrid.dt, F0, 100)
    harm = gaussian_filter(sl, 1.0 / kgrid.dt, 2.0 * F0, 30)

    # --- Envelope detection ---
    fund = envelope_detection(fund)
    harm = envelope_detection(harm)

    # --- Log compression ---
    fund = log_compression(fund, 3, True)
    harm = log_compression(harm, 3, True)

    return fund, harm


# ---------------------------------------------------------------------------
# Step 8: quantitative comparison metrics
# ---------------------------------------------------------------------------
def compute_metrics(kwave_img, pkw_img, label):
    """
    Compute Pearson r, RMS ratio, and PSNR between two images.

    Parameters
    ----------
    kwave_img, pkw_img : ndarray (n_lines, Nt)

    Returns
    -------
    dict with keys: pearson_r, rms_ratio, psnr_db
    """
    a = kwave_img.ravel()
    b = pkw_img.ravel()

    # Pearson correlation
    r = float(np.corrcoef(a, b)[0, 1])

    # RMS ratio
    rms_kw = float(np.sqrt(np.mean(a ** 2)))
    rms_pk = float(np.sqrt(np.mean(b ** 2)))
    rms_ratio = rms_pk / (rms_kw + 1e-30)

    # PSNR
    diff = a - b
    max_val = max(a.max(), b.max())
    mse = float(np.mean(diff ** 2))
    psnr = 20.0 * np.log10(max_val / (np.sqrt(mse) + 1e-30))

    print(f"  [{label}] Pearson r={r:.4f}  RMS ratio={rms_ratio:.4f}  PSNR={psnr:.1f} dB")
    return {"pearson_r": r, "rms_ratio": rms_ratio, "psnr_db": psnr}

# ---------------------------------------------------------------------------
# Step 9: plotting
# ---------------------------------------------------------------------------
def plot_comparison(kwave_fund, kwave_harm, pkw_fund, pkw_harm,
                    sound_speed_map, kgrid, scan_line_indices):
    """
    2x3 comparison figure:
      Row 1: k-wave-python  | Row 2: pykwavers
      Col 1: Sound Speed    | Col 2: Fundamental | Col 3: Harmonic

    Axes in mm, 300 dpi.
    """
    image_size = kgrid.size   # metres
    x_axis_mm = [0.0, image_size[0] * 1e3 * 1.1]
    y_axis_mm = [-0.5 * image_size[1] * 1e3, 0.5 * image_size[1] * 1e3]
    # Sound speed: central z-slice of the first scan line's slice
    ss_centre_z = sound_speed_map[:, 64:-64, int(GRID_SIZE_PTS.z / 2)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row_idx, (fund, harm, row_label) in enumerate([
        (kwave_fund, kwave_harm, "k-wave-python (reference)"),
        (pkw_fund, pkw_harm, "pykwavers"),
    ]):
        # Column 0: Sound Speed (same for both rows)
        ax = axes[row_idx, 0]
        im = ax.imshow(
            ss_centre_z, aspect="auto",
            extent=[y_axis_mm[0], y_axis_mm[1], x_axis_mm[1], x_axis_mm[0]],
        )
        ax.set_ylim(40, 5)
        ax.set_xlabel("Width [mm]", fontsize=10)
        ax.set_ylabel("Depth [mm]", fontsize=10)
        ax.set_title(f"Sound Speed  ({row_label})", fontsize=9)
        plt.colorbar(im, ax=ax, label="m/s", fraction=0.046, pad=0.04)

        # Column 1: Fundamental
        ax = axes[row_idx, 1]
        # Compute y extent from scan line lateral positions
        y_start_m = (scan_line_indices[0] * ELEM_WIDTH) * DX - image_size[1] / 2
        y_end_m = (scan_line_indices[-1] * ELEM_WIDTH + 1) * DX - image_size[1] / 2
        bmode_extent = [y_start_m * 1e3, y_end_m * 1e3, x_axis_mm[1], x_axis_mm[0]]
        ax.imshow(fund.T, cmap="grey", aspect="auto", extent=bmode_extent)
        ax.set_ylim(40, 5)
        ax.set_xlabel("Width [mm]", fontsize=10)
        ax.set_ylabel("Depth [mm]", fontsize=10) if row_idx == 0 else ax.set_yticks([])
        ax.set_title(f"Fundamental ({row_label})", fontsize=9)

        # Column 2: Harmonic
        ax = axes[row_idx, 2]
        ax.imshow(harm.T, cmap="grey", aspect="auto", extent=bmode_extent)
        ax.set_ylim(40, 5)
        ax.set_xlabel("Width [mm]", fontsize=10)
        ax.set_yticks([])
        ax.set_title(f"Harmonic ({row_label})", fontsize=9)

    fig.suptitle(
        "us_bmode_linear_transducer: k-wave-python vs pykwavers\n"
        f"({len(scan_line_indices)}/{N_SCAN_LINES} scan lines, "
        f"f0={F0*1e-6:.1f} MHz, focus={FOCUS_DIST*1e3:.0f} mm)",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "us_bmode_linear_transducer_compare.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also save individual reference figure (k-wave-python only, 1x3)
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    for col_idx, (img, title, cmap) in enumerate([
        (ss_centre_z, "Sound Speed", "viridis"),
        (kwave_fund.T, "Fundamental", "grey"),
        (kwave_harm.T, "Harmonic", "grey"),
    ]):
        ax = axes2[col_idx]
        if col_idx == 0:
            ax.imshow(img, aspect="auto",
                      extent=[y_axis_mm[0], y_axis_mm[1], x_axis_mm[1], x_axis_mm[0]])
        else:
            ax.imshow(img, cmap=cmap, aspect="auto", extent=bmode_extent)
        ax.set_ylim(40, 5)
        ax.set_xlabel("Width [mm]", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Depth [mm]", fontsize=10)
        else:
            ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    fig2.tight_layout()
    ref_path = OUTPUT_DIR / "us_bmode_linear_transducer_kwave.png"
    fig2.savefig(str(ref_path), dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {ref_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _USE_GPU

    parser = argparse.ArgumentParser(description="us_bmode_linear_transducer comparison")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true",
                       help="Use 16 evenly-spaced scan lines (fast)")
    group.add_argument("--full", action="store_true",
                       help="Use all 96 scan lines (accurate)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (default 4; each sim ~800 MB RAM)")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Use GPU PSTD solver (default; requires 'gpu' feature)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU PSTD solver (multi-worker, nonlinear+absorption enabled)")
    parser.add_argument("--run-kwave", action="store_true",
                        help="Actually run k-Wave-Python OMP (CPU) and time it "
                             "(requires kspaceFirstOrder-omp.exe)")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Return success even when scientific parity targets fail.")
    args = parser.parse_args()

    _USE_GPU = not args.cpu  # GPU is default; --cpu forces CPU path
    workers = 1 if _USE_GPU else args.workers

    n_lines = 16 if args.quick else N_SCAN_LINES
    scan_line_indices = np.linspace(0, N_SCAN_LINES - 1, n_lines, dtype=int).tolist()

    print("=" * 60)
    print("us_bmode_linear_transducer: k-wave-python vs pykwavers")
    print(f"  Grid   : {int(GRID_SIZE_PTS.x)}x{int(GRID_SIZE_PTS.y)}x{int(GRID_SIZE_PTS.z)}"
          f"  dx={DX*1e3:.3f} mm")
    print(f"  PML    : [{int(PML_SIZE.x)}, {int(PML_SIZE.y)}, {int(PML_SIZE.z)}] (pml_inside=False)")
    print(f"  Mode   : {'quick (16 lines)' if args.quick else 'full (96 lines)'}")
    print(f"  Solver : {'GPU PSTD (GpuPstdSession)' if _USE_GPU else 'CPU PSTD'}")
    print(f"  Workers: {workers}")
    print(f"  k-Wave : {'OMP run + timed' if args.run_kwave else 'load pre-computed reference'}")
    print("=" * 60)

    # --- Load phantom ---
    sound_speed_map, density_map = load_phantom()
    print(f"  Phantom shape: {sound_speed_map.shape}")

    # --- Build k-Wave objects (timing + beamformer) ---
    print("  Building kWaveGrid and NotATransducer...")
    kgrid, not_transducer, input_signal = build_kwave_objects()
    print(f"  dt={kgrid.dt:.3e} s  Nt={kgrid.Nt}  t_end={kgrid.Nt * kgrid.dt:.3e} s")

    # --- k-wave-python reference ---
    kwave_elapsed_s = None
    if args.run_kwave:
        print(f"\n[1/3] Running k-Wave-Python OMP ({n_lines} scan lines, timed)...")
        kwave_sl, kwave_elapsed_s = run_kwave_omp_timed(
            sound_speed_map, density_map, kgrid, not_transducer, scan_line_indices,
        )
        print(f"  k-Wave OMP: {kwave_elapsed_s:.1f}s total  "
              f"({kwave_elapsed_s/n_lines:.1f}s/line, "
              f"extrapolated 96 lines = {kwave_elapsed_s/n_lines*96/60:.1f} min)")
    else:
        print("\n[1/3] Loading k-wave-python reference scan lines...")
        kwave_sl = load_kwave_reference(scan_line_indices)
    print(f"  Reference scan lines shape: {kwave_sl.shape}")

    # --- Run pykwavers B-mode (with NPZ caching for sweep re-use) ---
    mode_tag = "quick" if args.quick else "full"
    solver_tag = "gpu" if _USE_GPU else "cpu"
    pkw_cache_path = OUTPUT_DIR / f"us_bmode_linear_transducer_pykwavers_cache_{mode_tag}_{solver_tag}.npz"
    if pkw_cache_path.exists():
        print(f"\n[2/3] Loading cached pykwavers scan lines ({pkw_cache_path.name})...")
        cached = np.load(pkw_cache_path)
        pkw_sl = cached["scan_lines"]
        pkw_elapsed_s = float(cached["runtime_s"])
        pkw_profiles = None
        print(f"  Loaded {pkw_sl.shape[0]} cached scan lines in {pkw_elapsed_s:.1f}s (original)")
    else:
        print(f"\n[2/3] Running pykwavers B-mode ({n_lines} scan lines)...")
        t0_pkw = time.perf_counter()
        pkw_sl, pkw_profiles = run_pykwavers_bmode(
            sound_speed_map, density_map, kgrid, not_transducer,
            input_signal, scan_line_indices, workers=workers,
        )
        pkw_elapsed_s = time.perf_counter() - t0_pkw
        np.savez(pkw_cache_path, scan_lines=pkw_sl, runtime_s=pkw_elapsed_s)
        print(f"  pykwavers: {pkw_elapsed_s:.1f}s total  "
              f"({pkw_elapsed_s/n_lines:.1f}s/line, "
              f"extrapolated 96 lines = {pkw_elapsed_s/n_lines*96/60:.1f} min)")
    print(f"  pykwavers scan lines shape: {pkw_sl.shape}")

    # --- Raw scan-line metrics (physics parity, pre-post-processing) ---
    # log_compression(normalize=True) in post_process() renormalises per-image
    # by its own max, which amplifies small peak differences into large
    # image-RMS differences even when raw scan lines agree within a few %.
    # Use raw pressure traces for the PASS/FAIL decision; the log-compressed
    # fund/harm metrics are kept for visualization / human-eye comparison.
    metrics_raw = compute_metrics(kwave_sl, pkw_sl, "Raw scan_lines")

    # --- Post-processing ---
    print("\n[3/3] Post-processing and comparison...")
    kwave_fund, kwave_harm = post_process(kwave_sl, kgrid)
    pkw_fund, pkw_harm = post_process(pkw_sl, kgrid)

    # --- Metrics ---
    metrics_fund = compute_metrics(kwave_fund, pkw_fund, "Fundamental")
    metrics_harm = compute_metrics(kwave_harm, pkw_harm, "Harmonic")
    eval_raw = evaluate_parity(metrics_raw, "fundamental", n_lines, N_SCAN_LINES)
    eval_fund = evaluate_parity(metrics_fund, "fundamental", n_lines, N_SCAN_LINES)
    eval_harm = evaluate_parity(metrics_harm, "harmonic", n_lines, N_SCAN_LINES)
    # Parity decision uses raw scan-line metrics (physics-level comparison).
    overall_status = "PASS" if eval_raw["status"] == "PASS" else "FAIL"

    # --- Figures ---
    plot_comparison(kwave_fund, kwave_harm, pkw_fund, pkw_harm,
                    sound_speed_map, kgrid, scan_line_indices)

    # --- Save metrics text ---
    solver_label = "GPU PSTD" if args.gpu else "CPU PSTD"
    metrics_path = OUTPUT_DIR / "us_bmode_linear_transducer_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"us_bmode_linear_transducer parity metrics\n")
        f.write(f"scan_lines: {n_lines}/{N_SCAN_LINES}\n")
        f.write(f"solver: pykwavers {solver_label}\n\n")
        f.write(f"parity_status: {overall_status}\n")
        f.write(f"validation_tier: {'Tier 2 heterogeneous/transducer parity'}\n\n")
        f.write(f"Timing ({n_lines} scan lines):\n")
        f.write(f"  pykwavers {solver_label}: {pkw_elapsed_s:.1f}s total  "
                f"({pkw_elapsed_s/n_lines:.1f}s/line)\n")
        if kwave_elapsed_s is not None:
            f.write(f"  k-Wave OMP (CPU):        {kwave_elapsed_s:.1f}s total  "
                    f"({kwave_elapsed_s/n_lines:.1f}s/line)\n")
            speedup = kwave_elapsed_s / pkw_elapsed_s
            f.write(f"  Speedup vs k-Wave OMP:   {speedup:.1f}x\n")
        else:
            f.write(f"  k-Wave OMP (CPU):        (pre-computed reference, not timed)\n")
            f.write(f"  Run with --run-kwave to measure k-Wave OMP timing.\n")
        f.write(f"\nRaw scan_lines (physics parity — drives PASS/FAIL):\n")
        f.write(f"  Pearson r  = {metrics_raw['pearson_r']:.6f}\n")
        f.write(f"  RMS ratio  = {metrics_raw['rms_ratio']:.6f}\n")
        f.write(f"  PSNR [dB]  = {metrics_raw['psnr_db']:.2f}\n")
        f.write(f"  Target tier = {eval_raw['tier']}\n")
        f.write(f"  Status      = {eval_raw['status']}\n")
        f.write(f"  Targets     = r>={eval_raw['target']['pearson_r']:.3f}, "
                f"{eval_raw['target']['rms_ratio_min']:.2f}<=RMS<={eval_raw['target']['rms_ratio_max']:.2f}, "
                f"PSNR>={eval_raw['target']['psnr_db']:.1f} dB\n")
        if pkw_profiles is not None:
            f.write(f"\nGPU scan-line profile (per line, ms):\n")
            for line in pkw_profiles.format_lines():
                f.write(line + "\n")
        f.write(f"\nFundamental (log-compressed, normalize=True — for visualization only):\n")
        f.write(f"  Pearson r  = {metrics_fund['pearson_r']:.6f}\n")
        f.write(f"  RMS ratio  = {metrics_fund['rms_ratio']:.6f}\n")
        f.write(f"  PSNR [dB]  = {metrics_fund['psnr_db']:.2f}\n\n")
        f.write(f"  Target tier = {eval_fund['tier']}\n")
        f.write(f"  Status      = {eval_fund['status']}\n")
        f.write(f"  Targets     = r>={eval_fund['target']['pearson_r']:.3f}, "
                f"{eval_fund['target']['rms_ratio_min']:.2f}<=RMS<={eval_fund['target']['rms_ratio_max']:.2f}, "
                f"PSNR>={eval_fund['target']['psnr_db']:.1f} dB\n\n")
        f.write(f"Harmonic (log-compressed, normalize=True — for visualization only):\n")
        f.write(f"  Pearson r  = {metrics_harm['pearson_r']:.6f}\n")
        f.write(f"  RMS ratio  = {metrics_harm['rms_ratio']:.6f}\n")
        f.write(f"  PSNR [dB]  = {metrics_harm['psnr_db']:.2f}\n")
        f.write(f"  Target tier = {eval_harm['tier']}\n")
        f.write(f"  Status      = {eval_harm['status']}\n")
        f.write(f"  Targets     = r>={eval_harm['target']['pearson_r']:.3f}, "
                f"{eval_harm['target']['rms_ratio_min']:.2f}<=RMS<={eval_harm['target']['rms_ratio_max']:.2f}, "
                f"PSNR>={eval_harm['target']['psnr_db']:.1f} dB\n")
    print(f"  Saved: {metrics_path}")
    print(f"  Scientific parity status: {overall_status}")
    print(f"parity_status: {overall_status}")
    if overall_status != "PASS" and not args.allow_failure:
        raise SystemExit(
            "Scientific parity targets were not met for the B-mode transducer workflow. "
            "Re-run with --allow-failure only when collecting diagnostics."
        )

    print("\nDone.")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
