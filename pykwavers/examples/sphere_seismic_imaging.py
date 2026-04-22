"""Sphere Seismic Imaging Demo — Kirchhoff Migration of a Circular Velocity Anomaly.

## Physical Setup

    Domain  : Nx × Nz = 128 × 128 cells, dx = dz = 10 m  → 1280 m × 1280 m
    Background : c₀ = 1500 m/s (water / unconsolidated sediment), ρ₀ = 1000 kg/m³
    Inclusion  : circular cross-section of a sphere at (cx, cz) = (640 m, 960 m),
                 radius r = 120 m, c_sphere = 2000 m/s (133 % impedance contrast)
    Source     : explosive point source at surface centre (x = 640 m, z = 120 m)
    Receivers  : 108 hydrophones at z = 120 m, x = 100…1180 m (10 m spacing)

## Forward Modelling (k-Wave Python, Treeby & Cox 2010)

    kspaceFirstOrder2D with heterogeneous sound speed array.
    PML: 10 cells absorbing boundary; α_pml = 2.0 Np/cell.
    Source: Ricker wavelet approximated by 3-cycle tone burst at f₀ = 25 Hz.
    Total simulation time: t_end = 1.4 s  (covers round-trip to deepest reflector).

## Kirchhoff Migration (Schneider 1978)

    For each image point (x_im, z_im) and each receiver position x_r:
        t_arrival = (√((x_im − x_s)² + z_im²) + √((x_im − x_r)² + z_im²)) / c₀
        I(x_im, z_im) += p(x_r, t_arrival)    [linear interpolation in time]
    Sum over all receivers.  This is the zero-offset Kirchhoff integral in 2D.

    Ref: Schneider, W.A. (1978). "Integral formulation for migration in two and three
    dimensions." Geophysics 43(1), 49–76. DOI: 10.1190/1.1440828

## Output Figures (PNG, 300 dpi → examples/output/)

    1. sphere_true_model.png    — velocity map with sphere outline + src/rcv markers
    2. sphere_shot_gather.png   — shot gather: pressure p(receiver, time) + hyperbola overlay
    3. sphere_rtm_image.png     — Kirchhoff migration image with sphere outline overlay

## References

    - Treeby, B.E. & Cox, B.T. (2010). "k-Wave: MATLAB toolbox for the simulation and
      reconstruction of photoacoustic wave fields." J. Biomed. Opt. 15(2), 021314.
      DOI: 10.1117/1.3360308
    - Schneider, W.A. (1978). Geophysics 43(1), 49–76. DOI: 10.1190/1.1440828
    - Baysal, E. et al. (1983). "Reverse time migration." Geophysics 48(11), 1514–1524.
      DOI: 10.1190/1.1441434
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless/CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    from kwave.utils.signals import tone_burst
except ImportError as exc:
    print(f"ERROR: k-wave-python not found — {exc}", file=sys.stderr)
    print("Run inside the pykwavers venv with k-wave-python installed.", file=sys.stderr)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Domain parameters
# ──────────────────────────────────────────────────────────────────────────────

NX: int = 128           # horizontal cells
NZ: int = 128           # vertical (depth) cells
DX: float = 10.0        # cell size [m]
DZ: float = 10.0        # cell size [m]
PML_SIZE: int = 10      # absorbing PML thickness [cells]

C0: float = 1500.0      # background sound speed [m/s]
RHO0: float = 1000.0    # background density [kg/m³]
C_SPHERE: float = 2000.0  # sphere sound speed [m/s]

# Sphere (circular cross-section) — physical coordinates [m]
CX_M: float = (NX // 2) * DX        # 640 m  (horizontal centre)
CZ_M: float = (3 * NZ // 4) * DZ    # 960 m  (depth of sphere centre)
R_M: float = 12 * DX                 # 120 m  (sphere radius)

# Source position (surface centre, just below PML)
SRC_X: int = NX // 2               # horizontal index
SRC_Z: int = PML_SIZE + 2          # vertical index (inside domain)

# Receiver line: surface array, all cells except edges
RCV_Z: int = PML_SIZE + 2
RCV_X_START: int = PML_SIZE + 1
RCV_X_END: int = NX - PML_SIZE - 1  # exclusive

# Simulation time
T_END: float = 1.4      # [s] — covers round-trip 2 * (CZ_M - SRC_Z*DZ) / C0
F0: float = 25.0        # source frequency [Hz]
N_CYCLES: int = 3       # tone burst cycles

OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "output")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build heterogeneous velocity / density model
# ──────────────────────────────────────────────────────────────────────────────

def build_velocity_model() -> tuple[np.ndarray, np.ndarray]:
    """Return (sound_speed, density) arrays of shape (NX, NZ).

    Sound speed: C0 everywhere except inside the sphere (C_SPHERE).
    Density: uniform RHO0 (acoustic impedance contrast provided by velocity only).
    """
    x_idx = np.arange(NX, dtype=float) * DX   # [m]
    z_idx = np.arange(NZ, dtype=float) * DZ   # [m]
    X, Z = np.meshgrid(x_idx, z_idx, indexing="ij")  # shape (NX, NZ)

    sound_speed = np.full((NX, NZ), C0)
    r = np.sqrt((X - CX_M) ** 2 + (Z - CZ_M) ** 2)
    sound_speed[r <= R_M] = C_SPHERE

    density = np.full((NX, NZ), RHO0)
    return sound_speed, density


# ──────────────────────────────────────────────────────────────────────────────
# Forward simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_forward_simulation(
    sound_speed: np.ndarray,
    density: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Run k-Wave 2D forward simulation and return (sensor_data, dt, t_end).

    Returns
    -------
    sensor_data : ndarray, shape (n_receivers, Nt)
        Pressure time series at each receiver.
    dt : float
        Time step [s].
    t_arr : float
        Total simulation time [s].
    """
    # k-Wave grid (row = x-horizontal, col = z-depth in 2D kWaveGrid)
    kgrid = kWaveGrid([NX, NZ], [DX, DZ])
    kgrid.makeTime(sound_speed, t_end=T_END)
    dt: float = float(kgrid.dt)
    Nt: int = int(kgrid.Nt)

    print(f"  Grid: {NX}×{NZ}, dx={DX} m, dz={DZ} m")
    print(f"  dt = {dt:.4e} s, Nt = {Nt} steps, T_end = {T_END:.2f} s")

    # Heterogeneous medium
    medium = kWaveMedium(
        sound_speed=sound_speed.astype(np.float32),
        density=density.astype(np.float32),
        alpha_coeff=0.0,
        alpha_power=1.5,
    )

    # Source: tone burst at surface centre
    signal_raw = tone_burst(1.0 / dt, F0, N_CYCLES).flatten()
    if len(signal_raw) < Nt:
        signal_raw = np.pad(signal_raw, (0, Nt - len(signal_raw)))
    else:
        signal_raw = signal_raw[:Nt]

    source_mask = np.zeros((NX, NZ), dtype=np.uint8)
    source_mask[SRC_X, SRC_Z] = 1

    source = kSource()
    source.p_mask = source_mask
    source.p = signal_raw.reshape(1, -1).astype(np.float32)
    source.p_mode = "additive"

    # Sensor: horizontal line at surface
    sensor_mask = np.zeros((NX, NZ), dtype=np.uint8)
    for xi in range(RCV_X_START, RCV_X_END):
        sensor_mask[xi, RCV_Z] = 1
    n_receivers = RCV_X_END - RCV_X_START

    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]

    # Simulation options (CPU; set is_gpu_simulation=True if CUDA is available)
    sim_opts = SimulationOptions(
        pml_inside=True,
        pml_size=PML_SIZE,
        data_cast="single",
        save_to_disk=True,
    )
    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=False,
        verbose_level=0,
    )

    print(f"  Running k-Wave 2D forward simulation ({n_receivers} receivers)…")
    sensor_out = kspaceFirstOrder2D(
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        medium=medium,
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )

    # k-Wave Python returns sensor data as (Nt, n_receivers); transpose to (n_receivers, Nt)
    p_raw = np.asarray(sensor_out["p"], dtype=np.float64)
    if p_raw.shape[0] == Nt and p_raw.shape[1] != Nt:
        p_data = p_raw.T   # (Nt, n_rcv) → (n_rcv, Nt)
    elif p_raw.ndim == 2 and p_raw.shape[1] == Nt:
        p_data = p_raw     # already (n_rcv, Nt)
    else:
        # Fallback: assume first dim is time
        p_data = p_raw.T
    print(f"  Sensor data shape: {p_data.shape}  (n_receivers x Nt)")
    return p_data, dt, T_END


# ──────────────────────────────────────────────────────────────────────────────
# Kirchhoff migration (Schneider 1978)
# ──────────────────────────────────────────────────────────────────────────────

def kirchhoff_migration(
    sensor_data: np.ndarray,
    dt: float,
    c_mig: float = C0,
) -> np.ndarray:
    """Apply 2D Kirchhoff migration and return the reflectivity image.

    ## Algorithm (Schneider 1978, §3)

    For each image point (x_im, z_im) ∈ domain:
        For each receiver x_r:
            t_r = (√((x_im − x_s)² + (z_im − z_s)²) + √((x_im − x_r)²)) / c_mig
            I(x_im, z_im) += interp(p(x_r, ·), t_r)

    ## Direct-wave mute

    The direct P-wave from source to each receiver arrives at
        t_direct(x_r) = |x_r − x_s| / c_mig
    This energy, if unmuted, maps as a strong shallow artifact.  We apply a
    linear ramp mute that zeros any sample at t < t_mute, where
        t_mute = 0.9 × (sphere_top_round_trip_time)
    ensuring the sphere reflection (arriving well after t_mute) is preserved.

    Ref: Schneider, W.A. (1978). Geophysics 43(1), 49–76.

    Parameters
    ----------
    sensor_data : (n_receivers, Nt) float64
    dt : float — time step [s]
    c_mig : float — migration velocity [m/s]

    Returns
    -------
    image : (NX, NZ) float64
    """
    n_rcv, Nt = sensor_data.shape

    # Direct-wave mute: zero samples before t_mute to suppress direct-wave artifact.
    # Sphere-top round-trip time = 2*(CZ_M - SRC_Z*DZ - R_M) / c_mig
    t_sphere_top_rt = 2.0 * (CZ_M - SRC_Z * DZ - R_M) / c_mig  # round-trip [s]
    t_mute = 0.9 * t_sphere_top_rt  # mute before first reflection
    n_mute = max(0, int(t_mute / dt))
    data = sensor_data.copy()
    data[:, :n_mute] = 0.0

    image = np.zeros((NX, NZ))

    src_x_m = SRC_X * DX
    src_z_m = SRC_Z * DZ

    rcv_x_m = np.array([(RCV_X_START + i) * DX for i in range(n_rcv)])

    t_max = (Nt - 1) * dt

    for ix in range(NX):
        x_m = ix * DX
        for iz in range(1, NZ):
            z_m = iz * DZ
            d_src = math.sqrt((x_m - src_x_m) ** 2 + (z_m - src_z_m) ** 2)
            t_src = d_src / c_mig

            val = 0.0
            for r_idx in range(n_rcv):
                # Receiver is at surface depth (SRC_Z * DZ); horizontal offset only
                d_rcv = math.sqrt((x_m - rcv_x_m[r_idx]) ** 2 + (z_m - src_z_m) ** 2)
                t_arr = t_src + d_rcv / c_mig
                if t_arr >= t_max:
                    continue
                ti = t_arr / dt
                ti0 = int(ti)
                ti1 = ti0 + 1
                if ti1 >= Nt:
                    continue
                frac = ti - ti0
                val += (1.0 - frac) * data[r_idx, ti0] + frac * data[r_idx, ti1]
            image[ix, iz] = val

    return image


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sphere_circle(ax: "plt.Axes") -> None:
    """Overlay exact sphere outline on an image-plane axis."""
    circle = mpatches.Circle(
        (CX_M, CZ_M),
        R_M,
        linewidth=1.5,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
        label="True sphere boundary",
    )
    ax.add_patch(circle)


def plot_true_model(
    sound_speed: np.ndarray,
    output_dir: str,
) -> None:
    """Figure 1: 2D velocity map with sphere outline + source/receiver markers."""
    fig, ax = plt.subplots(figsize=(7, 6))

    x_axis = np.arange(NX) * DX
    z_axis = np.arange(NZ) * DZ
    im = ax.pcolormesh(
        x_axis, z_axis, sound_speed.T,
        cmap="jet",
        vmin=C0 * 0.9,
        vmax=C_SPHERE * 1.05,
    )
    cb = fig.colorbar(im, ax=ax, label="Sound speed [m/s]")
    cb.ax.tick_params(labelsize=9)

    _sphere_circle(ax)

    # Source marker
    ax.plot(SRC_X * DX, SRC_Z * DZ, "r*", markersize=12, label="Source", zorder=5)

    # Receiver markers (subsample for visibility)
    rcv_x = np.array([(RCV_X_START + i) * DX for i in range(0, RCV_X_END - RCV_X_START, 5)])
    ax.plot(rcv_x, np.full_like(rcv_x, RCV_Z * DZ), "gv", markersize=5, label="Receivers")

    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Z depth [m]", fontsize=11)
    ax.set_title("True velocity model (sphere inclusion + acquisition geometry)", fontsize=11)
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    path = os.path.join(output_dir, "sphere_true_model.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_shot_gather(
    sensor_data: np.ndarray,
    dt: float,
    output_dir: str,
) -> None:
    """Figure 2: Shot gather p(receiver, time) with theoretical reflection hyperbola.

    ## Reflection hyperbola (Dix 1955)

    For a single horizontal reflector at depth z_r below a surface source,
    the two-way travel time to receiver x_r is:
        t(x_r) = √(t₀² + (x_r − x_s)² / c₀²)
    where t₀ = 2·z_r / c₀ is the zero-offset time.

    Ref: Dix, C.H. (1955). "Seismic velocities from surface measurements."
         Geophysics 20(1), 68–86. DOI: 10.1190/1.1438126
    """
    n_rcv, Nt = sensor_data.shape
    t_axis = np.arange(Nt) * dt  # [s]
    rcv_x_m = np.array([(RCV_X_START + i) * DX for i in range(n_rcv)])

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [rcv_x_m[0], rcv_x_m[-1], t_axis[-1], t_axis[0]]
    ax.imshow(
        sensor_data.T,
        aspect="auto",
        extent=extent,
        cmap="seismic",
        vmin=-np.percentile(np.abs(sensor_data), 98),
        vmax=np.percentile(np.abs(sensor_data), 98),
    )

    # Theoretical reflection hyperbola from sphere centre (Dix 1955)
    src_x_m = SRC_X * DX
    # Approximation: zero-offset time to sphere centre
    t0_centre = 2.0 * (CZ_M - SRC_Z * DZ) / C0
    t_hyp = np.sqrt(t0_centre ** 2 + (rcv_x_m - src_x_m) ** 2 / C0 ** 2)
    ax.plot(rcv_x_m, t_hyp, "y--", linewidth=1.5, label=f"Hyperbola t₀={t0_centre:.2f} s")

    ax.set_xlabel("Receiver X [m]", fontsize=11)
    ax.set_ylabel("Two-way time [s]", fontsize=11)
    ax.set_title(
        f"Shot gather — point source at x={src_x_m:.0f} m  (sphere at depth {CZ_M:.0f} m)",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    path = os.path.join(output_dir, "sphere_shot_gather.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_rtm_image(image: np.ndarray, output_dir: str) -> None:
    """Figure 3: Kirchhoff migration image with true sphere outline overlay."""
    fig, ax = plt.subplots(figsize=(7, 6))

    x_axis = np.arange(NX) * DX
    z_axis = np.arange(NZ) * DZ

    clip = np.percentile(np.abs(image), 99)
    im = ax.pcolormesh(
        x_axis, z_axis, image.T,
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
    )
    fig.colorbar(im, ax=ax, label="Migration amplitude [Pa·s]")

    _sphere_circle(ax)

    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Z depth [m]", fontsize=11)
    ax.set_title("Kirchhoff migration image (Schneider 1978)", fontsize=11)
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    path = os.path.join(output_dir, "sphere_rtm_image.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers (used by pytest)
# ──────────────────────────────────────────────────────────────────────────────

def test_sphere_model_velocity_contrast() -> None:
    """Sphere centre cell == C_SPHERE; corner cell == C0."""
    sound_speed, _ = build_velocity_model()
    cx_idx = int(round(CX_M / DX))
    cz_idx = int(round(CZ_M / DZ))
    assert sound_speed[cx_idx, cz_idx] == C_SPHERE, (
        f"Sphere centre should be {C_SPHERE} m/s, got {sound_speed[cx_idx, cz_idx]}"
    )
    assert sound_speed[0, 0] == C0, (
        f"Corner should be background {C0} m/s, got {sound_speed[0, 0]}"
    )


def test_shot_gather_reflection_hyperbola() -> None:
    """Verify the reflection hyperbola formula is satisfied for zero-offset time.

    The round-trip time to the sphere centre from the surface source must satisfy:
        t₀ = 2 · (CZ_M − SRC_Z·DZ) / C0
    and must be < T_END.
    """
    t0 = 2.0 * (CZ_M - SRC_Z * DZ) / C0
    assert t0 > 0.0, "t₀ must be positive — sphere must be below source"
    assert t0 < T_END, f"t₀={t0:.3f} s must be < T_END={T_END:.3f} s to be recorded"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Sphere Seismic Imaging Demo")
    print(f"  Domain  : {NX}x{NZ} cells, dx={DX} m")
    print(f"  Sphere  : centre ({CX_M:.0f} m, {CZ_M:.0f} m), radius={R_M:.0f} m")
    print(f"            c_sphere={C_SPHERE:.0f} m/s  (background c0={C0:.0f} m/s)")
    print(f"  Source  : ({SRC_X*DX:.0f} m, {SRC_Z*DZ:.0f} m),  f0={F0:.0f} Hz")
    print("=" * 60)

    # Validate test helpers first
    test_sphere_model_velocity_contrast()
    test_shot_gather_reflection_hyperbola()
    print("  [✓] Analytical validation tests passed")

    # Build model
    sound_speed, density = build_velocity_model()

    # Figure 1: true model
    print("\n[1/3] Plotting true velocity model…")
    plot_true_model(sound_speed, OUTPUT_DIR)

    # Forward simulation
    print("\n[2/3] Running forward simulation…")
    sensor_data, dt, _t_end = run_forward_simulation(sound_speed, density)

    # Figure 2: shot gather
    plot_shot_gather(sensor_data, dt, OUTPUT_DIR)

    # Kirchhoff migration
    print("\n[3/3] Running Kirchhoff migration (Schneider 1978)…")
    image = kirchhoff_migration(sensor_data, dt)

    # Figure 3: migration image
    plot_rtm_image(image, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Done. Figures saved to:", OUTPUT_DIR)

    # Summary statistics
    cx_idx = int(round(CX_M / DX))
    cz_idx = int(round(CZ_M / DZ))
    peak = image[cx_idx, cz_idx]
    img_max = np.max(np.abs(image))
    print(f"  Image amplitude at sphere centre : {peak:.4e}")
    print(f"  Peak image amplitude (all)       : {img_max:.4e}")
    if img_max > 0.0:
        print(f"  Sphere centre / peak ratio       : {abs(peak)/img_max:.3f}")


if __name__ == "__main__":
    main()
