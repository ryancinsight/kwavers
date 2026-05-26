"""
Chapter 26: Low-Intensity Ultrasound Neuromodulation
====================================================

Executable simulations for docs/book/neuromodulation.md.

The chapter models microbubble-free transcranial ultrasound stimulation as a
coupled acoustic, thermal, mechanochemical, and neural-response problem.  The
script is deterministic and emits figures plus a metrics.json file under
docs/book/figures/ch26/.

Physics contract
----------------
All wave-physics and biophysics computation (membrane tension, channel gating,
LIF ODE, Gaussian beam envelope, Pennes bioheat) execute in Rust via pykwavers.
Python contains only orchestration, parameter dataclasses, plotting, and
thin wrapper functions that marshal arguments to Rust and package the results.
No physics loops, no ODE steppers, no logistic computations appear in this file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pykwavers as kw
    _HAS_KW = True
except ImportError:
    _HAS_KW = False
    raise RuntimeError(
        "pykwavers not found — build with `maturin develop --release` "
        "from the pykwavers directory."
    )

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch26")

RHO_BRAIN = 1040.0
C_BRAIN = 1540.0
CP_BRAIN = 3600.0
K_BRAIN = 0.51          # W/(m·K)  Duck 1990 / IT'IS v4.1
RHO_BLOOD = 1060.0
CP_BLOOD = 3600.0
PERFUSION_S = 0.012
ALPHA_BRAIN_NP_M = 4.0
T0_C = 37.0
CELL_RADIUS_M = 10.0e-6
BODY_TEMPERATURE_K = 310.0  # IT'IS v4.1


@dataclass(frozen=True)
class Protocol:
    frequency_hz: float = 500.0e3
    pressure_pa: float = 300.0e3
    duty_cycle: float = 0.05
    sonication_s: float = 30.0
    pulse_repetition_hz: float = 10.0
    lateral_fwhm_m: float = 5.0e-3
    axial_fwhm_m: float = 30.0e-3


@dataclass(frozen=True)
class Channel:
    name: str
    half_tension_mn_m: float
    slope_mn_m: float
    reversal_mv: float
    conductance_weight: float


@dataclass(frozen=True)
class ThermalResult:
    time_s: np.ndarray
    temperature_c: np.ndarray
    cem43_min: np.ndarray


CHANNELS = (
    Channel("TRPC1/TRPP2 calcium cluster", 0.085, 0.018, 30.0, 1.0),
    Channel("Piezo1 calcium channel", 0.18, 0.035, 10.0, 0.65),
    Channel("TREK/TRAAK K2P leak", 0.42, 0.075, -90.0, -0.45),
)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch26/{name}.{{pdf,png}}")


def mechanical_index(pressure_pa: np.ndarray | float, frequency_hz: float) -> np.ndarray:
    """MI = |p| [MPa] / sqrt(f [MHz]) via kw.mechanical_index_field (Rust kernel)."""
    arr = np.asarray(pressure_pa, dtype=np.float64)
    shape = arr.shape
    result = np.asarray(
        kw.mechanical_index_field(np.ascontiguousarray(arr.ravel()), float(frequency_hz))
    )
    return result.reshape(shape) if shape else float(result[0])


def intensity_sppa_w_m2(pressure_pa: np.ndarray | float) -> np.ndarray:
    """ISPPA = p²/(2ρc) [W/m²] via kw.acoustic_intensity_from_amplitude (Rust kernel)."""
    arr = np.asarray(pressure_pa, dtype=np.float64)
    shape = arr.shape
    result = np.asarray(
        kw.acoustic_intensity_from_amplitude(
            np.ascontiguousarray(arr.ravel()), float(RHO_BRAIN), float(C_BRAIN)
        )
    )
    return result.reshape(shape) if shape else float(result[0])


def acoustic_energy_tension_mn_m(pressure_pa: np.ndarray | float) -> np.ndarray:
    """Acoustic membrane tension from peak pressure via Rust (Timoshenko 1959; Sarvazyan 2010).

    Delegates to ``kw.compute_acoustic_membrane_tension_py``.  Radiation-pressure
    derivation (P_rad = I/c) and Laplace thin-shell equilibrium (ΔT = P_rad·R/2)
    execute in Rust.  Returns tension in mN/m, same shape as input.
    """
    arr = np.asarray(pressure_pa, dtype=np.float64)
    shape = arr.shape
    flat = np.ascontiguousarray(arr.ravel())
    result = np.asarray(
        kw.compute_acoustic_membrane_tension_py(
            flat,
            density_kg_m3=RHO_BRAIN,
            sound_speed_m_s=C_BRAIN,
            cell_radius_m=CELL_RADIUS_M,
        ),
        dtype=np.float64,
    )
    return result.reshape(shape) if shape else result[0]


def open_probability(tension_mn_m: np.ndarray | float, channel: Channel) -> np.ndarray:
    """Boltzmann open probability from membrane tension via Rust (Sukharev 1997; Cox 2016).

    Delegates to ``kw.boltzmann_open_probability_py``.  The slope parameterisation
    ``σ = k_B·θ / A_gate`` converts to gating area inside Rust.  All exponential
    evaluation executes in Rust.  Returns array of same shape as input.
    """
    arr = np.asarray(tension_mn_m, dtype=np.float64)
    shape = arr.shape
    flat = np.ascontiguousarray(arr.ravel())
    result = np.asarray(
        kw.boltzmann_open_probability_py(
            flat,
            half_tension_mn_m=channel.half_tension_mn_m,
            slope_mn_m=channel.slope_mn_m,
            temperature_k=BODY_TEMPERATURE_K,
        ),
        dtype=np.float64,
    )
    return result.reshape(shape) if shape else result[0]


def coupled_channel_drive(pressure_pa: np.ndarray | float) -> np.ndarray:
    """Normalised mechanochemical channel drive from acoustic pressure via Rust.

    Delegates to ``kw.coupled_channel_drive_py``.  Tension derivation, per-channel
    Boltzmann gating, and weighted normalisation all execute in Rust.

    Pipeline (Rust):
    1. p → I = p²/(2ρc)          [W/m²]
    2. I → ΔT = I·R/(2c)         [N/m]  (Laplace thin-shell)
    3. ΔT → P_open,k = Boltzmann(ΔT; T_half,k, slope_k, θ)
    4. drive = clamp(Σ_k w_k·P_open,k / Σ_k |w_k|, −1, 1)
    """
    arr = np.asarray(pressure_pa, dtype=np.float64)
    shape = arr.shape
    flat = np.ascontiguousarray(arr.ravel())
    result = np.asarray(
        kw.coupled_channel_drive_py(
            flat,
            half_tensions_mn_m=[ch.half_tension_mn_m for ch in CHANNELS],
            slopes_mn_m=[ch.slope_mn_m for ch in CHANNELS],
            conductance_weights=[ch.conductance_weight for ch in CHANNELS],
            density_kg_m3=RHO_BRAIN,
            sound_speed_m_s=C_BRAIN,
            cell_radius_m=CELL_RADIUS_M,
            temperature_k=BODY_TEMPERATURE_K,
        ),
        dtype=np.float64,
    )
    return result.reshape(shape) if shape else result[0]


def gaussian_focus(
    protocol: Protocol,
    shape: tuple[int, int, int] = (81, 81, 61),
    spacing_m: float = 1.0e-3,
) -> dict[str, np.ndarray]:
    """Analytical 3-D Gaussian beam pressure field via Rust (Goodman 2005 §3.3).

    Delegates to ``kw.gaussian_beam_pressure_field_py``.  Grid construction,
    sigma derivation, and Gaussian exponent evaluation execute in Rust.
    """
    field = kw.gaussian_beam_pressure_field_py(
        nx=shape[0],
        ny=shape[1],
        nz=shape[2],
        dx_m=spacing_m,
        dy_m=spacing_m,
        dz_m=spacing_m,
        peak_pressure_pa=protocol.pressure_pa,
        lateral_fwhm_m=protocol.lateral_fwhm_m,
        axial_fwhm_m=protocol.axial_fwhm_m,
    )
    return {
        "x": np.asarray(field["x"], dtype=np.float64),
        "y": np.asarray(field["y"], dtype=np.float64),
        "z": np.asarray(field["z"], dtype=np.float64),
        "pressure": np.asarray(field["pressure"], dtype=np.float64),
    }


def pennes_temperature(intensity_spta_w_m2: float, sonication_s: float, dt_s: float = 0.02) -> ThermalResult:
    """Pennes bioheat ODE for a single focal point (pykwavers Rust solver).

    For a 1-point simulation (nx=ny=nz=1) spatial diffusion is zero; the
    solver integrates only the Pennes perfusion term. Stability: dt_max for
    perfusion ≈ 2·ρ·cp/(WB·ρB·cpB) ≈ 163 s >> dt_s. CEM43 is accumulated
    via kw.compute_cem43 on downsampled growing sub-trajectories.
    """
    n_steps = int(np.ceil(sonication_s / dt_s))
    Q_field = np.array([[[2.0 * ALPHA_BRAIN_NP_M * intensity_spta_w_m2]]])
    sensor_mask = np.ones((1, 1, 1), dtype=bool)
    res = kw.ThermalSimulation(
        1, 1, 1, 1e-3, 1e-3, 1e-3,
        thermal_conductivity=K_BRAIN,
        density=RHO_BRAIN, specific_heat=CP_BRAIN,
        enable_bioheat=True, perfusion_rate=PERFUSION_S,
        blood_density=RHO_BLOOD, blood_specific_heat=CP_BLOOD,
        arterial_temperature=T0_C, initial_temperature=T0_C,
        track_thermal_dose=False,
    ).run(n_steps, dt_s, heat_source=Q_field, sensor_mask=sensor_mask)
    T_arr = np.asarray(res.temperature_at_sensors)[0]
    time_arr = np.asarray(res.time)
    # Cumulative CEM43 via kw.compute_cem43 on 30 downsampled sub-trajectories
    n_q = min(30, n_steps)
    q_idx = np.linspace(0, n_steps - 1, n_q, dtype=int)
    cem43_sp = np.array([kw.compute_cem43(T_arr[: i + 1].astype(float), dt_s) for i in q_idx])
    cem43_arr = np.interp(np.arange(n_steps), q_idx, cem43_sp)
    return ThermalResult(time_arr, T_arr, cem43_arr)


def pulse_envelope(time_s: np.ndarray, protocol: Protocol) -> np.ndarray:
    period = 1.0 / protocol.pulse_repetition_hz
    width = protocol.duty_cycle * period
    return ((time_s % period) < width).astype(float)


def neural_response(
    protocol: Protocol,
    duration_s: float = 4.0,
    dt_s: float = 0.001,
    i_max_pa: float = 300.0,
    smoothing_sigma_s: float = 0.200,
) -> dict[str, np.ndarray]:
    """LIF neuron driven by mechanosensitive ion current via Rust (Koch 1999).

    Pipeline (all heavy computation in Rust):

    1. ``coupled_channel_drive_py`` → normalised drive ∈ [−1, 1] per time step.
    2. Scale to ion current: I_ion = max(drive, 0) × I_max [A].
       Depolarising channels (drive > 0) drive inward current; inhibitory drive
       (drive < 0, K⁺ channels) reduces below rheobase (handled by LIF G_leak).
    3. ``simulate_lif_neuron_py`` → membrane voltage trace V(t) and spike times
       (forward-Euler LIF, Koch 1999 canonical params).
    4. Response probability = Gaussian-smoothed instantaneous spike density
       normalised to theoretical maximum firing rate f_max = 1/(τ_ref + τ_m).

    Parameters
    ----------
    protocol:
        ``Protocol`` dataclass carrying frequency, pressure, duty cycle.
    duration_s:
        Simulation duration [s].
    dt_s:
        Time step [s]; stable for LIF when dt ≪ τ_m = 10 ms.
    i_max_pa:
        Maximum depolarising current at unit channel drive [pA].
    smoothing_sigma_s:
        Gaussian kernel σ for spike-density smoothing [s].
    """
    from scipy.ndimage import gaussian_filter1d

    time_s = np.arange(0.0, duration_s + dt_s, dt_s)
    n = time_s.size
    envelope = pulse_envelope(time_s, protocol)
    pressure = protocol.pressure_pa * envelope

    # Mechanochemical channel drive → ion current (Rust).
    channel_drive = coupled_channel_drive(pressure)
    # Only depolarising (positive) drive generates injected current; inhibitory
    # drive (negative weights from K⁺ channels) reduces net input toward zero.
    i_ion_a = np.maximum(channel_drive, 0.0) * (i_max_pa * 1.0e-12)

    # LIF ODE in Rust (Koch 1999 canonical parameters).
    lif_result = kw.simulate_lif_neuron_py(i_ion_a, dt_s)
    voltage_v = np.asarray(lif_result["voltage_v"], dtype=np.float64)
    spike_times_s = np.asarray(lif_result["spike_times_s"], dtype=np.float64)

    # Response probability: Gaussian-smoothed spike density / f_max.
    # f_max = 1 / (τ_ref + τ_m) = 1 / (2ms + 10ms) ≈ 83 Hz.
    f_max_hz = 1.0 / (2.0e-3 + 10.0e-3)
    spike_train = np.zeros(n, dtype=np.float64)
    if spike_times_s.size > 0:
        spike_idx = np.rint(spike_times_s / dt_s).astype(int)
        spike_idx = spike_idx[(spike_idx >= 0) & (spike_idx < n)]
        spike_train[spike_idx] = 1.0
    sigma_samples = smoothing_sigma_s / dt_s
    smoothed = gaussian_filter1d(spike_train / dt_s, sigma=sigma_samples)
    response = np.clip(smoothed / f_max_hz, 0.0, 1.0)

    return {
        "time_s": time_s,
        "envelope": envelope,
        "voltage_mv": voltage_v * 1.0e3,    # V → mV for plotting
        "response_probability": response,
        "channel_drive": channel_drive,
        "spike_times_s": spike_times_s,
    }


def cavitation_risk(mi: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-18.0 * (np.asarray(mi, dtype=float) - 0.7)))


def guidance_map() -> dict[str, np.ndarray]:
    """Thermal safety parameter sweep via batch pykwavers ThermalSimulation.

    Runs a single ThermalSimulation(140, 120, 1) call instead of a 16 800-point
    serial loop. DX_BATCH=0.1 m → Fourier number Fo = D·t/DX² ≈ 4×10⁻⁴ ≪ 1,
    so neighbouring grid cells are thermally decoupled and each cell integrates
    only its own Q and perfusion term — equivalent to 16 800 independent 1-point
    Pennes ODEs. Stability: dt_max = DX²/(2D) ≈ 37 000 s ≫ dt=0.25 s.
    """
    pressure_kpa = np.linspace(50.0, 900.0, 140)
    duty_percent = np.linspace(1.0, 30.0, 120)
    p_grid, dc_grid = np.meshgrid(pressure_kpa * 1.0e3, duty_percent / 100.0, indexing="ij")
    protocol = Protocol()
    # ISPPA = p²/(2ρc) via Rust kernel; ISPTA = ISPPA × duty_cycle
    ispta = intensity_sppa_w_m2(p_grid) * dc_grid
    mi = mechanical_index(p_grid, protocol.frequency_hz)
    drive = coupled_channel_drive(p_grid)

    # Batch Pennes bioheat over (140, 120) parameter grid
    # Q = α·p²/(ρ·c)·duty [W/m³] via kw.acoustic_heat_source_density (Rust)
    DX_BATCH = 0.1  # m — large enough to decouple neighbouring cells
    _q_isppa = np.asarray(
        kw.acoustic_heat_source_density(
            np.ascontiguousarray(p_grid.ravel().astype(np.float64)),
            float(ALPHA_BRAIN_NP_M), float(RHO_BRAIN), float(C_BRAIN),
        )
    ).reshape(p_grid.shape)
    Q_batch = (_q_isppa * dc_grid)[:, :, np.newaxis].astype(float)  # (140,120,1) W/m³
    n_steps_g = int(protocol.sonication_s / 0.25)
    res_g = kw.ThermalSimulation(
        140, 120, 1, DX_BATCH, DX_BATCH, DX_BATCH,
        thermal_conductivity=K_BRAIN,
        density=RHO_BRAIN, specific_heat=CP_BRAIN,
        enable_bioheat=True, perfusion_rate=PERFUSION_S,
        blood_density=RHO_BLOOD, blood_specific_heat=CP_BLOOD,
        arterial_temperature=T0_C, initial_temperature=T0_C,
        track_thermal_dose=True,
    ).run(n_steps_g, 0.25, heat_source=Q_batch)
    delta_t = np.asarray(res_g.temperature)[:, :, 0] - T0_C  # (140,120)
    cem43 = np.asarray(res_g.thermal_dose)[:, :, 0]           # (140,120)

    feasible = (mi <= 1.9) & (ispta <= 7200.0) & (delta_t < 2.0) & (cem43 < 0.25) & (cavitation_risk(mi) < 0.10)
    objective = np.where(feasible, drive, np.nan)
    return {
        "pressure_kpa": pressure_kpa,
        "duty_percent": duty_percent,
        "mechanical_index": mi,
        "ispta_w_cm2": ispta / 1.0e4,
        "delta_t_c": delta_t,
        "cem43_min": cem43,
        "channel_drive": drive,
        "feasible": feasible,
        "objective": objective,
    }


def plot_acoustic_field(protocol: Protocol) -> dict[str, float]:
    field = gaussian_focus(protocol)
    pressure = field["pressure"]
    mid_y = pressure.shape[1] // 2
    p_kpa = pressure[:, mid_y, :].T / 1.0e3
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    extent = [
        field["x"][:, 0, 0].min() * 1.0e3,
        field["x"][:, 0, 0].max() * 1.0e3,
        field["z"][0, 0, :].min() * 1.0e3,
        field["z"][0, 0, :].max() * 1.0e3,
    ]
    image = ax.imshow(p_kpa, origin="lower", extent=extent, cmap="magma", aspect="auto")
    x_axis_mm = field["x"][:, 0, 0] * 1.0e3
    z_axis_mm = field["z"][0, 0, :] * 1.0e3
    ax.contour(x_axis_mm, z_axis_mm, p_kpa, levels=[protocol.pressure_pa / 2.0e3], colors="cyan", linewidths=1.2)
    ax.set_xlabel("Lateral position (mm)")
    ax.set_ylabel("Axial position (mm)")
    ax.set_title("In-situ pressure field for low-intensity transcranial neuromodulation")
    fig.colorbar(image, ax=ax, label="Peak pressure (kPa)")
    fig.tight_layout()
    savefig("fig01_acoustic_focus")
    plt.close(fig)
    return {
        "peak_pressure_pa": float(pressure.max()),
        "peak_mi": float(mechanical_index(pressure.max(), protocol.frequency_hz)),
        "peak_ispta_w_cm2": float(intensity_sppa_w_m2(pressure.max()) * protocol.duty_cycle / 1.0e4),
    }


def plot_response(protocol: Protocol) -> dict[str, float]:
    """LIF neuron response traces for three pressure levels (Koch 1999).

    Three sub-panels:
    - Mechanochemical channel drive (Rust coupled_channel_drive_py).
    - Membrane voltage from Rust LIF ODE [mV].
    - Response probability (Gaussian-smoothed spike density / f_max).
    """
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
    summaries: dict[str, float] = {}
    for pressure_kpa, color in [(150.0, "#2166ac"), (300.0, "#4dac26"), (500.0, "#d6604d")]:
        trial = neural_response(Protocol(pressure_pa=pressure_kpa * 1.0e3))
        label = f"{pressure_kpa:.0f} kPa"
        axes[0].plot(trial["time_s"], trial["channel_drive"], color=color, label=label)
        axes[1].plot(trial["time_s"], trial["voltage_mv"], color=color)
        axes[2].plot(trial["time_s"], trial["response_probability"], color=color)
        summaries[f"peak_response_{int(pressure_kpa)}_kpa"] = float(trial["response_probability"].max())
    axes[0].set_ylabel("Mechanochemical channel drive")
    axes[1].set_ylabel("Membrane voltage (mV)")
    axes[2].set_ylabel("Response probability")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title("LIF neuron response to pulsed tFUS (Rust, Koch 1999)")
    axes[0].legend(loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig("fig02_mechanochemical_response")
    plt.close(fig)
    return summaries


def plot_channels() -> None:
    pressure_kpa = np.linspace(0.0, 900.0, 500)
    tension = acoustic_energy_tension_mn_m(pressure_kpa * 1.0e3)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for ch, color in zip(CHANNELS, ["#2166ac", "#4dac26", "#d6604d"]):
        ax.plot(pressure_kpa, open_probability(tension, ch), color=color, label=ch.name)
    ax.axvspan(100.0, 1000.0, color="#999999", alpha=0.12, label="human study pressure envelope")
    ax.set_xlabel("Peak pressure (kPa)")
    ax.set_ylabel("Open probability")
    ax.set_title("Mechanochemical channel gating from acoustic energy density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig("fig03_channel_activation")
    plt.close(fig)


def plot_safety(protocol: Protocol) -> dict[str, float]:
    """Thermal safety sweep: 160 pressures × 4 duty cycles via single batch call.

    Replaces 640 serial ThermalSimulation(1,1,1) invocations with one
    ThermalSimulation(160, 4, 1).  DX_BATCH = 0.1 m gives Fourier number
    Fo = D·dt/DX² = (K/(ρ·cp))·dt/DX² = (0.51/3.744e6)·0.10/0.01 ≈ 1.4×10⁻⁶ ≪ 1,
    so adjacent cells are thermally decoupled — each (i, j) cell integrates only
    its own Q[i,j,0] and the Pennes perfusion term.
    """
    pressures = np.linspace(50.0e3, 900.0e3, 160)
    duties = [0.01, 0.05, 0.10, 0.20]
    DX_BATCH = 0.1  # m — thermal decoupling grid spacing

    # Q[i, j, 0] = α·p²/(ρ·c)·duty [W/m³] — Pennes heat source; Q = 2α·ISPTA = 2α·ISPPA·duty.
    # Uses kw.acoustic_heat_source_density (Rust) then scales by duty cycle.
    q_grid = np.array(
        [
            [
                float(
                    kw.acoustic_heat_source_density(
                        np.array([p], dtype=np.float64),
                        float(ALPHA_BRAIN_NP_M),
                        float(RHO_BRAIN),
                        float(C_BRAIN),
                    )[0]
                ) * d
                for d in duties
            ]
            for p in pressures
        ],
        dtype=float,
    )  # shape (160, 4)
    Q_batch = q_grid[:, :, np.newaxis]  # shape (160, 4, 1)

    dt_s = 0.10
    n_steps = int(np.ceil(protocol.sonication_s / dt_s))  # 300 steps for 30 s

    res = kw.ThermalSimulation(
        160, 4, 1, DX_BATCH, DX_BATCH, DX_BATCH,
        thermal_conductivity=K_BRAIN,
        density=RHO_BRAIN, specific_heat=CP_BRAIN,
        enable_bioheat=True, perfusion_rate=PERFUSION_S,
        blood_density=RHO_BLOOD, blood_specific_heat=CP_BLOOD,
        arterial_temperature=T0_C, initial_temperature=T0_C,
        track_thermal_dose=True,
    ).run(n_steps, dt_s, heat_source=Q_batch)

    # Final-state spatial arrays — shape (160, 4, 1) → slice to (160, 4)
    delta_t_arr = np.asarray(res.temperature)[:, :, 0] - T0_C  # (160, 4)
    cem43_arr = np.asarray(res.thermal_dose)[:, :, 0]           # (160, 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    summary: dict[str, float] = {}
    for j, (duty, color) in enumerate(
        zip(duties, ["#2166ac", "#4dac26", "#fdae61", "#d6604d"])
    ):
        label = f"DC {100.0 * duty:.0f}%"
        ax1.plot(pressures / 1.0e3, delta_t_arr[:, j], color=color, label=label)
        ax2.semilogy(pressures / 1.0e3, cem43_arr[:, j] + 1.0e-12, color=color, label=label)
        if duty == protocol.duty_cycle:
            summary["default_delta_t_c"] = float(
                np.interp(protocol.pressure_pa, pressures, delta_t_arr[:, j])
            )
            summary["default_cem43_min"] = float(
                np.interp(protocol.pressure_pa, pressures, cem43_arr[:, j])
            )
    ax1.axhline(2.0, color="black", linestyle="--", linewidth=1.0, label="2 C guardrail")
    ax2.axhline(0.25, color="black", linestyle="--", linewidth=1.0, label="0.25 CEM43")
    ax1.set_xlabel("Peak pressure (kPa)")
    ax1.set_ylabel("Temperature rise after 30 s (C)")
    ax2.set_xlabel("Peak pressure (kPa)")
    ax2.set_ylabel("CEM43 (min)")
    ax1.set_title("Pennes thermal rise")
    ax2.set_title("Thermal dose")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax2.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    savefig("fig04_thermal_safety")
    plt.close(fig)
    return summary


def plot_clinical_guidance() -> None:
    studies = [
        ("S1 sensory cortex\nLegon 2014", 500.0, 0.30, 1.0, 1),
        ("Thalamic/S1 targeting\nKim 2023", 250.0, 0.45, 1.0, 2),
        ("Amygdala / affective\nactive trials", 650.0, 0.35, 0.7, 2),
        ("Pallidal PD beta\nEraifej 2026", 555.0, 0.38, 0.5, 2),
        ("Dementia TPS\nBeisteiner 2019", 250.0, 0.20, 0.3, 3),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for idx, (label, freq_khz, mi, duration_s, evidence) in enumerate(studies):
        size = 90 + 70 * evidence
        ax.scatter(freq_khz, mi, s=size, c=duration_s, cmap="viridis", vmin=0.2, vmax=1.2, edgecolor="black")
        ax.text(freq_khz + 12.0, mi + 0.015, label, fontsize=8, va="center")
    ax.axhline(1.9, color="#555555", linestyle="--", linewidth=1.0, label="MI 1.9 diagnostic benchmark")
    ax.axhspan(0.2, 0.7, color="#4dac26", alpha=0.10, label="common neuromodulation MI band")
    ax.set_xlim(180.0, 760.0)
    ax.set_ylim(0.0, 2.05)
    ax.set_xlabel("Carrier frequency (kHz)")
    ax.set_ylabel("Mechanical Index")
    ax.set_title("Clinical-study guidance space: target, frequency, and MI")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    savefig("fig05_clinical_guidance_space")
    plt.close(fig)


def plot_guidance_map(protocol: Protocol) -> dict[str, float]:
    grid = guidance_map()
    p, d = np.meshgrid(grid["pressure_kpa"], grid["duty_percent"], indexing="ij")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    image = ax.contourf(p, d, grid["channel_drive"], levels=24, cmap="viridis")
    ax.contour(p, d, grid["feasible"].astype(float), levels=[0.5], colors="white", linewidths=1.8)
    ax.contour(p, d, grid["ispta_w_cm2"], levels=[0.72], colors="red", linestyles="--", linewidths=1.2)
    ax.contour(p, d, grid["delta_t_c"], levels=[2.0], colors="orange", linestyles="--", linewidths=1.2)
    ax.set_xlabel("Peak pressure (kPa)")
    ax.set_ylabel("Duty cycle (%)")
    ax.set_title("Closed-loop planning map: maximize channel drive inside safety constraints")
    fig.colorbar(image, ax=ax, label="Mechanochemical drive")
    fig.tight_layout()
    savefig("fig06_guidance_map")
    plt.close(fig)
    objective = grid["objective"]
    best_index = np.unravel_index(np.nanargmax(objective), objective.shape)
    return {
        "best_pressure_kpa": float(grid["pressure_kpa"][best_index[0]]),
        "best_duty_percent": float(grid["duty_percent"][best_index[1]]),
        "best_channel_drive": float(objective[best_index]),
        "feasible_fraction": float(np.count_nonzero(grid["feasible"]) / grid["feasible"].size),
    }


def run() -> dict[str, object]:
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
    })
    protocol = Protocol()
    print("[ch26] Simulating acoustic, thermal, mechanochemical, and neural response models")
    metrics: dict[str, object] = {"protocol": asdict(protocol)}
    metrics["acoustic"] = plot_acoustic_field(protocol)
    metrics["response"] = plot_response(protocol)
    plot_channels()
    metrics["thermal"] = plot_safety(protocol)
    plot_clinical_guidance()
    metrics["guidance"] = plot_guidance_map(protocol)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print("[ch26] Complete")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__" or __name__ == "ch26":
    run()
