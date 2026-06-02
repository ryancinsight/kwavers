"""
HIFU procedure simulation with acoustic focus and Pennes temperature rise.

Model
-----
Acoustic field:
    A focused circular aperture is evaluated by the Rayleigh-Sommerfeld
    velocity-potential surface integral. The aperture phase is pre-delayed so
    rays from every aperture point arrive in phase at the geometric focus.

Heating:
    The absorbed acoustic power density is

        Q(x, z) = 2 alpha I(x, z),

    where alpha is the amplitude absorption coefficient [Np/m] and I is the
    temporal-average acoustic intensity [W/m^2].

Thermal model:
    Temperature evolves by the Pennes bioheat equation,

        rho c dT/dt = k nabla^2 T - w_b rho_b c_b (T - T_a) + Q + Q_m.

    The explicit finite-difference step enforces the two-dimensional diffusion
    stability bound before the procedure is advanced.

References
----------
- Pennes, H. H. (1948). Analysis of tissue and arterial blood temperatures in
  the resting human forearm. Journal of Applied Physiology, 1(2), 93-122.
- ter Haar, G. and Coussios, C. (2007). High intensity focused ultrasound:
  physical principles and devices. International Journal of Hyperthermia.
- Treeby, B. E. and Cox, B. T. (2010). k-Wave: MATLAB toolbox for the
  simulation and reconstruction of photoacoustic wave fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from example_parity_utils import DEFAULT_OUTPUT_DIR, save_text_report


FIELD_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "hifu_procedure_focal_field.png"
TEMPERATURE_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "hifu_procedure_temperature.png"
CAVITATION_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "hifu_procedure_cavitation.png"
CAVITATION_FEEDBACK_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "hifu_procedure_cavitation_feedback.png"
REPORT_PATH = DEFAULT_OUTPUT_DIR / "hifu_procedure_metrics.txt"


@dataclass(frozen=True)
class AcousticConfig:
    frequency_hz: float = 1.0e6
    sound_speed_m_s: float = 1540.0
    density_kg_m3: float = 1050.0
    aperture_radius_m: float = 15.0e-3
    focal_length_m: float = 35.0e-3
    target_peak_intensity_w_m2: float = 8.0e5
    absorption_np_m: float = 8.0
    aperture_radial_samples: int = 36
    aperture_angular_samples: int = 96


@dataclass(frozen=True)
class ThermalConfig:
    baseline_temperature_c: float = 37.0
    arterial_temperature_c: float = 37.0
    conductivity_w_m_k: float = 0.52
    density_kg_m3: float = 1050.0
    specific_heat_j_kg_k: float = 3600.0
    blood_density_kg_m3: float = 1060.0
    blood_specific_heat_j_kg_k: float = 3860.0
    blood_perfusion_s_inv: float = 0.004
    metabolic_heat_w_m3: float = 420.0
    sonication_s: float = 20.0
    cooling_s: float = 20.0
    dt_s: float = 0.05


@dataclass(frozen=True)
class BubbleConfig:
    equilibrium_radius_m: float = 1.0e-6
    ambient_pressure_pa: float = 101_325.0
    vapor_pressure_pa: float = 2_330.0
    surface_tension_n_m: float = 0.0728
    viscosity_pa_s: float = 1.0e-3
    polytropic_exponent: float = 1.4
    dt_s: float = 2.0e-9
    cycles_per_window: int = 16
    control_windows: int = 80
    receiver_distances_m: tuple[float, ...] = (18.0e-3, 24.0e-3)
    target_inertial_radius_ratio: float = 1.8
    target_subharmonic_ratio: float = 0.18
    target_receiver_rms_pa: float = 15.0
    controller_gain: float = 0.15
    initial_pressure_fraction: float = 0.05
    min_pressure_fraction: float = 0.04
    max_pressure_fraction: float = 0.10
    nominal_pressure_fraction: float = 0.10
    feedback_period_s: float = 2.5


@dataclass(frozen=True)
class GridConfig:
    x_extent_m: float = 24.0e-3
    z_min_m: float = 5.0e-3
    z_max_m: float = 60.0e-3
    dx_m: float = 0.5e-3
    dz_m: float = 0.5e-3


def build_grid(config: GridConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(-config.x_extent_m, config.x_extent_m + 0.5 * config.dx_m, config.dx_m)
    z = np.arange(config.z_min_m, config.z_max_m + 0.5 * config.dz_m, config.dz_m)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    return x, z, xx, zz


def focused_aperture_intensity(
    x_field: np.ndarray,
    z_field: np.ndarray,
    config: AcousticConfig,
) -> np.ndarray:
    """Compute normalized HIFU intensity from a focused circular aperture.

    The Rayleigh-Sommerfeld aperture integral is discretized in polar
    coordinates. The annular quadrature uses midpoint radii so every cell has
    non-zero area and no singular source sample lies on the aperture centre.
    """
    k = 2.0 * np.pi * config.frequency_hz / config.sound_speed_m_s
    dr = config.aperture_radius_m / config.aperture_radial_samples
    dtheta = 2.0 * np.pi / config.aperture_angular_samples
    field = np.zeros_like(x_field, dtype=np.complex128)

    for radial_idx in range(config.aperture_radial_samples):
        r_ap = (radial_idx + 0.5) * dr
        area_weight = r_ap * dr * dtheta
        for angular_idx in range(config.aperture_angular_samples):
            theta = (angular_idx + 0.5) * dtheta
            xa = r_ap * np.cos(theta)
            ya = r_ap * np.sin(theta)
            distance_to_focus = np.sqrt(xa * xa + ya * ya + config.focal_length_m**2)
            focus_phase = -k * (distance_to_focus - config.focal_length_m)
            propagation_distance = np.sqrt((x_field - xa) ** 2 + ya * ya + z_field**2)
            field += area_weight * np.exp(1j * (k * propagation_distance + focus_phase)) / propagation_distance

    pressure_magnitude = np.abs(field)
    intensity = pressure_magnitude**2 / (2.0 * config.density_kg_m3 * config.sound_speed_m_s)
    peak = float(np.max(intensity))
    if peak <= 0.0 or not np.isfinite(peak):
        raise ValueError("focused aperture produced a non-positive or non-finite intensity field")
    return intensity * (config.target_peak_intensity_w_m2 / peak)


def focal_metrics(intensity: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, float]:
    peak_index = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    peak = float(intensity[peak_index])
    half_max = 0.5 * peak
    focus_x = float(x[peak_index[0]])
    focus_z = float(z[peak_index[1]])

    lateral = intensity[:, peak_index[1]]
    axial = intensity[peak_index[0], :]
    lateral_support = np.flatnonzero(lateral >= half_max)
    axial_support = np.flatnonzero(axial >= half_max)
    lateral_fwhm = float(x[lateral_support[-1]] - x[lateral_support[0]]) if lateral_support.size else 0.0
    axial_fwhm = float(z[axial_support[-1]] - z[axial_support[0]]) if axial_support.size else 0.0

    return {
        "peak_intensity_w_m2": peak,
        "focus_x_m": focus_x,
        "focus_z_m": focus_z,
        "lateral_fwhm_m": lateral_fwhm,
        "axial_fwhm_m": axial_fwhm,
    }


def cavitation_metrics(
    intensity: np.ndarray,
    acoustic: AcousticConfig,
    grid_config: GridConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return pressure amplitude, mechanical index, and cavitation-risk metrics.

    The pressure amplitude follows the plane-wave intensity relation

        I = p_rms^2 / (rho c), p_peak = sqrt(2 rho c I).

    Mechanical index is the standard cavitation-risk proxy

        MI = p_neg,peak[MPa] / sqrt(f[MHz]).

    This is not a bubble-dynamics solver; it is a conservative plot of where
    the acoustic field exceeds a pressure-index threshold associated with
    cavitation risk screening.
    """
    pressure_peak_pa = np.sqrt(2.0 * acoustic.density_kg_m3 * acoustic.sound_speed_m_s * intensity)
    pressure_peak_mpa = pressure_peak_pa / 1.0e6
    frequency_mhz = acoustic.frequency_hz / 1.0e6
    mechanical_index = pressure_peak_mpa / np.sqrt(frequency_mhz)
    mi_threshold = 1.9
    cell_area_mm2 = grid_config.dx_m * grid_config.dz_m * 1.0e6
    threshold_area_mm2 = float(np.count_nonzero(mechanical_index >= mi_threshold) * cell_area_mm2)
    metrics = {
        "peak_pressure_mpa": float(np.max(pressure_peak_mpa)),
        "peak_mechanical_index": float(np.max(mechanical_index)),
        "mi_threshold": mi_threshold,
        "mi_threshold_area_mm2": threshold_area_mm2,
    }
    return pressure_peak_mpa, mechanical_index, metrics


def _bubble_acceleration(
    radius: float,
    wall_velocity: float,
    acoustic_pressure: float,
    acoustic_pressure_dt: float,
    pressure_scale: float,
    acoustic: AcousticConfig,
    bubble: BubbleConfig,
) -> float:
    """Keller-Miksis bubble-wall acceleration for a free gas bubble."""
    radius = max(radius, 0.05 * bubble.equilibrium_radius_m)
    mach = np.clip(wall_velocity / acoustic.sound_speed_m_s, -0.95, 0.95)
    gas_pressure = (
        bubble.ambient_pressure_pa
        + 2.0 * bubble.surface_tension_n_m / bubble.equilibrium_radius_m
        - bubble.vapor_pressure_pa
    ) * (bubble.equilibrium_radius_m / radius) ** (3.0 * bubble.polytropic_exponent) + bubble.vapor_pressure_pa
    wall_pressure = (
        gas_pressure
        - 2.0 * bubble.surface_tension_n_m / radius
        - 4.0 * bubble.viscosity_pa_s * wall_velocity / radius
    )
    far_pressure = bubble.ambient_pressure_pa + pressure_scale * acoustic_pressure
    net_pressure = wall_pressure - far_pressure
    net_pressure_dt = -pressure_scale * acoustic_pressure_dt
    numerator = (
        (1.0 + mach) * net_pressure / acoustic.density_kg_m3
        + radius * net_pressure_dt / (acoustic.density_kg_m3 * acoustic.sound_speed_m_s)
        - 1.5 * (1.0 - mach / 3.0) * wall_velocity * wall_velocity
    )
    denominator = (1.0 - mach) * radius
    return float(numerator / denominator)


def _rk4_bubble_step(
    radius: float,
    wall_velocity: float,
    time_s: float,
    pressure_scale: float,
    peak_pressure_pa: float,
    acoustic: AcousticConfig,
    bubble: BubbleConfig,
) -> tuple[float, float, float]:
    omega = 2.0 * np.pi * acoustic.frequency_hz

    def rhs(r: float, v: float, t: float) -> tuple[float, float]:
        acoustic_pressure = -peak_pressure_pa * np.sin(omega * t)
        acoustic_pressure_dt = -peak_pressure_pa * omega * np.cos(omega * t)
        return v, _bubble_acceleration(
            r,
            v,
            acoustic_pressure,
            acoustic_pressure_dt,
            pressure_scale,
            acoustic,
            bubble,
        )

    dt = bubble.dt_s
    k1_r, k1_v = rhs(radius, wall_velocity, time_s)
    k2_r, k2_v = rhs(radius + 0.5 * dt * k1_r, wall_velocity + 0.5 * dt * k1_v, time_s + 0.5 * dt)
    k3_r, k3_v = rhs(radius + 0.5 * dt * k2_r, wall_velocity + 0.5 * dt * k2_v, time_s + 0.5 * dt)
    k4_r, k4_v = rhs(radius + dt * k3_r, wall_velocity + dt * k3_v, time_s + dt)
    next_radius = radius + dt * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r) / 6.0
    next_velocity = wall_velocity + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    next_radius = max(next_radius, 0.05 * bubble.equilibrium_radius_m)
    acceleration = _bubble_acceleration(
        next_radius,
        next_velocity,
        -peak_pressure_pa * np.sin(omega * (time_s + dt)),
        -peak_pressure_pa * omega * np.cos(omega * (time_s + dt)),
        pressure_scale,
        acoustic,
        bubble,
    )
    return next_radius, next_velocity, acceleration


def _subharmonic_ratio(signal: np.ndarray, sample_rate_hz: float, fundamental_hz: float) -> float:
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft((signal - np.mean(signal)) * window))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate_hz)
    fundamental_bin = int(np.argmin(np.abs(frequencies - fundamental_hz)))
    subharmonic_bin = int(np.argmin(np.abs(frequencies - 0.5 * fundamental_hz)))
    fundamental = float(spectrum[fundamental_bin])
    if fundamental <= 1.0e-30:
        return 0.0
    return float(spectrum[subharmonic_bin] / fundamental)


def simulate_cavitation_feedback(
    acoustic: AcousticConfig,
    bubble: BubbleConfig,
    peak_pressure_pa: float,
) -> dict[str, np.ndarray | float]:
    """Simulate receiver-feedback control from Keller-Miksis bubble emissions."""
    steps_per_window = int(round(bubble.cycles_per_window / (acoustic.frequency_hz * bubble.dt_s)))
    if steps_per_window < 16:
        raise ValueError("bubble control window has too few samples for spectral detection")
    total_steps = steps_per_window * bubble.control_windows
    sample_rate = 1.0 / bubble.dt_s

    times = np.empty(total_steps, dtype=np.float64)
    radius = np.empty(total_steps, dtype=np.float64)
    receiver_signal = np.empty(total_steps, dtype=np.float64)
    pressure_fraction = np.empty(total_steps, dtype=np.float64)
    window_times = np.empty(bubble.control_windows, dtype=np.float64)
    subharmonic = np.empty(bubble.control_windows, dtype=np.float64)
    radius_activity = np.empty(bubble.control_windows, dtype=np.float64)
    receiver_activity = np.empty(bubble.control_windows, dtype=np.float64)
    controller_output = np.empty(bubble.control_windows, dtype=np.float64)

    r = bubble.equilibrium_radius_m
    v = 0.0
    scale = float(np.clip(bubble.initial_pressure_fraction, bubble.min_pressure_fraction, bubble.max_pressure_fraction))
    receiver_distance = float(np.mean(bubble.receiver_distances_m))

    for window_index in range(bubble.control_windows):
        start = window_index * steps_per_window
        stop = start + steps_per_window
        for step in range(start, stop):
            t = step * bubble.dt_s
            r, v, accel = _rk4_bubble_step(r, v, t, scale, peak_pressure_pa, acoustic, bubble)
            if not (np.isfinite(r) and np.isfinite(v) and np.isfinite(accel)):
                raise FloatingPointError(
                    "Keller-Miksis integration produced a non-finite state; reduce dt or pressure bounds"
                )
            times[step] = t
            radius[step] = r
            pressure_fraction[step] = scale
            receiver_signal[step] = acoustic.density_kg_m3 * (r * r * accel + 2.0 * r * v * v) / receiver_distance

        window_signal = receiver_signal[start:stop]
        window_radius_ratio = float(np.max(radius[start:stop]) / bubble.equilibrium_radius_m)
        ratio = _subharmonic_ratio(window_signal, sample_rate, acoustic.frequency_hz)
        rms_activity = float(np.sqrt(np.mean(window_signal * window_signal)) / bubble.target_receiver_rms_pa)
        radius_onset_activity = window_radius_ratio / max(bubble.target_inertial_radius_ratio, 1.0e-30)
        activity = max(ratio / max(bubble.target_subharmonic_ratio, 1.0e-30), rms_activity, radius_onset_activity)
        error = activity - 1.0
        scale = float(
            np.clip(
                scale * (1.0 - bubble.controller_gain * error),
                bubble.min_pressure_fraction,
                bubble.max_pressure_fraction,
            )
        )
        window_times[window_index] = times[stop - 1]
        subharmonic[window_index] = ratio
        radius_activity[window_index] = radius_onset_activity
        receiver_activity[window_index] = activity
        controller_output[window_index] = scale

    return {
        "time_s": times,
        "radius_m": radius,
        "receiver_pressure_pa": receiver_signal,
        "pressure_fraction": pressure_fraction,
        "window_time_s": window_times,
        "subharmonic_ratio": subharmonic,
        "radius_activity": radius_activity,
        "receiver_activity": receiver_activity,
        "controller_output": controller_output,
        "max_radius_ratio": float(np.max(radius) / bubble.equilibrium_radius_m),
        "max_receiver_pressure_pa": float(np.max(np.abs(receiver_signal))),
        "final_pressure_fraction": float(controller_output[-1]),
        "mean_subharmonic_ratio": float(np.mean(subharmonic[-max(1, bubble.control_windows // 4) :])),
        "mean_terminal_receiver_activity": float(np.mean(receiver_activity[-max(1, bubble.control_windows // 4) :])),
    }


def feedback_power_envelope(
    thermal_times_s: np.ndarray,
    thermal: ThermalConfig,
    feedback: dict[str, np.ndarray | float],
    reference_pressure_fraction: float,
    feedback_period_s: float,
) -> np.ndarray:
    """Return the Pennes heat-source scale imposed by pressure feedback.

    The acoustic intensity relation gives

        I(t) / I_ref = (p(t) / p_ref)^2.

    `reference_pressure_fraction` is the planned cavitation-onset treatment
    pressure. The thermal source term therefore represents nominal treatment
    power, and receiver feedback only trims around that operating point.

    The Keller-Miksis controller operates on acoustic-cycle receiver windows
    while the thermal solver advances on tissue time scales. The envelope
    repeats the measured receiver-control burst over the requested feedback
    period, modelling sample-and-hold cavitation monitoring throughout the
    sonication instead of holding the first terminal controller value.
    """
    if reference_pressure_fraction <= 0.0 or not np.isfinite(reference_pressure_fraction):
        raise ValueError("reference_pressure_fraction must be positive and finite")
    if feedback_period_s <= 0.0 or not np.isfinite(feedback_period_s):
        raise ValueError("feedback_period_s must be positive and finite")

    window_times = np.asarray(feedback["window_time_s"], dtype=np.float64)
    controller = np.asarray(feedback["controller_output"], dtype=np.float64)
    if window_times.ndim != 1 or controller.ndim != 1 or window_times.size != controller.size or window_times.size == 0:
        raise ValueError("feedback controller output must be a non-empty 1-D time series")
    if np.any(~np.isfinite(window_times)) or np.any(~np.isfinite(controller)):
        raise ValueError("feedback controller output must be finite")
    if np.any(np.diff(window_times) < 0.0):
        raise ValueError("feedback controller times must be monotone")

    burst_duration_s = float(window_times[-1])
    if burst_duration_s <= 0.0:
        raise ValueError("feedback controller times must span a positive burst duration")

    controller_with_initial = np.concatenate(([controller[0]], controller))
    phase_with_initial = np.concatenate(([0.0], window_times / burst_duration_s))
    active = thermal_times_s <= thermal.sonication_s
    cycle_phase = np.zeros_like(thermal_times_s, dtype=np.float64)
    cycle_phase[active] = np.mod(thermal_times_s[active], feedback_period_s) / feedback_period_s
    pressure_fraction = np.interp(
        cycle_phase,
        phase_with_initial,
        controller_with_initial,
        left=controller_with_initial[0],
        right=controller_with_initial[-1],
    )
    pressure_fraction = np.where(active, pressure_fraction, 0.0)
    return (pressure_fraction / reference_pressure_fraction) ** 2


def pennes_temperature(
    heat_source: np.ndarray,
    grid_config: GridConfig,
    thermal: ThermalConfig,
    power_envelope: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance Pennes bioheat and return time, focus temperature, max temperature."""
    alpha_th = thermal.conductivity_w_m_k / (thermal.density_kg_m3 * thermal.specific_heat_j_kg_k)
    stability = alpha_th * thermal.dt_s * (
        1.0 / grid_config.dx_m**2 + 1.0 / grid_config.dz_m**2
    )
    if stability > 0.5:
        raise ValueError(f"unstable explicit Pennes step: stability={stability:.6f} > 0.5")

    perfusion_rate = (
        thermal.blood_perfusion_s_inv
        * thermal.blood_density_kg_m3
        * thermal.blood_specific_heat_j_kg_k
        / (thermal.density_kg_m3 * thermal.specific_heat_j_kg_k)
    )
    source_scale = 1.0 / (thermal.density_kg_m3 * thermal.specific_heat_j_kg_k)
    total_s = thermal.sonication_s + thermal.cooling_s
    n_steps = int(round(total_s / thermal.dt_s))
    times = np.linspace(0.0, n_steps * thermal.dt_s, n_steps + 1)
    if power_envelope is None:
        source_envelope = np.where(times <= thermal.sonication_s, 1.0, 0.0)
    else:
        source_envelope = np.asarray(power_envelope, dtype=np.float64)
        if source_envelope.shape != times.shape:
            raise ValueError("power_envelope must have one value per thermal time sample")
        if np.any(~np.isfinite(source_envelope)) or np.any(source_envelope < 0.0):
            raise ValueError("power_envelope values must be finite and non-negative")
        source_envelope = np.where(times <= thermal.sonication_s, source_envelope, 0.0)

    temperature = np.full_like(heat_source, thermal.baseline_temperature_c, dtype=np.float64)
    peak_index = np.unravel_index(int(np.argmax(heat_source)), heat_source.shape)
    focus_temperature = np.empty(n_steps + 1, dtype=np.float64)
    max_temperature = np.empty(n_steps + 1, dtype=np.float64)
    focus_temperature[0] = temperature[peak_index]
    max_temperature[0] = float(np.max(temperature))

    for step in range(1, n_steps + 1):
        previous = temperature.copy()
        laplacian = (
            (previous[2:, 1:-1] - 2.0 * previous[1:-1, 1:-1] + previous[:-2, 1:-1])
            / grid_config.dx_m**2
            + (previous[1:-1, 2:] - 2.0 * previous[1:-1, 1:-1] + previous[1:-1, :-2])
            / grid_config.dz_m**2
        )
        active_heat = heat_source[1:-1, 1:-1] * source_envelope[step]
        d_t_dt = (
            alpha_th * laplacian
            - perfusion_rate * (previous[1:-1, 1:-1] - thermal.arterial_temperature_c)
            + thermal.metabolic_heat_w_m3 * source_scale
            + active_heat * source_scale
        )
        temperature[1:-1, 1:-1] = previous[1:-1, 1:-1] + thermal.dt_s * d_t_dt
        temperature[0, :] = temperature[1, :]
        temperature[-1, :] = temperature[-2, :]
        temperature[:, 0] = temperature[:, 1]
        temperature[:, -1] = temperature[:, -2]
        focus_temperature[step] = temperature[peak_index]
        max_temperature[step] = float(np.max(temperature))

    return times, focus_temperature, max_temperature, temperature


def save_figures(
    x: np.ndarray,
    z: np.ndarray,
    intensity: np.ndarray,
    heat_source: np.ndarray,
    temperature: np.ndarray,
    times: np.ndarray,
    focus_temperature: np.ndarray,
    max_temperature: np.ndarray,
    metrics: dict[str, float],
    unmodulated_focus_temperature: np.ndarray | None = None,
    power_envelope: np.ndarray | None = None,
    sonication_s: float = 20.0,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_mm = x * 1e3
    z_mm = z * 1e3
    extent = [z_mm[0], z_mm[-1], x_mm[0], x_mm[-1]]
    focus_z_mm = metrics["focus_z_m"] * 1e3
    focus_x_mm = metrics["focus_x_m"] * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)
    im0 = axes[0].imshow(intensity / 1.0e4, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    axes[0].plot(focus_z_mm, focus_x_mm, "c+", markersize=10, markeredgewidth=1.8)
    axes[0].set_title("Intensity [W/cm^2]")
    axes[0].set_xlabel("axial z [mm]")
    axes[0].set_ylabel("lateral x [mm]")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(heat_source / 1.0e6, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[1].plot(focus_z_mm, focus_x_mm, "c+", markersize=10, markeredgewidth=1.8)
    axes[1].set_title("Heat source [MW/m^3]")
    axes[1].set_xlabel("axial z [mm]")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(temperature, origin="lower", aspect="auto", extent=extent, cmap="hot")
    axes[2].plot(focus_z_mm, focus_x_mm, "c+", markersize=10, markeredgewidth=1.8)
    axes[2].set_title("Temperature after procedure [deg C]")
    axes[2].set_xlabel("axial z [mm]")
    fig.colorbar(im2, ax=axes[2])
    fig.suptitle("HIFU focal spot and absorbed heating")
    fig.savefig(FIELD_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, (ax, ax_rate) = plt.subplots(2, 1, figsize=(8.0, 6.6), sharex=True, constrained_layout=True)
    ax.plot(times, focus_temperature, label="feedback-controlled focal voxel")
    ax.plot(times, max_temperature, label="feedback-controlled spatial maximum", linestyle="--")
    if unmodulated_focus_temperature is not None:
        ax.plot(times, unmodulated_focus_temperature, label="constant-power focal voxel", color="0.35", linestyle=":")
    ax.axvline(sonication_s, color="0.4", linewidth=1.0, label="sonication off")
    ax.set_ylabel("temperature [deg C]")
    ax.set_title("HIFU temperature rise over time")
    ax.grid(True, alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    if power_envelope is not None:
        ax_power = ax.twinx()
        power_line = ax_power.step(times, power_envelope, where="post", color="tab:green", alpha=0.45, label="relative acoustic power")[0]
        ax_power.set_ylabel("relative acoustic power")
        ax_power.set_ylim(0.0, max(1.1, 1.05 * float(np.max(power_envelope))))
        lines.append(power_line)
        labels.append("relative acoustic power")
    ax.legend(lines, labels)
    heating_rate = np.gradient(focus_temperature, times)
    ax_rate.plot(times, heating_rate, color="tab:red", label="focal dT/dt")
    ax_rate.axhline(0.0, color="0.4", linewidth=0.8)
    ax_rate.axvline(sonication_s, color="0.4", linewidth=1.0)
    ax_rate.set_xlabel("time [s]")
    ax_rate.set_ylabel("dT/dt [deg C/s]")
    ax_rate.set_title("Temperature-rate response to power modulation")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.legend()
    fig.savefig(TEMPERATURE_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return FIELD_FIGURE_PATH, TEMPERATURE_FIGURE_PATH


def save_cavitation_figure(
    x: np.ndarray,
    z: np.ndarray,
    pressure_peak_mpa: np.ndarray,
    mechanical_index: np.ndarray,
    cavitation: dict[str, float],
    metrics: dict[str, float],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_mm = x * 1e3
    z_mm = z * 1e3
    extent = [z_mm[0], z_mm[-1], x_mm[0], x_mm[-1]]
    focus_z_mm = metrics["focus_z_m"] * 1e3
    focus_x_mm = metrics["focus_x_m"] * 1e3
    threshold = cavitation["mi_threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(pressure_peak_mpa, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[0].plot(focus_z_mm, focus_x_mm, "r+", markersize=10, markeredgewidth=1.8)
    axes[0].set_title("Peak rarefactional pressure proxy [MPa]")
    axes[0].set_xlabel("axial z [mm]")
    axes[0].set_ylabel("lateral x [mm]")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mechanical_index, origin="lower", aspect="auto", extent=extent, cmap="plasma")
    if float(np.max(mechanical_index)) >= threshold:
        axes[1].contour(z_mm, x_mm, mechanical_index, levels=[threshold], colors=["cyan"], linewidths=1.4)
    axes[1].plot(focus_z_mm, focus_x_mm, "c+", markersize=10, markeredgewidth=1.8)
    axes[1].set_title(f"Mechanical index, contour MI={threshold:.1f}")
    axes[1].set_xlabel("axial z [mm]")
    fig.colorbar(im1, ax=axes[1])
    fig.suptitle("HIFU cavitation-risk proxy from pressure field")
    fig.savefig(CAVITATION_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return CAVITATION_FIGURE_PATH


def save_cavitation_feedback_figure(feedback: dict[str, np.ndarray | float], bubble: BubbleConfig) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_ms = np.asarray(feedback["time_s"], dtype=float) * 1.0e3
    window_ms = np.asarray(feedback["window_time_s"], dtype=float) * 1.0e3
    radius_um = np.asarray(feedback["radius_m"], dtype=float) * 1.0e6
    receiver_kpa = np.asarray(feedback["receiver_pressure_pa"], dtype=float) / 1.0e3
    pressure_fraction = np.asarray(feedback["pressure_fraction"], dtype=float)
    subharmonic = np.asarray(feedback["subharmonic_ratio"], dtype=float)
    radius_activity = np.asarray(feedback["radius_activity"], dtype=float)
    receiver_activity = np.asarray(feedback["receiver_activity"], dtype=float)
    controller = np.asarray(feedback["controller_output"], dtype=float)

    fig, axes = plt.subplots(5, 1, figsize=(10.5, 10.8), sharex=False, constrained_layout=True)
    axes[0].plot(time_ms, radius_um)
    axes[0].axhline(bubble.equilibrium_radius_m * 1.0e6, color="0.4", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("radius [um]")
    axes[0].set_title("Keller-Miksis bubble response")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_ms, receiver_kpa)
    axes[1].set_ylabel("receiver [kPa]")
    axes[1].set_title("Passive receiver pressure from bubble volume acceleration")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(window_ms, subharmonic, marker="o", markersize=2.5)
    axes[2].axhline(bubble.target_subharmonic_ratio, color="r", linewidth=1.0, linestyle="--")
    axes[2].set_ylabel("subharmonic/fund.")
    axes[2].set_title("Subharmonic detector output")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(window_ms, receiver_activity, marker="o", markersize=2.5, label="controller activity")
    axes[3].plot(window_ms, radius_activity, marker=".", markersize=3.0, label="Rmax/R0 onset")
    axes[3].axhline(1.0, color="r", linewidth=1.0, linestyle="--")
    axes[3].set_ylabel("activity/target")
    axes[3].set_title("Receiver cavitation activity used for feedback")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(time_ms, pressure_fraction, label="applied pressure fraction")
    axes[4].step(window_ms, controller, where="post", label="controller update", alpha=0.8)
    axes[4].set_xlabel("time [ms]")
    axes[4].set_ylabel("pressure fraction")
    axes[4].set_title("Closed-loop pressure modulation")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    fig.savefig(CAVITATION_FEEDBACK_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return CAVITATION_FEEDBACK_FIGURE_PATH


def main() -> int:
    acoustic = AcousticConfig()
    thermal = ThermalConfig()
    bubble = BubbleConfig()
    grid_config = GridConfig()
    x, z, xx, zz = build_grid(grid_config)
    intensity = focused_aperture_intensity(xx, zz, acoustic)
    heat_source = 2.0 * acoustic.absorption_np_m * intensity
    metrics = focal_metrics(intensity, x, z)
    pressure_peak_mpa, mechanical_index, cavitation = cavitation_metrics(intensity, acoustic, grid_config)
    feedback = simulate_cavitation_feedback(acoustic, bubble, cavitation["peak_pressure_mpa"] * 1.0e6)
    total_s = thermal.sonication_s + thermal.cooling_s
    n_thermal_steps = int(round(total_s / thermal.dt_s))
    thermal_times = np.linspace(0.0, n_thermal_steps * thermal.dt_s, n_thermal_steps + 1)
    power_envelope = feedback_power_envelope(
        thermal_times,
        thermal,
        feedback,
        bubble.nominal_pressure_fraction,
        bubble.feedback_period_s,
    )
    times, focus_temperature, max_temperature, final_temperature = pennes_temperature(
        heat_source,
        grid_config,
        thermal,
        power_envelope,
    )
    _, unmodulated_focus_temperature, _, _ = pennes_temperature(
        heat_source,
        grid_config,
        thermal,
    )
    field_figure, temperature_figure = save_figures(
        x,
        z,
        intensity,
        heat_source,
        final_temperature,
        times,
        focus_temperature,
        max_temperature,
        metrics,
        unmodulated_focus_temperature,
        power_envelope,
        thermal.sonication_s,
    )
    cavitation_figure = save_cavitation_figure(
        x,
        z,
        pressure_peak_mpa,
        mechanical_index,
        cavitation,
        metrics,
    )
    cavitation_feedback_figure = save_cavitation_feedback_figure(feedback, bubble)

    peak_temp = float(np.max(final_temperature))
    focus_final = float(focus_temperature[-1])
    focus_peak = float(np.max(focus_temperature))
    unmodulated_focus_peak = float(np.max(unmodulated_focus_temperature))
    if not (0.0 < metrics["focus_z_m"] < grid_config.z_max_m):
        raise AssertionError("focus must lie inside the simulated axial domain")
    if focus_peak <= thermal.baseline_temperature_c:
        raise AssertionError("HIFU sonication must raise focal temperature")
    if peak_temp < focus_final:
        raise AssertionError("spatial peak temperature cannot be below focal temperature")

    save_text_report(
        REPORT_PATH,
        "hifu_procedure_simulation metrics",
        [
            "procedure_status: PASS",
            f"frequency_hz: {acoustic.frequency_hz:.6e}",
            f"aperture_radius_m: {acoustic.aperture_radius_m:.6e}",
            f"geometric_focal_length_m: {acoustic.focal_length_m:.6e}",
            f"target_peak_intensity_w_m2: {acoustic.target_peak_intensity_w_m2:.6e}",
            f"absorption_np_m: {acoustic.absorption_np_m:.6e}",
            f"computed_focus_x_m: {metrics['focus_x_m']:.6e}",
            f"computed_focus_z_m: {metrics['focus_z_m']:.6e}",
            f"lateral_fwhm_m: {metrics['lateral_fwhm_m']:.6e}",
            f"axial_fwhm_m: {metrics['axial_fwhm_m']:.6e}",
            f"peak_heat_source_w_m3: {float(np.max(heat_source)):.6e}",
            f"peak_pressure_mpa: {cavitation['peak_pressure_mpa']:.6f}",
            f"peak_mechanical_index: {cavitation['peak_mechanical_index']:.6f}",
            f"mechanical_index_threshold: {cavitation['mi_threshold']:.6f}",
            f"mechanical_index_threshold_area_mm2: {cavitation['mi_threshold_area_mm2']:.6f}",
            f"bubble_equilibrium_radius_m: {bubble.equilibrium_radius_m:.6e}",
            f"bubble_control_windows: {bubble.control_windows}",
            f"bubble_cycles_per_window: {bubble.cycles_per_window}",
            f"receiver_mean_distance_m: {float(np.mean(bubble.receiver_distances_m)):.6e}",
            f"feedback_target_inertial_radius_ratio: {bubble.target_inertial_radius_ratio:.6f}",
            f"feedback_target_subharmonic_ratio: {bubble.target_subharmonic_ratio:.6f}",
            f"feedback_target_receiver_rms_pa: {bubble.target_receiver_rms_pa:.6e}",
            f"feedback_initial_pressure_fraction: {bubble.initial_pressure_fraction:.6f}",
            f"feedback_nominal_pressure_fraction: {bubble.nominal_pressure_fraction:.6f}",
            f"feedback_period_s: {bubble.feedback_period_s:.6e}",
            f"feedback_mean_terminal_subharmonic_ratio: {feedback['mean_subharmonic_ratio']:.6f}",
            f"feedback_mean_terminal_receiver_activity: {feedback['mean_terminal_receiver_activity']:.6f}",
            f"feedback_final_pressure_fraction: {feedback['final_pressure_fraction']:.6f}",
            f"feedback_max_radius_ratio: {feedback['max_radius_ratio']:.6f}",
            f"feedback_max_receiver_pressure_pa: {feedback['max_receiver_pressure_pa']:.6e}",
            f"feedback_terminal_power_scale: {power_envelope[int(round(thermal.sonication_s / thermal.dt_s))]:.6e}",
            f"feedback_mean_sonication_power_scale: {float(np.mean(power_envelope[times <= thermal.sonication_s])):.6e}",
            f"feedback_min_sonication_power_scale: {float(np.min(power_envelope[times <= thermal.sonication_s])):.6e}",
            f"feedback_max_sonication_power_scale: {float(np.max(power_envelope[times <= thermal.sonication_s])):.6e}",
            f"feedback_std_sonication_power_scale: {float(np.std(power_envelope[times <= thermal.sonication_s])):.6e}",
            f"unmodulated_focus_peak_temperature_c: {unmodulated_focus_peak:.6f}",
            f"feedback_temperature_reduction_c: {unmodulated_focus_peak - focus_peak:.6f}",
            f"sonication_s: {thermal.sonication_s:.6e}",
            f"cooling_s: {thermal.cooling_s:.6e}",
            f"thermal_dt_s: {thermal.dt_s:.6e}",
            f"focus_peak_temperature_c: {focus_peak:.6f}",
            f"focus_final_temperature_c: {focus_final:.6f}",
            f"spatial_peak_final_temperature_c: {peak_temp:.6f}",
            f"figure_field: {field_figure.name}",
            f"figure_temperature: {temperature_figure.name}",
            f"figure_cavitation: {cavitation_figure.name}",
            f"figure_cavitation_feedback: {cavitation_feedback_figure.name}",
        ],
    )
    print(f"HIFU procedure simulation PASS")
    print(f"  focus: x={metrics['focus_x_m']*1e3:.2f} mm, z={metrics['focus_z_m']*1e3:.2f} mm")
    print(f"  FWHM: lateral={metrics['lateral_fwhm_m']*1e3:.2f} mm, axial={metrics['axial_fwhm_m']*1e3:.2f} mm")
    print(f"  peak focal temperature: {focus_peak:.2f} deg C")
    print(f"  peak pressure proxy: {cavitation['peak_pressure_mpa']:.2f} MPa, MI={cavitation['peak_mechanical_index']:.2f}")
    print(
        "  cavitation feedback: "
        f"max R/R0={feedback['max_radius_ratio']:.2f}, "
        f"terminal subharmonic={feedback['mean_subharmonic_ratio']:.3f}, "
        f"terminal activity={feedback['mean_terminal_receiver_activity']:.3f}, "
        f"final pressure fraction={feedback['final_pressure_fraction']:.2f}"
    )
    print(f"  field figure: {field_figure}")
    print(f"  temperature figure: {temperature_figure}")
    print(f"  cavitation figure: {cavitation_figure}")
    print(f"  cavitation feedback figure: {cavitation_feedback_figure}")
    print(f"  report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
