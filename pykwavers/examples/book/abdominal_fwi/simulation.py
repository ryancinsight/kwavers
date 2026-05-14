"""Time-domain nonlinear forward models for abdominal FWI channels.

The module supplies source maps for Chapter 28. It does not implement the
receiver-side adjoint of a full nonlinear inversion. The contract is narrower:
run a bounded 2-D heterogeneous Westervelt FDTD burst on the CT slice, demodulate
the fundamental and second harmonic after target-arrival time, and drive a
lesion-local Rayleigh-Plesset bubble model from the simulated pressure to obtain
the subharmonic source density inverted by the reduced FWI operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abdominal_fwi.constants import C_REF_M_S


@dataclass(frozen=True)
class SimulationChannels:
    """Simulation-derived sources inverted by the reduced FWI operators."""

    fundamental_pressure_pa: np.ndarray
    second_harmonic_pressure_pa: np.ndarray
    subharmonic_source: np.ndarray
    nonlinear_source: np.ndarray
    metrics: dict[str, float | int | str]


def simulate_westervelt_channels(prepared, lesion_mask: np.ndarray, config) -> SimulationChannels:
    """Run a bounded 2-D Westervelt solve plus bubble subharmonic model."""

    pressure = westervelt_pressure_spectra(prepared, config)
    subharmonic = rayleigh_plesset_subharmonic(
        pressure.fundamental_pressure_pa,
        lesion_mask,
        prepared.ct_hu,
        config,
        pressure.dt_s,
        pressure.steps,
    )
    nonlinear = normalize_source(
        pressure.second_harmonic_pressure_pa * lesion_activity(lesion_mask)
    )
    active = np.asarray(lesion_mask, dtype=bool)
    lesion_fundamental_peak = (
        float(np.max(pressure.fundamental_pressure_pa[active])) if np.any(active) else 0.0
    )
    metrics = {
        "nonlinear_forward_model": "2d_westervelt_fdtd",
        "subharmonic_forward_model": "rayleigh_plesset_transient",
        "westervelt_steps": int(pressure.steps),
        "westervelt_dt_s": float(pressure.dt_s),
        "westervelt_frequency_hz": float(config.westervelt_frequency_hz),
        "westervelt_source_pressure_pa": float(config.westervelt_source_pressure_pa),
        "fundamental_peak_pa": float(np.max(pressure.fundamental_pressure_pa)),
        "second_harmonic_peak_pa": float(np.max(pressure.second_harmonic_pressure_pa)),
        "lesion_fundamental_peak_pa": lesion_fundamental_peak,
        "bubble_drive_peak_pa": float(config.bubble_drive_gain * lesion_fundamental_peak),
        "subharmonic_peak": float(np.max(subharmonic)),
    }
    return SimulationChannels(
        pressure.fundamental_pressure_pa,
        pressure.second_harmonic_pressure_pa,
        subharmonic,
        nonlinear,
        metrics,
    )


@dataclass(frozen=True)
class PressureSpectra:
    """Demodulated pressure spectra from the Westervelt time-domain solve."""

    fundamental_pressure_pa: np.ndarray
    second_harmonic_pressure_pa: np.ndarray
    dt_s: float
    steps: int


def westervelt_pressure_spectra(prepared, config) -> PressureSpectra:
    """Solve a 2-D heterogeneous Westervelt equation and demodulate harmonics."""

    c = np.clip(prepared.sound_speed_m_s.astype(np.float32), 900.0, 2600.0)
    rho = density_map(prepared.ct_hu)
    beta = nonlinearity_map(prepared.ct_hu)
    dx = float(prepared.spacing_m)
    dt = float(config.westervelt_cfl * dx / (np.sqrt(2.0) * float(np.max(c))))
    p_prev2 = np.zeros_like(c, dtype=np.float32)
    p_prev = np.zeros_like(c, dtype=np.float32)
    p_curr = np.zeros_like(c, dtype=np.float32)
    damping = damping_mask(c.shape, config.westervelt_pml_cells)
    source = focused_boundary_source(prepared, dt, config)
    steps = int(source.samples.shape[0])
    omega = 2.0 * np.pi * config.westervelt_frequency_hz
    start_accum = min(
        max(source.focus_arrival_step + source.burst_steps // 4, 0),
        max(steps - 1, 0),
    )
    count = 0
    acc_f = np.zeros_like(c, dtype=np.complex64)
    acc_2f = np.zeros_like(c, dtype=np.complex64)

    for step in range(steps):
        t = step * dt
        lap = laplacian(p_curr, dx)
        nonlinear = beta / (rho * c * c) * (
            p_curr * p_curr - 2.0 * p_prev * p_prev + p_prev2 * p_prev2
        )
        p_next = 2.0 * p_curr - p_prev + (dt * dt) * c * c * lap + nonlinear
        p_next[:, source.column] += source.samples[step]
        p_next *= damping
        p_prev2, p_prev, p_curr = p_prev, p_curr, p_next.astype(np.float32)

        if step >= start_accum:
            acc_f += p_curr * np.exp(-1j * omega * t)
            acc_2f += p_curr * np.exp(-2j * omega * t)
            count += 1

    scale = 2.0 / max(count, 1)
    return PressureSpectra(
        fundamental_pressure_pa=(np.abs(acc_f) * scale).astype(np.float32),
        second_harmonic_pressure_pa=(np.abs(acc_2f) * scale).astype(np.float32),
        dt_s=dt,
        steps=steps,
    )


@dataclass(frozen=True)
class BoundarySource:
    """Focused pressure drive imposed on one source boundary."""

    column: int
    samples: np.ndarray
    focus_arrival_step: int
    burst_steps: int


def focused_boundary_source(prepared, dt: float, config) -> BoundarySource:
    """Return a finite delayed burst focused at the segmented target centroid."""

    ny, nx = prepared.ct_hu.shape
    dx = float(prepared.spacing_m)
    column = min(max(config.westervelt_pml_cells + 1, 1), nx - 2)
    target_y, target_x = np.nonzero(prepared.target_mask)
    if target_y.size:
        focus_y = float(np.mean(target_y))
        focus_x = float(np.mean(target_x))
    else:
        focus_y = 0.5 * (ny - 1)
        focus_x = 0.5 * (nx - 1)
    y_m = (np.arange(ny, dtype=np.float32) - focus_y) * dx
    source_x_m = (column - focus_x) * dx
    distance = np.sqrt(y_m * y_m + source_x_m * source_x_m)
    max_distance = float(np.max(distance))
    delay = (max_distance - distance) / C_REF_M_S
    burst_steps = max(
        int(np.ceil(config.westervelt_cycles / (config.westervelt_frequency_hz * dt))),
        24,
    )
    focus_arrival_step = int(np.ceil(max_distance / (C_REF_M_S * dt)))
    ringdown_steps = max(int(np.ceil(2.0 / (config.westervelt_frequency_hz * dt))), 1)
    steps = focus_arrival_step + burst_steps + ringdown_steps
    samples = np.zeros((steps, ny), dtype=np.float32)
    aperture = np.exp(-0.5 * (y_m / max(0.5 * config.therapy_lateral_extent_m, dx)) ** 2)
    ramp_steps = max(int(2.0 / (config.westervelt_frequency_hz * dt)), 1)
    omega = 2.0 * np.pi * config.westervelt_frequency_hz
    for step in range(steps):
        t = step * dt
        local_t = t - delay
        active = (local_t >= 0.0) & (
            local_t < config.westervelt_cycles / config.westervelt_frequency_hz
        )
        if not np.any(active):
            continue
        source_step = local_t / dt
        ramp = np.minimum.reduce(
            [
                source_step / ramp_steps,
                (burst_steps - source_step) / ramp_steps,
                np.ones_like(source_step),
            ]
        )
        wave = np.sin(omega * local_t)
        samples[step, :] = (
            config.westervelt_source_pressure_pa * config.westervelt_source_gain
            * np.clip(ramp, 0.0, 1.0)
            * aperture
            * wave
            * active
        )
    return BoundarySource(
        column=column,
        samples=samples,
        focus_arrival_step=focus_arrival_step,
        burst_steps=burst_steps,
    )


def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """Second-order five-point Laplacian with zero normal edge extension."""

    padded = np.pad(field, 1, mode="edge")
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * field
    ) / (dx * dx)


def density_map(ct_hu: np.ndarray) -> np.ndarray:
    """Map CT HU to bounded acoustic density."""

    ct = ct_hu.astype(np.float32)
    rho = 1000.0 + 0.35 * np.clip(ct, -300.0, 1200.0)
    return np.clip(rho, 850.0, 1850.0).astype(np.float32)


def nonlinearity_map(ct_hu: np.ndarray) -> np.ndarray:
    """Map CT HU to the combined Westervelt nonlinearity coefficient beta."""

    ct = ct_hu.astype(np.float32)
    beta = np.full(ct.shape, 4.5, dtype=np.float32)
    beta[(ct > -100.0) & (ct <= 250.0)] = 5.5
    beta[ct > 250.0] = 6.5
    return beta


def damping_mask(shape: tuple[int, int], cells: int) -> np.ndarray:
    """Quadratic absorbing sponge for bounded FDTD simulations."""

    if cells <= 0:
        return np.ones(shape, dtype=np.float32)
    yy, xx = np.indices(shape)
    distance = np.minimum.reduce([yy, xx, shape[0] - 1 - yy, shape[1] - 1 - xx])
    depth = np.clip((cells - distance) / cells, 0.0, 1.0)
    return np.exp(-0.18 * depth * depth).astype(np.float32)


def rayleigh_plesset_subharmonic(
    fundamental_pressure_pa: np.ndarray,
    lesion_mask: np.ndarray,
    ct_hu: np.ndarray,
    config,
    dt: float,
    steps: int,
) -> np.ndarray:
    """Compute lesion-localized transient subharmonic bubble emission."""

    active = np.asarray(lesion_mask, dtype=bool)
    out = np.zeros(fundamental_pressure_pa.shape, dtype=np.float32)
    if not np.any(active):
        return out

    drive = config.bubble_drive_gain * fundamental_pressure_pa[active].astype(np.float64)
    if float(np.max(drive)) <= 0.0:
        return out
    texture = np.clip(ct_hu[active].astype(np.float64), -100.0, 150.0)
    texture = texture - float(np.mean(texture))
    r0 = config.bubble_radius_m * (1.0 + 0.08 * texture / max(float(np.ptp(texture)), 1.0))
    response = bubble_half_frequency_response(r0, drive, config, dt, steps)
    out[active] = normalize_source(response)
    return out


def bubble_half_frequency_response(
    r0: np.ndarray,
    drive_pa: np.ndarray,
    config,
    dt: float,
    steps: int,
) -> np.ndarray:
    """Integrate the Rayleigh-Plesset equation and demodulate at f0/2."""

    rho = 1000.0
    sigma = 0.072
    viscosity = 0.001
    gamma = 1.4
    p0 = 101_325.0
    pv = 2_338.0
    omega = 2.0 * np.pi * config.westervelt_frequency_hz
    r = r0.copy()
    rdot = np.zeros_like(r)
    hard_core = 0.18 * r0
    max_radius = config.bubble_max_radius_ratio * r0
    substeps = max(int(config.bubble_substeps), 1)
    sub_dt = dt / substeps
    acc = np.zeros_like(r, dtype=np.complex128)
    count = 0
    gas0 = p0 + 2.0 * sigma / r0 - pv

    for step in range(steps):
        for substep in range(substeps):
            t = (step + substep / substeps) * dt
            p_ac = drive_pa * np.sin(omega * t)
            gas = gas0 * np.power(np.maximum(r0 / r, 1.0e-6), 3.0 * gamma)
            rhs = gas + pv - p0 - p_ac - 2.0 * sigma / r - 4.0 * viscosity * rdot / r
            rddot = rhs / (rho * r) - 1.5 * rdot * rdot / r
            rdot = np.clip(
                rdot + sub_dt * rddot,
                -config.bubble_wall_speed_limit_m_s,
                config.bubble_wall_speed_limit_m_s,
            )
            r_next = r + sub_dt * rdot
            lower_hit = r_next < hard_core
            upper_hit = r_next > max_radius
            r = np.clip(r_next, hard_core, max_radius)
            rdot[lower_hit & (rdot < 0.0)] = 0.0
            rdot[upper_hit & (rdot > 0.0)] = 0.0
        if step >= steps // 2:
            t = step * dt
            acc += (np.power(r / r0, 3.0) - 1.0) * np.exp(-0.5j * omega * t)
            count += 1
    return (2.0 * np.abs(acc) / max(count, 1)).astype(np.float32)


def lesion_activity(mask: np.ndarray) -> np.ndarray:
    """Return a float lesion activity map for nonlinear source weighting."""

    return np.asarray(mask, dtype=np.float32)


def normalize_source(values: np.ndarray) -> np.ndarray:
    """Normalize a nonnegative source map while preserving zero maps."""

    values = np.asarray(values, dtype=np.float32)
    peak = float(np.max(np.abs(values)))
    if peak <= 0.0:
        return np.zeros(values.shape, dtype=np.float32)
    return (values / peak).astype(np.float32)
