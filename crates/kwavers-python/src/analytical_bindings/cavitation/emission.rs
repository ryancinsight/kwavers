//! Bubble and population emission simulation PyO3 wrappers.

use kwavers_physics::analytical::cavitation;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// True single-bubble acoustic-emission simulation via the production adaptive
/// Keller–Miksis solver (gas thermodynamics, mass transfer, compressible
/// radiation damping; Richardson-extrapolation adaptive sub-stepping that
/// survives inertial collapse where a fixed-step RK4 diverges).
///
/// Drives the bubble with p_ac(t) = drive_amp·sin(2π f t) and records the
/// far-field emission p_sc(t) = rho·R/r_obs·(2 Rdot² + R Rddot) using the exact
/// wall acceleration. The harmonic/subharmonic/broadband content of the
/// resulting spectrum is emergent, not imposed.
///
/// Args:
///     r0_m: Equilibrium radius [m].
///     drive_amp_pa: Peak acoustic drive pressure [Pa].
///     drive_freq_hz: Drive frequency [Hz].
///     n_cycles: Number of drive cycles to simulate.
///     n_out: Number of uniform output samples.
///     r_obs_m: Far-field observation distance [m].
///     p0_pa, rho, c_liquid, mu, sigma, pv, gamma: liquid/gas properties.
///     thermal_effects: include gas thermodynamics + mass transfer.
///
/// Returns:
///     (time, radius, wall_velocity, emission, max_compression, max_mach,
///      collapse_count, converged) — four arrays then four diagnostics.
#[pyfunction]
#[pyo3(signature = (r0_m, drive_amp_pa, drive_freq_hz, n_cycles, n_out, r_obs_m,
                    p0_pa=101_325.0, rho=998.0, c_liquid=1481.0, mu=1.0e-3,
                    sigma=0.0725, pv=2330.0, gamma=1.4, thermal_effects=false))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn simulate_bubble_emission(
    py: Python<'_>,
    r0_m: f64,
    drive_amp_pa: f64,
    drive_freq_hz: f64,
    n_cycles: f64,
    n_out: usize,
    r_obs_m: f64,
    p0_pa: f64,
    rho: f64,
    c_liquid: f64,
    mu: f64,
    sigma: f64,
    pv: f64,
    gamma: f64,
    thermal_effects: bool,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    u32,
    bool,
)> {
    let cfg = cavitation::BubbleDriveConfig {
        r0_m,
        p0_pa,
        rho,
        c_liquid,
        mu,
        sigma,
        pv,
        gamma,
        drive_freq_hz,
        drive_amp_pa,
        n_cycles,
        n_out,
        r_obs_m,
        thermal_effects,
    };
    let tr = cavitation::simulate_bubble_emission(&cfg);
    Ok((
        tr.time.to_pyarray(py).unbind(),
        tr.radius.to_pyarray(py).unbind(),
        tr.wall_velocity.to_pyarray(py).unbind(),
        tr.emission.to_pyarray(py).unbind(),
        tr.max_compression,
        tr.max_mach,
        tr.collapse_count,
        tr.converged,
    ))
}

/// True *coated* (encapsulated) microbubble emission simulation via the
/// Marmottant shell model (lipid/protein shell with buckling and rupture).
///
/// The shell's piecewise surface tension (σ→0 when buckled, σ→σ_water when
/// ruptured) period-doubles the dynamics, so a clinical contrast microbubble
/// emits a SUBHARMONIC at low drive pressures where a free bubble does not —
/// the marker BBB-opening controllers track. Shell-damped Rayleigh–Plesset is
/// integrated with a fixed-step RK4; the emergent emission spectrum is returned.
///
/// Args:
///     r0_m, drive_amp_pa, drive_freq_hz, n_cycles, n_out, r_obs_m: as for
///         simulate_bubble_emission.
///     chi: shell elastic compression modulus χ [N/m] (lipid ≈ 0.25–1.0).
///     shell_viscosity: shell shear viscosity [Pa·s] (lipid ≈ 0.5).
///     shell_thickness: shell thickness [m] (lipid ≈ 3e-9).
///     sigma_initial: unstressed shell surface tension [N/m] (≈ 0.04).
///     steps_per_cycle: RK4 sub-steps per drive cycle.
///     p0_pa, rho, c_liquid, mu, gamma: liquid/gas properties.
///
/// Returns:
///     (time, radius, wall_velocity, emission, max_compression, max_mach,
///      collapse_count, converged).
#[pyfunction]
#[pyo3(signature = (r0_m, drive_amp_pa, drive_freq_hz, n_cycles, n_out, r_obs_m,
                    chi=0.5, shell_viscosity=0.5, shell_thickness=3.0e-9,
                    sigma_initial=0.04, steps_per_cycle=2000, p0_pa=101_325.0,
                    rho=998.0, c_liquid=1481.0, mu=1.0e-3, gamma=1.4))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn simulate_coated_bubble_emission(
    py: Python<'_>,
    r0_m: f64,
    drive_amp_pa: f64,
    drive_freq_hz: f64,
    n_cycles: f64,
    n_out: usize,
    r_obs_m: f64,
    chi: f64,
    shell_viscosity: f64,
    shell_thickness: f64,
    sigma_initial: f64,
    steps_per_cycle: usize,
    p0_pa: f64,
    rho: f64,
    c_liquid: f64,
    mu: f64,
    gamma: f64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    u32,
    bool,
)> {
    let cfg = cavitation::ShellDriveConfig {
        r0_m,
        p0_pa,
        rho,
        c_liquid,
        mu,
        gamma,
        drive_freq_hz,
        drive_amp_pa,
        n_cycles,
        steps_per_cycle,
        n_out,
        r_obs_m,
        chi,
        shell_viscosity,
        shell_thickness,
        sigma_initial,
    };
    let tr = cavitation::simulate_coated_bubble_emission(&cfg);
    Ok((
        tr.time.to_pyarray(py).unbind(),
        tr.radius.to_pyarray(py).unbind(),
        tr.wall_velocity.to_pyarray(py).unbind(),
        tr.emission.to_pyarray(py).unbind(),
        tr.max_compression,
        tr.max_mach,
        tr.collapse_count,
        tr.converged,
    ))
}

/// Simulate population-level passive-cavitation emission.
///
/// Draws a deterministic log-normal bubble-radius population, runs each bubble
/// through the Rust free or Marmottant-shell emission solver, rejects non-finite
/// or non-physical traces, builds the population PSD with the Apollo-backed FFT,
/// and decomposes it into harmonic, subharmonic, ultraharmonic, and broadband
/// bands.
#[pyfunction]
#[pyo3(signature = (
    drive_pa, f0_hz, n_bubbles, seed, r0_median_m=1.5e-6, r0_sigma_ln=0.4,
    n_cycles=12.0, n_out=8192, r_obs_m=5.0e-2, rel_halfwidth=0.12,
    noise_floor=0.0, thermal_effects=false, coated=false, chi=0.5,
    shell_viscosity=0.5, shell_thickness=3.0e-9, sigma_initial=0.04,
    steps_per_cycle=2000, p0_pa=101_325.0, rho=998.0, c_liquid=1481.0,
    mu=1.0e-3, sigma=0.0725, pv=2330.0, gamma=1.4
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn simulate_population_emission(
    py: Python<'_>,
    drive_pa: f64,
    f0_hz: f64,
    n_bubbles: usize,
    seed: u64,
    r0_median_m: f64,
    r0_sigma_ln: f64,
    n_cycles: f64,
    n_out: usize,
    r_obs_m: f64,
    rel_halfwidth: f64,
    noise_floor: f64,
    thermal_effects: bool,
    coated: bool,
    chi: f64,
    shell_viscosity: f64,
    shell_thickness: f64,
    sigma_initial: f64,
    steps_per_cycle: usize,
    p0_pa: f64,
    rho: f64,
    c_liquid: f64,
    mu: f64,
    sigma: f64,
    pv: f64,
    gamma: f64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    usize,
    f64,
    f64,
)> {
    let result = py
        .detach(|| {
            cavitation::simulate_population_emission(cavitation::PopulationEmissionInput {
                drive_pa,
                f0_hz,
                n_bubbles,
                seed,
                r0_median_m,
                r0_sigma_ln,
                n_cycles,
                n_out,
                r_obs_m,
                rel_halfwidth,
                noise_floor,
                thermal_effects,
                medium: cavitation::PopulationMedium {
                    p0_pa,
                    rho,
                    c_liquid,
                    mu,
                    sigma,
                    pv,
                    gamma,
                },
                shell: cavitation::PopulationShell {
                    coated,
                    chi,
                    shell_viscosity,
                    shell_thickness,
                    sigma_initial,
                    steps_per_cycle,
                },
            })
        })
        .ok_or_else(|| PyRuntimeError::new_err("invalid population emission parameters"))?;
    let stable = result.bands.stable_emission();
    let total = result.bands.fundamental + stable + result.bands.broadband;
    Ok((
        result.freqs_hz.to_pyarray(py).unbind(),
        result.psd.to_pyarray(py).unbind(),
        result.bands.fundamental,
        result.bands.subharmonic,
        result.bands.ultraharmonic,
        result.bands.broadband,
        stable,
        total,
        result.n_active,
        result.max_compression,
        result.max_mach,
    ))
}

/// Simulate a pressure sweep of population-level passive-cavitation emission.
///
/// Returns `(harmonic, subharmonic, ultraharmonic, stable, inertial, signal,
/// n_active, max_compression, max_mach)`. The Rust core owns the pressure loop,
/// deterministic per-pressure population seeds, per-pressure population
/// emission simulation, and band aggregation.
#[pyfunction]
#[pyo3(signature = (
    pressures_pa, f0_hz, n_bubbles, seed, r0_median_m=1.5e-6,
    r0_sigma_ln=0.4, n_cycles=12.0, n_out=8192, r_obs_m=5.0e-2,
    rel_halfwidth=0.12, noise_floor=0.0, thermal_effects=false,
    coated=false, chi=0.5, shell_viscosity=0.5, shell_thickness=3.0e-9,
    sigma_initial=0.04, steps_per_cycle=2000, p0_pa=101_325.0,
    rho=998.0, c_liquid=1481.0, mu=1.0e-3, sigma=0.0725,
    pv=2330.0, gamma=1.4
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn population_emission_sweep(
    py: Python<'_>,
    pressures_pa: PyReadonlyArray1<f64>,
    f0_hz: f64,
    n_bubbles: usize,
    seed: u64,
    r0_median_m: f64,
    r0_sigma_ln: f64,
    n_cycles: f64,
    n_out: usize,
    r_obs_m: f64,
    rel_halfwidth: f64,
    noise_floor: f64,
    thermal_effects: bool,
    coated: bool,
    chi: f64,
    shell_viscosity: f64,
    shell_thickness: f64,
    sigma_initial: f64,
    steps_per_cycle: usize,
    p0_pa: f64,
    rho: f64,
    c_liquid: f64,
    mu: f64,
    sigma: f64,
    pv: f64,
    gamma: f64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<usize>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
    let pressures = pressures_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let sweep = py
        .detach(|| {
            cavitation::population_emission_sweep(cavitation::PopulationEmissionSweepInput {
                pressures_pa: pressures,
                f0_hz,
                n_bubbles,
                seed,
                r0_median_m,
                r0_sigma_ln,
                n_cycles,
                n_out,
                r_obs_m,
                rel_halfwidth,
                noise_floor,
                thermal_effects,
                medium: cavitation::PopulationMedium {
                    p0_pa,
                    rho,
                    c_liquid,
                    mu,
                    sigma,
                    pv,
                    gamma,
                },
                shell: cavitation::PopulationShell {
                    coated,
                    chi,
                    shell_viscosity,
                    shell_thickness,
                    sigma_initial,
                    steps_per_cycle,
                },
            })
        })
        .ok_or_else(|| PyRuntimeError::new_err("invalid population emission sweep parameters"))?;

    Ok((
        sweep.harmonic.to_pyarray(py).unbind(),
        sweep.subharmonic.to_pyarray(py).unbind(),
        sweep.ultraharmonic.to_pyarray(py).unbind(),
        sweep.stable.to_pyarray(py).unbind(),
        sweep.inertial.to_pyarray(py).unbind(),
        sweep.signal.to_pyarray(py).unbind(),
        sweep.n_active.to_pyarray(py).unbind(),
        sweep.max_compression.to_pyarray(py).unbind(),
        sweep.max_mach.to_pyarray(py).unbind(),
    ))
}

/// Receiver/array-integrated emission spectrum for one focal volume `V_s`.
#[pyfunction]
#[pyo3(signature = (
    drive_pa, f0_hz, r0_population_m, n_cycles=12.0, steps_per_cycle=4000,
    r_obs_m=5.0e-2, n_fft=2048, transient_fraction=0.4, p0_pa=101_325.0,
    rho=998.0, sigma=0.0725, gamma=1.4, mu=1.0e-3, pv=2330.0,
    c_liquid=1481.0, xi_s=0.0
))]
#[allow(clippy::too_many_arguments)]
pub fn volume_emission_spectrum(
    py: Python<'_>,
    drive_pa: f64,
    f0_hz: f64,
    r0_population_m: PyReadonlyArray1<f64>,
    n_cycles: f64,
    steps_per_cycle: usize,
    r_obs_m: f64,
    n_fft: usize,
    transient_fraction: f64,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    gamma: f64,
    mu: f64,
    pv: f64,
    c_liquid: f64,
    xi_s: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, usize)> {
    let radii = r0_population_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let spectrum = py
        .detach(|| {
            cavitation::volume_emission_spectrum(cavitation::VolumeEmissionSpectrumInput {
                drive_pa,
                f0_hz,
                r0_population_m: radii,
                medium: cavitation::VolumeSpectrumMedium {
                    p0_pa,
                    rho,
                    sigma,
                    gamma,
                    mu,
                    pv_pa: pv,
                    c_liquid,
                    xi_s,
                },
                n_cycles,
                steps_per_cycle,
                r_obs_m,
                n_fft,
                transient_fraction,
            })
        })
        .ok_or_else(|| PyRuntimeError::new_err("invalid V_s emission spectrum parameters"))?;

    Ok((
        spectrum.freqs_hz.to_pyarray(py).unbind(),
        spectrum.psd.to_pyarray(py).unbind(),
        spectrum.n_active,
    ))
}

/// Band-resolved V_s-integrated emission pressure sweep.
#[pyfunction]
#[pyo3(signature = (
    pressures_pa, f0_hz, r0_population_m, rel_halfwidth=0.04,
    noise_floor=0.0, n_cycles=12.0, steps_per_cycle=4000, r_obs_m=5.0e-2,
    n_fft=2048, transient_fraction=0.4, p0_pa=101_325.0, rho=998.0,
    sigma=0.0725, gamma=1.4, mu=1.0e-3, pv=2330.0, c_liquid=1481.0,
    xi_s=0.0
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn volume_emission_sweep(
    py: Python<'_>,
    pressures_pa: PyReadonlyArray1<f64>,
    f0_hz: f64,
    r0_population_m: PyReadonlyArray1<f64>,
    rel_halfwidth: f64,
    noise_floor: f64,
    n_cycles: f64,
    steps_per_cycle: usize,
    r_obs_m: f64,
    n_fft: usize,
    transient_fraction: f64,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    gamma: f64,
    mu: f64,
    pv: f64,
    c_liquid: f64,
    xi_s: f64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<usize>>,
)> {
    let pressures = pressures_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let radii = r0_population_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let sweep = py
        .detach(|| {
            cavitation::volume_emission_sweep(cavitation::VolumeEmissionSweepInput {
                pressures_pa: pressures,
                f0_hz,
                r0_population_m: radii,
                medium: cavitation::VolumeSpectrumMedium {
                    p0_pa,
                    rho,
                    sigma,
                    gamma,
                    mu,
                    pv_pa: pv,
                    c_liquid,
                    xi_s,
                },
                rel_halfwidth,
                noise_floor,
                n_cycles,
                steps_per_cycle,
                r_obs_m,
                n_fft,
                transient_fraction,
            })
        })
        .ok_or_else(|| PyRuntimeError::new_err("invalid V_s emission sweep parameters"))?;

    Ok((
        sweep.harmonic.to_pyarray(py).unbind(),
        sweep.subharmonic.to_pyarray(py).unbind(),
        sweep.ultraharmonic.to_pyarray(py).unbind(),
        sweep.stable.to_pyarray(py).unbind(),
        sweep.inertial.to_pyarray(py).unbind(),
        sweep.n_active.to_pyarray(py).unbind(),
    ))
}
