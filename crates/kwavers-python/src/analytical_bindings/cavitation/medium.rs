//! Residual-gas and bubbly-medium PyO3 wrappers.

use pyo3::prelude::*;

/// Epstein–Plesset gas-diffusion dissolution time of a free air bubble in water.
///
/// Integrates the complete Epstein–Plesset (1950) model (Laplace-overpressure
/// surface-tension drive + transient diffusion term) to the time R₀ → ~0, giving
/// the first-principles residual-bubble dissolution time τ_d that governs
/// inter-pulse shielding.
///
/// Args:
///     r0_m: equilibrium bubble radius [m].
///     saturation_fraction: dissolved-gas saturation f = C∞/C_s (1.0 saturated).
///
/// Returns:
///     Dissolution time [s], or -1.0 if the bubble does not dissolve in the
///     integration window (e.g. supersaturated/growing).
///
/// Reference: Epstein & Plesset (1950) J. Chem. Phys. 18, 1505.
#[pyfunction]
#[pyo3(signature = (r0_m, saturation_fraction=0.5))]
pub fn epstein_plesset_dissolution_time(r0_m: f64, saturation_fraction: f64) -> PyResult<f64> {
    use kwavers_physics::acoustics::bubble_dynamics::{
        dissolution_time_numeric, EpsteinPlessetDissolution, GasDiffusionParams,
    };
    let model =
        EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(saturation_fraction));
    Ok(dissolution_time_numeric(&model, r0_m, 1e-9).unwrap_or(-1.0))
}

/// Shelled-microbubble (Sarkar 2009) dissolution time with a finite shell
/// gas-permeability — the coated-contrast-agent persistence time.
///
/// Args:
///     r0_m: equilibrium radius [m].
///     saturation_fraction: dissolved-gas saturation f.
///     shell_permeability_m_s: shell gas-permeation coefficient k_s [m/s]
///         (lipid ≈ 1e-6; k_s→∞ recovers the free bubble, k_s→0 stabilises it).
///
/// Returns:
///     Dissolution time [s], or -1.0 if it does not dissolve in the window.
///
/// Reference: Sarkar, Katiyar & Jain (2009) Ann. Biomed. Eng. 37, 2196.
#[pyfunction]
#[pyo3(signature = (r0_m, saturation_fraction=0.5, shell_permeability_m_s=1.0e-6))]
pub fn shelled_dissolution_time(
    r0_m: f64,
    saturation_fraction: f64,
    shell_permeability_m_s: f64,
) -> PyResult<f64> {
    use kwavers_physics::acoustics::bubble_dynamics::{
        dissolution_time_numeric, GasDiffusionParams, ShellPermeationDissolution,
    };
    let model = ShellPermeationDissolution::new(
        GasDiffusionParams::air_in_water(saturation_fraction),
        shell_permeability_m_s,
    );
    Ok(dissolution_time_numeric(&model, r0_m, 1e-9).unwrap_or(-1.0))
}

/// Low-frequency sound speed of a bubbly liquid (Wood 1930) at gas void fraction β.
///
/// Args:
///     void_fraction, c_liquid, rho_liquid, c_gas, rho_gas.
///
/// Returns:
///     Mixture sound speed [m/s] (collapses far below c_liquid for β ≳ 1e-4).
#[pyfunction]
#[pyo3(signature = (void_fraction, c_liquid=1481.0, rho_liquid=998.0, c_gas=343.0, rho_gas=1.2))]
pub fn wood_sound_speed(
    void_fraction: f64,
    c_liquid: f64,
    rho_liquid: f64,
    c_gas: f64,
    rho_gas: f64,
) -> PyResult<f64> {
    use kwavers_physics::acoustics::bubble_dynamics::wood_sound_speed as wood;
    Ok(wood(void_fraction, c_liquid, rho_liquid, c_gas, rho_gas))
}

/// Resonant-scattering attenuation of a monodisperse bubble cloud
/// (Commander & Prosperetti 1989), in Np/m.
///
/// Args:
///     freq_hz, void_fraction, r0_m, c_liquid, rho_liquid, mu_liquid, p0_pa, polytropic.
#[pyfunction]
#[pyo3(signature = (freq_hz, void_fraction, r0_m, c_liquid=1481.0, rho_liquid=998.0,
                    mu_liquid=1.0e-3, p0_pa=101_325.0, polytropic=1.4))]
#[allow(clippy::too_many_arguments)]
pub fn bubbly_cloud_attenuation(
    freq_hz: f64,
    void_fraction: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> PyResult<f64> {
    use kwavers_physics::acoustics::bubble_dynamics::commander_prosperetti_attenuation as cp;
    Ok(cp(
        freq_hz,
        void_fraction,
        r0_m,
        c_liquid,
        rho_liquid,
        mu_liquid,
        p0_pa,
        polytropic,
    ))
}

/// Frequency-dependent phase velocity of a monodisperse bubble cloud
/// (Commander & Prosperetti 1989), in m/s — the dispersive companion of
/// [`bubbly_cloud_attenuation`].
///
/// The real part of the complex mixture wavenumber sets `c_p(ω) = ω/Re(k_m)`.
/// Below the bubble (Minnaert) resonance the cloud slows the wave toward the
/// Wood limit; above resonance the phase velocity exceeds `c_liquid` (anomalous
/// dispersion). Its `ω→0` limit reproduces `wood_sound_speed`.
///
/// Args:
///     freq_hz, void_fraction, r0_m, c_liquid, rho_liquid, mu_liquid, p0_pa, polytropic.
///
/// Returns:
///     Phase velocity c_p(f) [m/s].
#[pyfunction]
#[pyo3(signature = (freq_hz, void_fraction, r0_m, c_liquid=1481.0, rho_liquid=998.0,
                    mu_liquid=1.0e-3, p0_pa=101_325.0, polytropic=1.4))]
#[allow(clippy::too_many_arguments)]
pub fn bubbly_cloud_phase_velocity(
    freq_hz: f64,
    void_fraction: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> PyResult<f64> {
    use kwavers_physics::acoustics::bubble_dynamics::commander_prosperetti_phase_velocity as cp;
    Ok(cp(
        freq_hz,
        void_fraction,
        r0_m,
        c_liquid,
        rho_liquid,
        mu_liquid,
        p0_pa,
        polytropic,
    ))
}
