//! Wave physics functions for book chapters ch01, ch02, ch03, ch08.
//!
//! Covers: linear wave equations, plane and spherical waves, reflection /
//! transmission, power-law attenuation, numerical dispersion, nonlinear
//! harmonic generation (Fubini / Westervelt), and shock formation.

mod bessel;
mod dispersion;
mod linear;
mod nonlinear;
#[cfg(test)]
mod tests;

pub use dispersion::{
    fdtd_cfl_limit, fdtd_phase_error_1d, kspace_correction_error, pstd_phase_error,
};
pub use linear::{
    absorption_power_law_db_cm, plane_wave_pressure_1d, power_law_attenuation_np_m,
    pulse_train_waveform, reflection_pressure_coeff, spherical_wave_pressure, standing_wave_1d,
    stokes_kirchhoff_absorption_np_m, tone_burst_waveform, transmission_pressure_coeff,
};
pub use nonlinear::{
    fubini_harmonic_amplitude, fubini_harmonic_spectrum, fubini_waveform,
    goldberg_shock_parameter_sweep, shock_enhanced_absorption_gain, shock_formation_distance,
    shock_heat_source_density, shock_vapor_pulse_waveform, shock_waveform_pressure,
    westervelt_harmonic_evolution,
};
