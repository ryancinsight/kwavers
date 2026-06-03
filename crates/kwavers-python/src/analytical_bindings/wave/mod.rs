//! PyO3 bindings for `kwavers_physics::analytical::wave`.
//!
//! Split by SRP domain:
//! - `propagation`: standing/plane/spherical waves, reflection, power-law attenuation
//! - `dispersion`: FDTD/PSTD phase error, k-space correction, CFL limit
//! - `nonlinear`: Fubini harmonics, shock physics, tone burst, Westervelt evolution

pub mod dispersion;
pub mod nonlinear;
pub mod propagation;

// Re-export all public items so call sites in analytical_bindings.rs
// continue to use `wave::function_name` without modification.
pub use dispersion::{
    fdtd_cfl_limit, fdtd_phase_error_1d, kspace_correction_error, pstd_phase_error,
};
pub use nonlinear::{
    fubini_harmonic_amplitude, fubini_harmonic_spectrum, fubini_waveform,
    goldberg_shock_parameter_sweep, pulse_train_waveform, shock_enhanced_absorption_gain,
    shock_formation_distance, shock_heat_source_density, shock_vapor_pulse_waveform,
    shock_waveform_pressure, tone_burst_waveform, westervelt_harmonic_evolution,
};
pub use propagation::{
    absorption_power_law_db_cm, plane_wave_pressure_1d, power_law_attenuation_np_m,
    reflection_pressure_coeff, spherical_wave_pressure, standing_wave_1d,
    stokes_kirchhoff_absorption_np_m, transmission_pressure_coeff,
};
