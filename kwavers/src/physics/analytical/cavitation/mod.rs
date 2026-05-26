//! Bubble dynamics and cavitation physics for book chapters ch07, ch09.
//!
//! Covers: Minnaert resonance, Blake threshold, Rayleigh collapse time,
//! Rayleigh–Plesset and Keller–Miksis ODE integrators (RK4), power spectrum,
//! mechanical index, inertial cavitation dose, and histotripsy lesion radius.

mod dynamics;
mod histotripsy;
mod power_spectrum;
#[cfg(test)]
mod tests;

pub use dynamics::{
    blake_threshold_pa, keller_miksis_rk4, minnaert_resonance_hz, rayleigh_collapse_time_s,
    rayleigh_plesset_rk4,
};
pub use histotripsy::{
    cumulative_cavitation_probability, frequency_dependent_intrinsic_threshold_pa,
    histotripsy_lesion_radius_m, inertial_cavitation_dose,
    intrinsic_threshold_cavitation_probability, mechanical_index, prf_efficacy_factor,
};
pub use power_spectrum::{bubble_power_spectrum, period_doubling_ratio};
