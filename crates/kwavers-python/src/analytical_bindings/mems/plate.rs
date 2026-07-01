//! Clamped-plate MEMS resonance bindings.

use kwavers_transducer::mems::plate;
use pyo3::prelude::*;

/// Clamped circular plate in-vacuo fundamental resonance [Hz].
#[pyfunction]
pub fn mems_clamped_plate_resonance(
    youngs: f64,
    thickness: f64,
    poisson: f64,
    density: f64,
    radius: f64,
) -> f64 {
    plate::vacuum_resonance(youngs, thickness, poisson, density, radius)
}

/// Lamb fluid-loaded (immersion) resonance [Hz].
#[pyfunction]
pub fn mems_immersion_resonance(
    vacuum_freq: f64,
    density_plate: f64,
    thickness: f64,
    density_fluid: f64,
    radius: f64,
) -> f64 {
    plate::immersion_resonance(vacuum_freq, density_plate, thickness, density_fluid, radius)
}
