//! Clamped-plate MEMS resonance bindings.

use aequitas::systems::si::quantities::{Frequency, Length, MassDensity, Pressure};
use kwavers_transducer::mems::plate;
use pyo3::prelude::*;

/// Clamped circular plate in-vacuo fundamental resonance `Hz`.
#[pyfunction]
pub fn mems_clamped_plate_resonance(
    youngs: f64,
    thickness: f64,
    poisson: f64,
    density: f64,
    radius: f64,
) -> f64 {
    plate::vacuum_resonance(
        Pressure::from_base(youngs),
        Length::from_base(thickness),
        poisson,
        MassDensity::from_base(density),
        Length::from_base(radius),
    )
    .into_base()
}

/// Lamb fluid-loaded (immersion) resonance `Hz`.
#[pyfunction]
pub fn mems_immersion_resonance(
    vacuum_freq: f64,
    density_plate: f64,
    thickness: f64,
    density_fluid: f64,
    radius: f64,
) -> f64 {
    plate::immersion_resonance(
        Frequency::from_base(vacuum_freq),
        MassDensity::from_base(density_plate),
        Length::from_base(thickness),
        MassDensity::from_base(density_fluid),
        Length::from_base(radius),
    )
    .into_base()
}
