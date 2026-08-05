//! Clamped-plate MEMS resonance bindings.

use aequitas::systems::si::quantities::{Dimensionless, Frequency, Length, MassDensity, Pressure};
use aequitas::systems::si::units::{Hertz, KilogramPerCubicMeter, Meter, Pascal};
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
        Pressure::from_unit::<Pascal>(youngs),
        Length::from_unit::<Meter>(thickness),
        Dimensionless::from_base(poisson),
        MassDensity::from_unit::<KilogramPerCubicMeter>(density),
        Length::from_unit::<Meter>(radius),
    )
    .in_unit::<Hertz>()
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
        Frequency::from_unit::<Hertz>(vacuum_freq),
        MassDensity::from_unit::<KilogramPerCubicMeter>(density_plate),
        Length::from_unit::<Meter>(thickness),
        MassDensity::from_unit::<KilogramPerCubicMeter>(density_fluid),
        Length::from_unit::<Meter>(radius),
    )
    .in_unit::<Hertz>()
}
