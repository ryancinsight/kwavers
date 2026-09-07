//! Clamped-plate MEMS resonance bindings.

use aequitas::systems::si::units::Hertz;
use kwavers_transducer::mems::plate;
use pyo3::prelude::*;

use crate::quantity_args::{PyDimensionless, PyFrequency, PyLength, PyMassDensity, PyPressure};

/// Clamped circular plate in-vacuo fundamental resonance `Hz`.
///
/// Each parameter accepts a float in canonical SI base units, or a quantity of
/// the matching dimension.
#[pyfunction]
pub fn mems_clamped_plate_resonance(
    youngs: PyPressure,
    thickness: PyLength,
    poisson: PyDimensionless,
    density: PyMassDensity,
    radius: PyLength,
) -> f64 {
    plate::vacuum_resonance(
        youngs.quantity(),
        thickness.quantity(),
        poisson.quantity(),
        density.quantity(),
        radius.quantity(),
    )
    .in_unit::<Hertz>()
}

/// Lamb fluid-loaded (immersion) resonance `Hz`.
///
/// Each parameter accepts a float in canonical SI base units, or a quantity of
/// the matching dimension.
#[pyfunction]
pub fn mems_immersion_resonance(
    vacuum_freq: PyFrequency,
    density_plate: PyMassDensity,
    thickness: PyLength,
    density_fluid: PyMassDensity,
    radius: PyLength,
) -> f64 {
    plate::immersion_resonance(
        vacuum_freq.quantity(),
        density_plate.quantity(),
        thickness.quantity(),
        density_fluid.quantity(),
        radius.quantity(),
    )
    .in_unit::<Hertz>()
}
