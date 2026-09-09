//! Dimensioned parameter types for the Python boundary.
//!
//! Every physical parameter here used to arrive as a bare `f64` and be wrapped
//! into its Aequitas quantity one frame inside the boundary, so the type
//! safety the rest of the stack relies on began *after* the value crossed.
//! From Python a radius was a naked float whose dimension lived in its
//! parameter name.
//!
//! These aliases move that check to the boundary. Each accepts either form:
//!
//! - a bare `float` or `int`, taken as canonical SI base units exactly as
//!   before, so every already-published call site keeps working unchanged;
//! - any object carrying the Aequitas quantity protocol -- a `pyaequitas`
//!   quantity, or anything exposing `__aequitas_base__` and
//!   `__aequitas_dimension__` -- whose dimension is checked against the
//!   parameter's own, so a time passed where a length belongs raises instead
//!   of being silently scaled.
//!
//! The check is a comparison of two 8-byte tags: the expected one is a
//! constant of the type parameter. The extraction itself lives upstream in
//! `aequitas_python` so this crate holds no second copy of the protocol.
//!
//! Sibling of [`crate::array_utils`], which does the same centralising job for
//! NumPy conversions.

use aequitas::systems::si::dimensions;
use aequitas::systems::si::quantities::Angle;
use aequitas::systems::si::units::Degree;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{Borrowed, FromPyObject};

use aequitas_python::protocol::{BASE_ATTR, DIMENSION_ATTR};

/// Parameter accepting a base-unit float or a matching quantity.
pub use aequitas_python::{Dimensioned, MayBeInfinite};

/// Length in metres.
pub type PyLength = Dimensioned<dimensions::Length>;

/// Length in metres, admitting an infinite sentinel.
///
/// `TransducerArray2D.set_focus_distance` documents `INF` as "no focusing" and
/// clears the focus on it, so this parameter opts out of the finite-only
/// default rather than losing a published contract. `NaN` stays rejected --
/// the extractor admits the sentinel, not the absence of a value.
pub type PyFocusDistance = Dimensioned<dimensions::Length, MayBeInfinite>;

/// Plane or rotational angle in radians.
///
/// Note the base unit: a bare float reaching an angle parameter is radians,
/// which is not always the historical contract. A site that documented degrees
/// keeps converting its float and accepts a quantity separately, rather than
/// silently reinterpreting the number.
pub type PyAngle = Dimensioned<dimensions::Angle>;

/// Frequency in hertz.
pub type PyFrequency = Dimensioned<dimensions::Frequency>;

/// Pressure in pascals.
pub type PyPressure = Dimensioned<dimensions::Pressure>;

/// Velocity in metres per second.
pub type PyVelocity = Dimensioned<dimensions::Velocity>;

/// Mass density in kilograms per cubic metre.
pub type PyMassDensity = Dimensioned<dimensions::MassDensity>;

/// Electric potential in volts.
pub type PyElectricPotential = Dimensioned<dimensions::ElectricPotential>;

/// Reciprocal length in inverse metres.
pub type PyReciprocalLength = Dimensioned<dimensions::ReciprocalLength>;

/// Dimensionless ratio.
///
/// Still worth naming: it rejects a quantity that carries a dimension, so a
/// pressure cannot be passed where a fraction belongs.
pub type PyDimensionless = Dimensioned<dimensions::Dimensionless>;

/// Parameter whose bare-float arm is **degrees**, not base units.
///
/// `TransducerArray2D.set_steering_angle` has always documented and accepted
/// degrees. Typing it as [`PyAngle`] would keep that call site compiling while
/// silently reinterpreting every existing caller's number as radians, so the
/// float arm keeps its published meaning and only the quantity arm is new. A
/// quantity is read in radians, its own base unit, whatever unit it was built
/// from -- `angle(30.0, "deg")` and `angle(pi/6, "rad")` are the same value
/// here, and both differ from the bare float `30.0`.
#[derive(Clone, Copy, Debug)]
pub struct PyDegrees(Angle<f64>);

impl PyDegrees {
    /// The angle, as an Aequitas quantity in radians.
    #[must_use]
    pub const fn quantity(self) -> Angle<f64> {
        self.0
    }
}

impl<'py> FromPyObject<'_, 'py> for PyDegrees {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        // Order matters here more than anywhere: `f64` extraction honours
        // `__float__`, so reading the number first would take a quantity's
        // magnitude and call it degrees whatever unit it actually carries.
        // A declared quantity goes through the dimension check.
        let declares_quantity = object.hasattr(BASE_ATTR).unwrap_or(false)
            && object.hasattr(DIMENSION_ATTR).unwrap_or(false);
        if declares_quantity {
            let angle: PyAngle = object.extract()?;
            return Ok(Self(angle.quantity()));
        }
        match object.extract::<f64>() {
            Ok(degrees) if degrees.is_finite() => Ok(Self(Angle::from_unit::<Degree>(degrees))),
            Ok(_) => Err(PyValueError::new_err("angle must be finite")),
            // Neither a number nor a declared quantity: let the quantity
            // extractor produce the diagnostic naming what it wanted.
            Err(_) => {
                let angle: PyAngle = object.extract()?;
                Ok(Self(angle.quantity()))
            }
        }
    }
}

#[cfg(test)]
mod tests;
