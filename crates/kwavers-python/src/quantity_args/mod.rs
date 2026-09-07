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

/// Parameter accepting a base-unit float or a matching quantity.
pub use aequitas_python::Dimensioned;

/// Length in metres.
pub type PyLength = Dimensioned<dimensions::Length>;

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

#[cfg(test)]
mod tests;
