//! Native scalar operation bodies with one arithmetic specification.

use eunomia::{Complex, FloatElement};
use std::ops::Neg;

mod sealed {
    pub trait Sealed {}
}

/// Concrete scalar implementations typecheck the same operation bodies in each
/// native precision. Primitive operators then lower directly to MIR arithmetic,
/// avoiding generic operator forwarding without optimizer or interpreter flags.
/// Required methods prevent an implicit generic implementation from restoring
/// that forwarding. All numerical formulas live in `operations.rs.in`.
pub(in super::super) trait Arithmetic:
    FloatElement + Neg<Output = Self> + sealed::Sealed
{
    fn product(incident: Complex<Self>, green: Complex<Self>) -> Complex<Self>;
    fn weight_block<const K: usize>(values: &mut [Complex<Self>; K], weights: &[Self]);
    fn accumulate_block(sum: Complex<Self>, values: &[Complex<Self>]) -> Complex<Self>;
    fn project_block<const K: usize, const ACCUMULATE: bool>(
        values: &[Complex<Self>; K],
        weights: &[Self],
        residual: Complex<Self>,
        output: &mut [Self; K],
    );
}

include!(concat!(env!("OUT_DIR"), "/native_arithmetic.rs"));

#[cfg(test)]
mod tests;
