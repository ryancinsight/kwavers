//! Generic Tyche-design collection.

use leto::Array2;
use tyche_core::Design;

use super::collect_points;
use crate::geometry::{GeometricDomain, GeometryError};

/// Map Tyche unit-hypercube designs into a physical geometric domain.
///
/// The blanket implementation is canonical for every domain: implementors
/// supply only [`GeometricDomain::map_unit_interior`]. A design is sampled
/// directly into a stack point, transformed into a second stack point, and
/// appended to one exact-capacity output buffer. No per-point allocation or
/// intermediate design matrix exists.
pub trait DesignSamplingExt: GeometricDomain {
    /// Collect a fixed-dimensional design into a physical point matrix.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] when the design dimension differs from the
    /// domain, the design violates its declared sample range, a unit point is
    /// invalid, or the output shape is not addressable or reservable.
    fn sample_design<const PARAMETERS: usize, D: Design<PARAMETERS>>(
        &self,
        design: &D,
    ) -> Result<Array2<f64>, GeometryError> {
        let dimensions = self.dimension().as_usize();
        if dimensions != PARAMETERS {
            return Err(GeometryError::DimensionMismatch {
                role: "design",
                expected: dimensions,
                actual: PARAMETERS,
            });
        }

        collect_points::<PARAMETERS>(design.sample_count(), |index, point| {
            let mut unit = [0.0; PARAMETERS];
            design.sample_unit_into(index, &mut unit).map_err(|error| {
                GeometryError::DesignSample {
                    index: error.index(),
                    sample_count: error.sample_count(),
                }
            })?;
            self.map_unit_interior(&unit, point)
        })
    }
}

impl<G: GeometricDomain + ?Sized> DesignSamplingExt for G {}
