//! Validated spatial domains and reproducible point sampling.
//!
//! A [`GeometricDomain`] owns the transformation from the unit hypercube to
//! its physical interior. [`sampling::DesignSamplingExt`] applies any Tyche
//! [`tyche_core::Design`] through that transformation without materializing an
//! intermediate design matrix.
//!
//! For an axis-aligned interval `[a, b]`, the affine map
//! `x = a + u(b-a)` has constant Jacobian `b-a`. For a disk, `r = R sqrt(u)`
//! makes `P(r <= q) = (q/R)^2`; for a ball, `r = R cbrt(u)` makes
//! `P(r <= q) = (q/R)^3`. The angular coordinates are uniform in angle in 2-D
//! and uniform in azimuth and `cos(theta)` in 3-D. These transforms therefore
//! preserve the corresponding normalized area or volume measure. The radial
//! construction follows the inverse-volume rule stated in Jonathan Goodman's
//! Monte Carlo notes, page 19:
//! <https://math.nyu.edu/~goodman/teaching/MonteCarlo17/notes/Week1.pdf>.
//!
//! Floating translation can round a sub-radius spherical point onto the closed
//! boundary. The spherical map moves each such component one representable
//! value toward the center. If the rounded norm is still indistinguishable
//! from the radius, it returns the center; validated positive measure proves
//! that the center lies strictly inside the representable open ball.

use leto::{Array1, Array2};
use tyche_core::Seed;

mod error;
mod rectangular;
pub mod sampling;
mod spherical;
#[cfg(test)]
mod tests;

pub use error::GeometryError;
pub use rectangular::RectangularDomain;
pub use spherical::SphericalDomain;

/// Spatial dimension of a geometric domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeometryDimension {
    /// One-dimensional interval.
    One,
    /// Two-dimensional region.
    Two,
    /// Three-dimensional region.
    Three,
}

impl GeometryDimension {
    /// Number of active coordinates.
    #[must_use]
    pub const fn as_usize(&self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }
}

/// Face of an axis-aligned grid domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainFace {
    /// Minimum x face.
    XMin,
    /// Maximum x face.
    XMax,
    /// Minimum y face.
    YMin,
    /// Maximum y face.
    YMax,
    /// Minimum z face.
    ZMin,
    /// Maximum z face.
    ZMax,
}

/// Point location relative to a domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointLocation {
    /// Strictly inside the domain.
    Interior,
    /// On the domain boundary within the supplied tolerance.
    Boundary,
    /// Outside the domain or invalid for the domain dimension.
    Exterior,
}

/// Solver-independent geometric-domain contract.
pub trait GeometricDomain: Send + Sync {
    /// Spatial dimension.
    fn dimension(&self) -> GeometryDimension;
    /// Whether `point` lies in the closed domain.
    fn contains(&self, point: &[f64]) -> bool;
    /// Classify `point` using a non-negative finite boundary tolerance.
    fn classify_point(&self, point: &[f64], tolerance: f64) -> PointLocation;
    /// Interleaved minimum/maximum coordinates of the axis-aligned enclosure.
    fn bounding_box(&self) -> Vec<f64>;
    /// Outward unit normal for a boundary point.
    fn normal(&self, point: &[f64], tolerance: f64) -> Option<Array1<f64>>;
    /// Length, area, or volume of the domain.
    fn measure(&self) -> f64;
    /// Map one normalized point in `[0, 1)^d` into the strict interior.
    ///
    /// The output is unchanged when validation fails.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] for a dimension mismatch or a non-finite or
    /// out-of-range normalized coordinate.
    fn map_unit_interior(&self, unit: &[f64], output: &mut [f64]) -> Result<(), GeometryError>;
    /// Draw reproducible, independent interior points.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] if the output shape exceeds addressable
    /// storage or exact output reservation fails.
    fn sample_interior(
        &self,
        sample_count: usize,
        seed: Seed,
    ) -> Result<Array2<f64>, GeometryError>;
    /// Draw reproducible points uniformly with respect to boundary measure.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] if the output shape exceeds addressable
    /// storage or exact output reservation fails.
    fn sample_boundary(
        &self,
        sample_count: usize,
        seed: Seed,
    ) -> Result<Array2<f64>, GeometryError>;
}
