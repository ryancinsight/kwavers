//! Typed geometric-domain failures.

use thiserror::Error;

/// Failure to construct or sample a geometric domain.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GeometryError {
    /// An interval is non-finite, unordered, or has no representable interior.
    #[error(
        "axis {axis} requires finite ordered bounds with a representable interior, got [{min}, {max}]"
    )]
    InvalidBounds {
        /// Zero-based coordinate axis.
        axis: usize,
        /// Supplied lower bound.
        min: f64,
        /// Supplied upper bound.
        max: f64,
    },
    /// A spherical center coordinate is not finite.
    #[error("axis {axis} center coordinate must be finite, got {value}")]
    InvalidCenter {
        /// Zero-based coordinate axis.
        axis: usize,
        /// Supplied coordinate.
        value: f64,
    },
    /// A radius is non-finite, non-positive, or not representable at the center.
    #[error("radius must be finite, positive, and representable at every center coordinate, got {radius}")]
    InvalidRadius {
        /// Supplied radius.
        radius: f64,
    },
    /// A derived geometric measure is not finite and positive.
    #[error("{kind} measure must be finite and positive, got {value}")]
    InvalidMeasure {
        /// Measure role.
        kind: &'static str,
        /// Derived measure.
        value: f64,
    },
    /// A coordinate slice has the wrong dimension.
    #[error("{role} dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Coordinate role.
        role: &'static str,
        /// Domain dimension.
        expected: usize,
        /// Supplied dimension.
        actual: usize,
    },
    /// A normalized coordinate lies outside `[0, 1)` or is non-finite.
    #[error("unit coordinate {axis} must be finite and in [0, 1), got {value}")]
    InvalidUnitCoordinate {
        /// Zero-based coordinate axis.
        axis: usize,
        /// Supplied coordinate.
        value: f64,
    },
    /// A design reports an out-of-range sample while it is being collected.
    #[error("design rejected sample {index} within its declared count {sample_count}")]
    DesignSample {
        /// Rejected sample index.
        index: usize,
        /// Declared design length.
        sample_count: usize,
    },
    /// The requested point matrix element count overflows `usize`.
    #[error("{sample_count} samples in {dimensions} dimensions exceed addressable storage")]
    ElementCountOverflow {
        /// Requested number of rows.
        sample_count: usize,
        /// Requested number of columns.
        dimensions: usize,
    },
    /// The allocator could not reserve the validated output size.
    #[error("could not reserve storage for {element_count} point coordinates")]
    AllocationFailed {
        /// Validated element count passed to the allocator.
        element_count: usize,
    },
    /// A Tyche design count exceeds its validated 32-bit range.
    #[error("sample count {sample_count} exceeds Tyche's maximum {maximum}")]
    SampleCountExceedsLimit {
        /// Requested design length.
        sample_count: usize,
        /// Maximum supported design length.
        maximum: u32,
    },
}
