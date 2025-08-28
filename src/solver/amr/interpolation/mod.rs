// solver/amr/interpolation/mod.rs - Unified interpolation for AMR

pub mod coefficients;
pub mod operations;
pub mod spectral;
pub mod validation;

use crate::error::KwaversResult;
use crate::solver::amr::octree::Octree;
use ndarray::Array3;

// Single unified interpolation implementation
pub use coefficients::InterpolationCoefficients;
pub use operations::{InterpolationOperator, Interpolator};
pub use spectral::SpectralInterpolator;
pub use validation::InterpolationValidator;

/// Unified interpolation scheme - single source of truth
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationScheme {
    Linear,
    Conservative,
    WENO5,
    Spectral,
}

impl InterpolationScheme {
    /// Get the order of accuracy
    pub fn order(&self) -> usize {
        match self {
            Self::Linear => 1,
            Self::Conservative => 2,
            Self::WENO5 => 5,
            Self::Spectral => 8,
        }
    }

    /// Check if conservation is preserved
    pub fn preserves_conservation(&self) -> bool {
        matches!(self, Self::Conservative | Self::WENO5)
    }
}

/// Interpolate field to refined grid - single implementation
pub fn interpolate_to_refined(
    field: &Array3<f64>,
    octree: &Octree,
    scheme: InterpolationScheme,
) -> KwaversResult<Array3<f64>> {
    let interpolator = Interpolator::from_scheme(scheme, octree);
    interpolator.interpolate(field, octree.refinement_ratio())
}

/// Restrict field to coarse grid - single implementation
pub fn restrict_to_coarse(
    field: &Array3<f64>,
    octree: &Octree,
    scheme: InterpolationScheme,
) -> KwaversResult<Array3<f64>> {
    let interpolator = Interpolator::from_scheme(scheme, octree);
    interpolator.restrict(field, octree.refinement_ratio())
}
