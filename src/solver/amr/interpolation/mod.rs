// solver/amr/interpolation/mod.rs - Modular interpolation for AMR

pub mod coefficients;
pub mod operations;
pub mod schemes;
pub mod spectral;
pub mod validation;

// Re-export main types
pub use coefficients::InterpolationCoefficients;
pub use operations::{InterpolationOperator, Interpolator};
pub use schemes::{InterpolationScheme, InterpolationType};
pub use spectral::SpectralInterpolator;
pub use validation::InterpolationValidator;

// Compatibility functions
use crate::error::KwaversResult;
use ndarray::Array3;

/// Interpolate field to refined grid (compatibility wrapper)
pub fn interpolate_to_refined(
    field: &Array3<f64>,
    _octree: &crate::solver::amr::octree::Octree,
    scheme: crate::solver::amr::InterpolationScheme,
) -> KwaversResult<Array3<f64>> {
    let interpolation_type = match scheme {
        crate::solver::amr::InterpolationScheme::Linear => InterpolationType::Linear,
        crate::solver::amr::InterpolationScheme::Conservative => InterpolationType::Quadratic,
        crate::solver::amr::InterpolationScheme::WENO5 => InterpolationType::Cubic,
        crate::solver::amr::InterpolationScheme::Spectral => InterpolationType::Spectral,
    };
    let interpolator = Interpolator::new(interpolation_type);
    // Default refinement ratio - would be determined from octree in production
    interpolator.interpolate(field, 2)
}

/// Restrict field to coarse grid (compatibility wrapper)
pub fn restrict_to_coarse(
    field: &Array3<f64>,
    _octree: &crate::solver::amr::octree::Octree,
    scheme: crate::solver::amr::InterpolationScheme,
) -> KwaversResult<Array3<f64>> {
    let interpolation_type = match scheme {
        crate::solver::amr::InterpolationScheme::Linear => InterpolationType::Linear,
        crate::solver::amr::InterpolationScheme::Conservative => InterpolationType::Quadratic,
        crate::solver::amr::InterpolationScheme::WENO5 => InterpolationType::Cubic,
        crate::solver::amr::InterpolationScheme::Spectral => InterpolationType::Spectral,
    };
    let interpolator = Interpolator::new(interpolation_type);
    // Default refinement ratio - would be determined from octree in production
    interpolator.restrict(field, 2)
}
