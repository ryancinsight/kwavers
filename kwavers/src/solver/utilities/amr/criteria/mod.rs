//! Refinement criteria and error estimation

use crate::core::error::KwaversResult;
use ndarray::Array3;

mod methods;

/// Refinement criterion types
#[derive(Debug, Clone, Copy)]
pub enum RefinementCriterion {
    /// Gradient-based criterion
    Gradient,
    /// Curvature-based criterion
    Curvature,
    /// Richardson extrapolation
    Richardson,
    /// Wavelet-based criterion
    Wavelet,
    /// Physics-based (e.g., shock detection)
    Physics,
}

/// Error estimator for adaptive refinement
#[derive(Debug)]
pub struct ErrorEstimator {
    pub(super) criterion: RefinementCriterion,
    /// Smoothing parameter for noise reduction
    pub(super) smoothing: f64,
}

impl Default for ErrorEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorEstimator {
    /// Create a new error estimator
    #[must_use]
    pub fn new() -> Self {
        Self {
            criterion: RefinementCriterion::Gradient,
            smoothing: 0.1,
        }
    }

    /// Estimate error in the field
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn estimate_error(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        match self.criterion {
            RefinementCriterion::Gradient => self.gradient_error(field),
            RefinementCriterion::Curvature => self.curvature_error(field),
            RefinementCriterion::Richardson => self.richardson_error(field),
            RefinementCriterion::Wavelet => self.wavelet_error(field),
            RefinementCriterion::Physics => self.physics_error(field),
        }
    }
}
