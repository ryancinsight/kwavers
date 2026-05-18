//! Interpolation schemes for inter-domain coupling

mod conservative;
mod cubic;
mod linear;

use crate::core::error::KwaversResult;
use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// Interpolation schemes for inter-domain coupling
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum HybridInterpolationScheme {
    /// Linear interpolation (2nd order)
    Linear,
    /// Cubic spline interpolation (4th order)
    #[default]
    CubicSpline,
    /// Spectral interpolation (machine precision)
    Spectral,
    /// Conservative interpolation (preserves integrals)
    Conservative,
    /// Adaptive interpolation (switches based on local conditions)
    Adaptive,
}

/// Manager for interpolation operations
#[derive(Debug)]
pub struct InterpolationManager {
    pub(super) scheme: HybridInterpolationScheme,
}

impl InterpolationManager {
    /// Create a new interpolation manager
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(scheme: HybridInterpolationScheme) -> Self {
        Self { scheme }
    }

    /// Interpolate fields from source to target grid
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn interpolate(
        &self,
        source_field: &Array3<f64>,
        source_coords: &[(f64, f64, f64)],
        target_coords: &[(f64, f64, f64)],
    ) -> KwaversResult<Array3<f64>> {
        match self.scheme {
            HybridInterpolationScheme::Linear => {
                self.linear_interpolation(source_field, source_coords, target_coords)
            }
            HybridInterpolationScheme::CubicSpline => {
                self.cubic_spline_interpolation(source_field, source_coords, target_coords)
            }
            HybridInterpolationScheme::Spectral => {
                self.spectral_interpolation(source_field, source_coords, target_coords)
            }
            HybridInterpolationScheme::Conservative => {
                self.conservative_interpolation(source_field, source_coords, target_coords)
            }
            HybridInterpolationScheme::Adaptive => {
                self.adaptive_interpolation(source_field, source_coords, target_coords)
            }
        }
    }
}
