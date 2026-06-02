//! Planar array reconstruction
//!
//! This module implements reconstruction algorithms for planar sensor arrays,
//! commonly used in photoacoustic and ultrasound imaging.

use kwavers_core::error::KwaversResult;
use kwavers_domain::grid::Grid;
use crate::reconstruction::{
    ReconstructionConfig, Reconstructor, UniversalBackProjection, WeightFunction,
};
use ndarray::{Array2, Array3};

/// Planar array reconstruction
#[derive(Debug)]
pub struct PlaneRecon {
    /// Back-projection algorithm
    back_projector: UniversalBackProjection,
}

impl PlaneRecon {
    /// Create a new planar array reconstruction
    #[must_use]
    pub fn new(_normal: [f64; 3], _center: [f64; 3]) -> Self {
        Self {
            back_projector: UniversalBackProjection::new(WeightFunction::SolidAngle),
        }
    }

    /// Set weight function for back-projection
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_weight_function(mut self, weight_function: WeightFunction) -> Self {
        self.back_projector = UniversalBackProjection::new(weight_function);
        self
    }
}

impl Reconstructor for PlaneRecon {
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
        config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>> {
        // Use universal back-projection with planar geometry considerations
        self.back_projector
            .reconstruct(sensor_data, sensor_positions, grid, config)
    }

    fn name(&self) -> &str {
        "Planar Array Reconstruction (planeRecon)"
    }
}
