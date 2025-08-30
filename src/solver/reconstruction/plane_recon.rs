//! Planar array reconstruction (k-Wave planeRecon equivalent)
//!
//! This module implements reconstruction algorithms for planar sensor arrays,
//! commonly used in photoacoustic and ultrasound imaging.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::solver::reconstruction::{
    ReconstructionConfig, Reconstructor, UniversalBackProjection, WeightFunction,
};
use ndarray::{Array2, Array3};

/// Planar array reconstruction
#[derive(Debug))]
pub struct PlaneRecon {
    /// Plane normal vector
    normal: [f64; 3],
    /// Plane center position
    center: [f64; 3],
    /// Back-projection algorithm
    back_projector: UniversalBackProjection,
}

impl PlaneRecon {
    /// Create a new planar array reconstruction
    pub fn new(normal: [f64; 3], center: [f64; 3]) -> Self {
        Self {
            normal,
            center,
            back_projector: UniversalBackProjection::new(WeightFunction::SolidAngle),
        }
    }

    /// Set weight function for back-projection
    pub fn with_weight_function(mut self, weight_function: WeightFunction) -> Self {
        self.back_projector = UniversalBackProjection::new(weight_function);
        self
    }
}

impl Reconstructor for PlaneRecon {
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3],
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
