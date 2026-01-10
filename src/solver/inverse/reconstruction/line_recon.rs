//! Linear array reconstruction
//!
//! This module implements reconstruction algorithms for linear sensor arrays,
//! commonly used in ultrasound imaging.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::solver::reconstruction::{
    ReconstructionConfig, Reconstructor, UniversalBackProjection, WeightFunction,
};
use ndarray::{Array2, Array3};

/// Linear array reconstruction
#[derive(Debug)]
pub struct LineRecon {
    /// Line direction vector
    #[allow(dead_code)]
    direction: [f64; 3],
    /// Line center position
    #[allow(dead_code)]
    center: [f64; 3],
    /// Array element pitch (spacing)
    #[allow(dead_code)]
    pitch: f64,
    /// Back-projection algorithm
    back_projector: UniversalBackProjection,
}

impl LineRecon {
    /// Create a new linear array reconstruction
    #[must_use]
    pub fn new(direction: [f64; 3], center: [f64; 3], pitch: f64) -> Self {
        Self {
            direction,
            center,
            pitch,
            back_projector: UniversalBackProjection::new(WeightFunction::Distance { power: 1.0 }),
        }
    }

    /// Set weight function for back-projection
    #[must_use]
    pub fn with_weight_function(mut self, weight_function: WeightFunction) -> Self {
        self.back_projector = UniversalBackProjection::new(weight_function);
        self
    }
}

impl Reconstructor for LineRecon {
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
        config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>> {
        // Use universal back-projection with linear geometry considerations
        self.back_projector
            .reconstruct(sensor_data, sensor_positions, grid, config)
    }

    fn name(&self) -> &str {
        "Linear Array Reconstruction (lineRecon)"
    }
}
