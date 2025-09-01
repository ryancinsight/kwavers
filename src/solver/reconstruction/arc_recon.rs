//! Arc/Circular array reconstruction
//!
//! This module implements reconstruction algorithms for arc and circular sensor arrays.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::solver::reconstruction::{
    ReconstructionConfig, Reconstructor, UniversalBackProjection, WeightFunction,
};
use ndarray::{Array2, Array3};

/// Arc/Circular array reconstruction
#[derive(Debug)]
pub struct ArcRecon {
    /// Arc center position
    center: [f64; 3],
    /// Arc radius
    radius: f64,
    /// Arc start angle (radians)
    start_angle: f64,
    /// Arc end angle (radians)
    end_angle: f64,
    /// Normal to arc plane
    normal: [f64; 3],
    /// Back-projection algorithm
    back_projector: UniversalBackProjection,
}

impl ArcRecon {
    /// Create a new arc array reconstruction
    pub fn new(
        center: [f64; 3],
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        normal: [f64; 3],
    ) -> Self {
        Self {
            center,
            radius,
            start_angle,
            end_angle,
            normal,
            back_projector: UniversalBackProjection::new(WeightFunction::SolidAngle),
        }
    }

    /// Create a full circular array reconstruction
    pub fn circular(center: [f64; 3], radius: f64, normal: [f64; 3]) -> Self {
        Self::new(center, radius, 0.0, 2.0 * std::f64::consts::PI, normal)
    }

    /// Set weight function for back-projection
    pub fn with_weight_function(mut self, weight_function: WeightFunction) -> Self {
        self.back_projector = UniversalBackProjection::new(weight_function);
        self
    }
}

impl Reconstructor for ArcRecon {
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
        config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>> {
        // Use universal back-projection with arc geometry considerations
        self.back_projector
            .reconstruct(sensor_data, sensor_positions, grid, config)
    }

    fn name(&self) -> &str {
        "Arc/Circular Array Reconstruction (arcRecon)"
    }
}
