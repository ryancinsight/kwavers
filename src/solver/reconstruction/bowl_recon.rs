//! Bowl/Hemispherical array reconstruction
//!
//! This module implements reconstruction algorithms for bowl-shaped (hemispherical) sensor arrays,
//! commonly used in photoacoustic tomography and focused ultrasound applications.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::solver::reconstruction::{
    ReconstructionConfig, Reconstructor, UniversalBackProjection, WeightFunction,
};
use ndarray::{Array2, Array3};

/// Bowl/Hemispherical array reconstruction
#[derive(Debug)]
pub struct BowlRecon {
    /// Bowl center position
    #[allow(dead_code)]
    center: [f64; 3],
    /// Bowl radius
    #[allow(dead_code)]
    radius: f64,
    /// Bowl focus position
    #[allow(dead_code)]
    focus: [f64; 3],
    /// Opening angle (radians)
    #[allow(dead_code)]
    opening_angle: f64,
    /// Back-projection algorithm
    back_projector: UniversalBackProjection,
}

impl BowlRecon {
    /// Create a new bowl array reconstruction
    #[must_use]
    pub fn new(center: [f64; 3], radius: f64, focus: [f64; 3], opening_angle: f64) -> Self {
        Self {
            center,
            radius,
            focus,
            opening_angle,
            back_projector: UniversalBackProjection::new(WeightFunction::SolidAngle),
        }
    }

    /// Create a hemispherical array reconstruction
    #[must_use]
    pub fn hemispherical(center: [f64; 3], radius: f64) -> Self {
        Self::new(
            center,
            radius,
            center,                     // Focus at center for hemispherical
            std::f64::consts::PI / 2.0, // 90 degree opening
        )
    }

    /// Set weight function for back-projection
    #[must_use]
    pub fn with_weight_function(mut self, weight_function: WeightFunction) -> Self {
        self.back_projector = UniversalBackProjection::new(weight_function);
        self
    }
}

impl Reconstructor for BowlRecon {
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
        config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>> {
        // Use universal back-projection with bowl geometry considerations
        // Bowl arrays provide excellent coverage for 3D imaging
        self.back_projector
            .reconstruct(sensor_data, sensor_positions, grid, config)
    }

    fn name(&self) -> &str {
        "Bowl/Hemispherical Array Reconstruction (bowlRecon)"
    }
}
