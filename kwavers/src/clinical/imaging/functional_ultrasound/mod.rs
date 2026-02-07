//! Functional Ultrasound Brain GPS Module
//!
//! Implements automatic brain imaging registration and neuronavigation for functional
//! ultrasound (fUS) with real-time tracking and targeting capabilities.
//!
//! ## Architecture
//!
//! This module provides precision targeting for neuroscience research applications:
//! - Automatic affine registration to reference brain atlas
//! - Vascular-based localization (arterial/venous classification)
//! - Stereotactic coordinate system integration
//! - Real-time tracking with continuous refinement
//! - Safety-validated targeting
//!
//! ## Integration
//!
//! - **Domain Layer**: Imaging configuration, registration parameters, coordinates
//! - **Physics Layer**: Acoustic imaging models, tissue characterization
//! - **Clinical Layer**: Workflow orchestration, safety validation
//!
//! ## References
//!
//! - Macé, E., et al. (2018). "Functional ultrasound imaging of the brain"
//! - Tiran, E., et al. (2017). "Transcranial functional ultrasound imaging"
//! - Allen Brain Atlas: http://mouse.brain-map.org

pub mod atlas;
pub mod registration;
pub mod targeting;
pub mod tracking;
pub mod vasculature;

pub use atlas::BrainAtlas;
pub use targeting::{StereotacticCoordinates, TargetingSystem};
pub use tracking::TrackingFilter;
pub use vasculature::{VesselClassification, VesselSegmentation};

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use ndarray::Array3;

/// Affine transformation matrix (3×4) for image registration
/// Row-major: [R11 R12 R13 Tx; R21 R22 R23 Ty; R31 R32 R33 Tz]
#[derive(Debug, Clone)]
pub struct AffineTransform3D {
    /// 3×4 transformation matrix in row-major order
    pub matrix: [[f64; 4]; 3],
}

impl AffineTransform3D {
    /// Create identity transformation
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
    }

    /// Transform a point
    pub fn transform_point(&self, p: &[f64; 3]) -> [f64; 3] {
        [
            self.matrix[0][0] * p[0]
                + self.matrix[0][1] * p[1]
                + self.matrix[0][2] * p[2]
                + self.matrix[0][3],
            self.matrix[1][0] * p[0]
                + self.matrix[1][1] * p[1]
                + self.matrix[1][2] * p[2]
                + self.matrix[1][3],
            self.matrix[2][0] * p[0]
                + self.matrix[2][1] * p[1]
                + self.matrix[2][2] * p[2]
                + self.matrix[2][3],
        ]
    }
}

/// Registration configuration parameters
#[derive(Debug, Clone)]
pub struct RegistrationConfig {
    /// Number of spatial samples for MI computation
    pub num_samples: usize,
    /// Number of histogram bins for MI metric
    pub num_bins: usize,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
}

impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            num_samples: 5000,
            num_bins: 50,
            max_iterations: 200,
            tolerance: 1e-6,
        }
    }
}

/// Registration engine for affine alignment to atlas
#[derive(Debug)]
pub struct RegistrationEngine {
    #[allow(dead_code)]
    config: RegistrationConfig,
}

impl RegistrationEngine {
    /// Create new registration engine
    pub fn new() -> KwaversResult<Self> {
        Ok(Self {
            config: RegistrationConfig::default(),
        })
    }

    /// Register moving image to fixed reference image
    ///
    /// # Errors
    /// Returns `KwaversError::NotImplemented` — Mattes MI registration pending.
    pub fn register(
        &self,
        _moving: &Array3<f64>,
        _fixed: &Array3<f64>,
        _config: &RegistrationConfig,
    ) -> KwaversResult<AffineTransform3D> {
        Err(KwaversError::NotImplemented(
            "Affine registration via Mattes mutual information metric \
             and CMA-ES optimization not yet implemented."
                .into(),
        ))
    }
}

/// Functional ultrasound brain GPS orchestrator
///
/// Combines registration, localization, and targeting into unified workflow
#[derive(Debug)]
pub struct FunctionalUltrasoundGPS {
    /// Registration engine for atlas alignment
    registration: RegistrationEngine,

    /// Vessel segmentation and classification
    vasculature: Option<VesselSegmentation>,

    /// Brain atlas reference
    atlas: BrainAtlas,

    /// Stereotactic targeting system
    targeting: TargetingSystem,

    /// Continuous tracking filter
    tracking: TrackingFilter,

    /// Current imaging grid
    #[allow(dead_code)]
    grid: Grid,
}

impl FunctionalUltrasoundGPS {
    /// Create new brain GPS system
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        let registration = RegistrationEngine::new()?;
        let atlas = BrainAtlas::load_default()?;
        let targeting = TargetingSystem::new(&atlas)?;
        let tracking = TrackingFilter::new()?;

        Ok(Self {
            registration,
            vasculature: None,
            atlas,
            targeting,
            tracking,
            grid: grid.clone(),
        })
    }

    /// Register ultrasound image to brain atlas
    pub fn register_to_atlas(&mut self, image: &Array3<f64>) -> KwaversResult<AffineTransform3D> {
        let config = RegistrationConfig::default();
        self.registration
            .register(image, &self.atlas.reference_image(), &config)
    }

    /// Segment vasculature from registered image
    pub fn segment_vasculature(&mut self, registered_image: &Array3<f64>) -> KwaversResult<()> {
        let segmentation = VesselSegmentation::segment(registered_image)?;
        self.vasculature = Some(segmentation);
        Ok(())
    }

    /// Localize target in stereotactic coordinates
    pub fn locate_target(
        &self,
        voxel_coords: &[usize; 3],
    ) -> KwaversResult<StereotacticCoordinates> {
        self.targeting
            .voxel_to_stereotactic(voxel_coords, &self.atlas)
    }

    /// Update tracking with new measurement
    pub fn update_tracking(&mut self, measurement: &[f64; 3]) -> KwaversResult<[f64; 3]> {
        self.tracking.update(measurement)
    }

    /// Get current tracked position
    pub fn get_tracked_position(&self) -> [f64; 3] {
        self.tracking.get_position()
    }

    /// Get vessel classification
    pub fn get_vessel_classification(&self) -> Option<&VesselClassification> {
        self.vasculature.as_ref().map(|v| &v.classification)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_functional_ultrasound_gps_creation() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
        let result = FunctionalUltrasoundGPS::new(&grid);
        assert!(result.is_ok());
    }
}
