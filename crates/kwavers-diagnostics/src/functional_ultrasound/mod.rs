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
//! - Allen Brain Atlas: <http://mouse.brain-map.org>

pub mod atlas;
pub mod targeting;
pub mod tracking;

// ULM super-resolution and vessel-filtering algorithms moved to the `analysis`
// layer (`kwavers_analysis::signal_processing::{ulm,vasculature}`); the
// neuronavigation workflow imports the vessel types it needs from there.
use kwavers_analysis::signal_processing::vasculature::{VesselClassification, VesselSegmentation};

pub use atlas::BrainAtlas;
pub use targeting::{StereotacticCoordinates, TargetingSystem};
pub use tracking::TrackingFilter;

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_domain::grid::Grid;
use ndarray::Array3;
use ritk_registration::ImageRegistration;

/// Affine transformation matrix (3×4) for image registration
/// Row-major: [R11 R12 R13 Tx; R21 R22 R23 Ty; R31 R32 R33 Tz]
#[derive(Debug, Clone)]
pub struct AffineTransform3D {
    /// 3×4 transformation matrix in row-major order
    pub matrix: [[f64; 4]; 3],
}

impl AffineTransform3D {
    /// Create identity transformation
    #[must_use]
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
    #[must_use]
    pub fn transform_point(&self, p: &[f64; 3]) -> [f64; 3] {
        [
            self.matrix[0][2].mul_add(
                p[2],
                self.matrix[0][0].mul_add(p[0], self.matrix[0][1] * p[1]),
            ) + self.matrix[0][3],
            self.matrix[1][2].mul_add(
                p[2],
                self.matrix[1][0].mul_add(p[0], self.matrix[1][1] * p[1]),
            ) + self.matrix[1][3],
            self.matrix[2][2].mul_add(
                p[2],
                self.matrix[2][0].mul_add(p[0], self.matrix[2][1] * p[1]),
            ) + self.matrix[2][3],
        ]
    }
}

/// Functional ultrasound brain GPS orchestrator
///
/// Combines registration, localization, and targeting into unified workflow
#[derive(Debug)]
pub struct FunctionalUltrasoundGPS {
    /// Registration engine for atlas alignment managed directly via ritk
    registration: ImageRegistration,

    /// Vessel segmentation and classification
    vasculature: Option<VesselSegmentation>,

    /// Brain atlas reference
    atlas: BrainAtlas,

    /// Stereotactic targeting system
    targeting: TargetingSystem,

    /// Continuous tracking filter
    tracking: TrackingFilter,
}

impl FunctionalUltrasoundGPS {
    /// Create new brain GPS system
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(_grid: &Grid) -> KwaversResult<Self> {
        let registration = ImageRegistration::default();
        let atlas = BrainAtlas::load_default()?;
        let targeting = TargetingSystem::new(&atlas)?;
        let tracking = TrackingFilter::new()?;

        Ok(Self {
            registration,
            vasculature: None,
            atlas,
            targeting,
            tracking,
        })
    }

    /// Register ultrasound image to brain atlas
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn register_to_atlas(&mut self, image: &Array3<f64>) -> KwaversResult<AffineTransform3D> {
        let initial_transform = [
            1f64, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        let result = self
            .registration
            .affine_registration_mutual_info(
                image,
                self.atlas.reference_image_ref(),
                &initial_transform,
            )
            .map_err(|e| {
                KwaversError::InvalidInput(format!("RITK Atlas registration failed: {:?}", e))
            })?;

        let m = result.transform;
        Ok(AffineTransform3D {
            matrix: [
                [m[0], m[1], m[2], m[3]],
                [m[4], m[5], m[6], m[7]],
                [m[8], m[9], m[10], m[11]],
            ],
        })
    }

    /// Segment vasculature from registered image
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn segment_vasculature(&mut self, registered_image: &Array3<f64>) -> KwaversResult<()> {
        let segmentation = VesselSegmentation::segment(registered_image)?;
        self.vasculature = Some(segmentation);
        Ok(())
    }

    /// Localize target in stereotactic coordinates
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn locate_target(
        &self,
        voxel_coords: &[usize; 3],
    ) -> KwaversResult<StereotacticCoordinates> {
        self.targeting
            .voxel_to_stereotactic(voxel_coords, &self.atlas)
    }

    /// Update tracking with new measurement
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn update_tracking(&mut self, measurement: &[f64; 3]) -> KwaversResult<[f64; 3]> {
        self.tracking.update(measurement)
    }

    /// Get current tracked position
    #[must_use]
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
        let _gps = FunctionalUltrasoundGPS::new(&grid).unwrap();
    }
}
