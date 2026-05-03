//! RTM struct definition, constructor, and Reconstructor trait implementation.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::solver::reconstruction::{ReconstructionConfig, Reconstructor};
use ndarray::{Array2, Array3};

use super::super::config::SeismicImagingConfig;

/// Reverse Time Migration reconstructor
/// Creates subsurface images by cross-correlating forward and backward wavefields
#[derive(Debug)]
pub struct ReverseTimeMigration {
    pub(super) config: SeismicImagingConfig,
    /// Velocity model for migration
    pub(super) velocity_model: Array3<f64>,
    /// Final migrated image
    pub(super) image: Array3<f64>,
    /// Source illumination for normalization
    pub(super) source_illumination: Array3<f64>,
}

impl ReverseTimeMigration {
    /// Create new RTM reconstructor
    #[must_use]
    pub fn new(config: SeismicImagingConfig, velocity_model: Array3<f64>) -> Self {
        let image = Array3::zeros(velocity_model.dim());
        let source_illumination = Array3::zeros(velocity_model.dim());

        Self {
            config,
            velocity_model,
            image,
            source_illumination,
        }
    }

    /// Get the migrated image
    #[must_use]
    pub fn get_image(&self) -> &Array3<f64> {
        &self.image
    }
}

impl Reconstructor for ReverseTimeMigration {
    fn name(&self) -> &str {
        "ReverseTimeMigration"
    }

    fn reconstruct(
        &self,
        _sensor_data: &Array2<f64>,
        _sensor_positions: &[[f64; 3]],
        _grid: &Grid,
        _config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>> {
        // RTM requires mutable state for migration
        // Using clone() is acceptable here as migration results are immutable once computed
        // This follows the functional programming principle of immutability
        Ok(self.image.clone())
    }
}
