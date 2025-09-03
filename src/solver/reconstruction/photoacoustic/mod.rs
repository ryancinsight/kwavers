//! Photoacoustic Reconstruction Module
//!
//! This module provides comprehensive photoacoustic imaging reconstruction algorithms
//! equivalent to k-Wave's photoacoustic reconstruction capabilities.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for each reconstruction algorithm
//! - **DRY**: Reusable reconstruction components
//! - **Zero-Copy**: Uses `ArrayView` for efficient data handling
//! - **KISS**: Clear interfaces for complex reconstruction algorithms
//!
//! # Literature References
//! - Xu & Wang (2005): "Universal back-projection algorithm for photoacoustic computed tomography"
//! - Burgholzer et al. (2007): "Exact and approximate imaging methods for photoacoustic tomography"
//! - Treeby & Cox (2010): "k-Wave: MATLAB toolbox"
//! - Wang & Yao (2016): "Photoacoustic tomography: in vivo imaging from organelles to organs"

mod algorithms;
mod config;
mod filters;
mod fourier;
mod iterative;
mod linear_algebra;
mod time_reversal;
mod utils;

pub use algorithms::{PhotoacousticAlgorithm, PhotoacousticReconstructor};
pub use config::PhotoacousticConfig;
pub use iterative::IterativeAlgorithm;

use crate::error::KwaversResult;
use crate::solver::reconstruction::Reconstructor;
use ndarray::{Array2, Array3};

impl Reconstructor for PhotoacousticReconstructor {
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        _grid: &crate::grid::Grid,
        _config: &crate::solver::reconstruction::ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>> {
        // Dispatch to appropriate algorithm
        match &self.config.algorithm {
            PhotoacousticAlgorithm::UniversalBackProjection => self.universal_back_projection(
                sensor_data.view(),
                sensor_positions,
                self.config.grid_size,
                self.config.sound_speed,
                self.config.sampling_frequency,
            ),
            PhotoacousticAlgorithm::FilteredBackProjection => {
                self.filtered_back_projection(sensor_data.view(), sensor_positions)
            }
            PhotoacousticAlgorithm::TimeReversal => {
                self.time_reversal_reconstruction(sensor_data.view(), sensor_positions, _grid)
            }
            PhotoacousticAlgorithm::FourierDomain => {
                self.fourier_domain_reconstruction(sensor_data.view(), sensor_positions)
            }
            PhotoacousticAlgorithm::Iterative { .. } => self.iterative_reconstruction(
                sensor_data.view(),
                sensor_positions,
                self.config.grid_size,
            ),
            PhotoacousticAlgorithm::ModelBased => {
                self.model_based_reconstruction(sensor_data.view(), sensor_positions)
            }
        }
    }

    fn name(&self) -> &str {
        "PhotoacousticReconstructor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_photoacoustic_reconstructor_creation() {
        let config = PhotoacousticConfig {
            algorithm: PhotoacousticAlgorithm::UniversalBackProjection,
            sensor_positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            grid_size: [100, 100, 100],
            sound_speed: 1500.0,
            sampling_frequency: 10e6,
            envelope_detection: false,
            bandpass_filter: None,
            regularization_parameter: 0.0,
        };

        let _reconstructor = PhotoacousticReconstructor::new(config);
    }

    #[test]
    fn test_universal_back_projection() {
        let config = PhotoacousticConfig {
            algorithm: PhotoacousticAlgorithm::UniversalBackProjection,
            sensor_positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            grid_size: [10, 10, 10],
            sound_speed: 1500.0,
            sampling_frequency: 10e6,
            envelope_detection: false,
            bandpass_filter: None,
            regularization_parameter: 0.0,
        };

        let reconstructor = PhotoacousticReconstructor::new(config);
        let sensor_data = Array2::zeros((100, 2));
        let result = reconstructor.universal_back_projection(
            sensor_data.view(),
            &reconstructor.config.sensor_positions,
            [10, 10, 10],
            1500.0,
            10e6,
        );
        assert!(result.is_ok());
    }
}
