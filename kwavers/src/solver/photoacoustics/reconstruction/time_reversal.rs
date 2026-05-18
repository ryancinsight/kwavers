use crate::core::error::KwaversResult;
use crate::domain::imaging::photoacoustic::{PhotoacousticScenario, PhotoacousticSignalSet};
use crate::solver::inverse::reconstruction::photoacoustic::{
    PhotoacousticAlgorithm, PhotoacousticReconstructor, ReconstructionPhotoacousticConfig,
};
use crate::solver::reconstruction::{ReconstructionConfig, Reconstructor};
use ndarray::Array3;

/// Time-reversal reconstruction path for general sensor geometries.
#[derive(Debug, Default)]
pub struct TimeReversalReconstruction;

impl TimeReversalReconstruction {
    /// Reconstruct.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn reconstruct(
        &self,
        scenario: &PhotoacousticScenario,
        signals: &PhotoacousticSignalSet,
    ) -> KwaversResult<Array3<f64>> {
        let config = ReconstructionPhotoacousticConfig {
            algorithm: PhotoacousticAlgorithm::TimeReversal,
            sensor_positions: signals.sensor_positions.clone(),
            grid_size: [scenario.grid.nx, scenario.grid.ny, scenario.grid.nz],
            sound_speed: scenario.config.acoustic.speed_of_sound_m_s,
            sampling_frequency: signals.sampling_frequency_hz.max(f64::MIN_POSITIVE),
            envelope_detection: false,
            bandpass_filter: None,
            regularization_parameter: 0.0,
        };
        let reconstructor = PhotoacousticReconstructor::new(config);
        reconstructor.reconstruct(
            &signals.sensor_data,
            &signals.sensor_positions,
            &scenario.grid,
            &ReconstructionConfig::default(),
        )
    }
}
