use super::config::{PAMConfig, PamBeamformingConfig};
use super::processor::PAMProcessor;
use super::PamBeamformingMethod;
use crate::signal_processing::beamforming::narrowband::{
    subspace_spatial_spectrum_point, SubspaceMethod, SubspaceSpectrumConfig,
};
use kwavers_core::error::KwaversResult;
use kwavers_transducer::beamforming::processor::BeamformingProcessor;
use kwavers_transducer::beamforming::BeamformingCoreConfig;
use kwavers_transducer::passive_acoustic_mapping::geometry::PamArrayGeometry;
use leto::Array3;

#[derive(Debug)]
pub struct PassiveAcousticMapper {
    pub(super) processor: PAMProcessor,
    pub(super) beamformer: BeamformingProcessor,
}

impl PassiveAcousticMapper {
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: PAMConfig, geometry: PamArrayGeometry) -> KwaversResult<Self> {
        config.beamforming.validate()?;

        let element_positions = geometry.element_positions();
        let core_cfg: BeamformingCoreConfig = config.beamforming.clone().into();
        let beamformer = BeamformingProcessor::new(core_cfg, element_positions);
        let processor = PAMProcessor::new(config)?;

        Ok(Self {
            processor,
            beamformer,
        })
    }
    /// Process.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn process(
        &mut self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
    ) -> KwaversResult<Array3<f64>> {
        let config = self.processor.config();

        config.beamforming.validate()?;

        let delays = self
            .beamformer
            .compute_delays(config.beamforming.focal_point);
        let sensor_data_leto = sensor_data.clone();

        let beamformed = match config.beamforming.method {
            PamBeamformingMethod::DelayAndSum => {
                let weights = vec![1.0; self.beamformer.num_sensors()];
                self.beamformer.delay_and_sum_with(
                    &sensor_data_leto,
                    sample_rate,
                    &delays,
                    &weights,
                )?
            }
            PamBeamformingMethod::CaponDiagonalLoading { diagonal_loading } => self
                .beamformer
                .mvdr_unsteered_weights_time_series(&sensor_data_leto, diagonal_loading)?,
            // Subspace localizers (Theorem 22.2) produce a per-focal-point
            // localization power, not a beamformed time series, so they bypass the
            // time-series cavitation-spectrum processor and return the map directly.
            PamBeamformingMethod::Music { num_sources } => {
                return self.subspace_localization_map(
                    sensor_data,
                    sample_rate,
                    SubspaceMethod::Music,
                    num_sources,
                );
            }
            PamBeamformingMethod::EigenspaceMinVariance {
                signal_subspace_dimension,
            } => {
                return self.subspace_localization_map(
                    sensor_data,
                    sample_rate,
                    SubspaceMethod::EigenspaceMv,
                    signal_subspace_dimension,
                );
            }
            PamBeamformingMethod::TimeExposureAcoustics => {
                let weights = vec![1.0; self.beamformer.num_sensors()];
                let das = self.beamformer.delay_and_sum_with(
                    &sensor_data_leto,
                    sample_rate,
                    &delays,
                    &weights,
                )?;
                let [nx, ny, nt] = das.shape();

                let mut tea = Array3::<f64>::zeros((nx, ny, 1));
                for ix in 0..nx {
                    for iy in 0..ny {
                        let mut integrated = 0.0;
                        for it in 0..nt {
                            let v = das[[ix, iy, it]];
                            integrated += v * v;
                        }
                        tea[[ix, iy, 0]] = integrated;
                    }
                }

                tea
            }
        };

        self.processor.process(&beamformed)
    }

    /// Evaluate a narrowband subspace localization map (Eigenspace-MV or MUSIC) at
    /// the configured focal point (PAM Theorem 22.2).
    ///
    /// Returns a `(1, 1, 1)` map holding the localization power at the focus — high
    /// when a cavitation source sits there, low otherwise. The cross-spectral
    /// matrix, eigendecomposition, and steering reuse the shared narrowband
    /// `subspace_spectrum` code (no duplication). The emission frequency is the
    /// centre of the configured `frequency_range`; the sampling rate is the rate of
    /// the supplied `sensor_data`.
    ///
    /// # Errors
    /// - Propagates snapshot-extraction, covariance, steering, and eigendecomposition errors.
    fn subspace_localization_map(
        &self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
        method: SubspaceMethod,
        num_sources: usize,
    ) -> KwaversResult<Array3<f64>> {
        let config = self.processor.config();
        let core = &config.beamforming.core;
        let (f_min, f_max) = config.beamforming.frequency_range;

        let cfg = SubspaceSpectrumConfig {
            frequency_hz: 0.5 * (f_min + f_max),
            sampling_frequency_hz: sample_rate,
            sound_speed: core.sound_speed,
            num_sources,
            diagonal_loading: core.diagonal_loading,
        };

        let positions = self.beamformer.sensor_positions().to_vec();
        let power = subspace_spatial_spectrum_point(
            method,
            sensor_data,
            &positions,
            config.beamforming.focal_point,
            &cfg,
        )?;

        let mut map = Array3::<f64>::zeros((1, 1, 1));
        map[[0, 0, 0]] = power;
        Ok(map)
    }
    /// Config.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn config(&self) -> &PAMConfig {
        self.processor.config()
    }
    /// Set config.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn set_config(&mut self, config: PAMConfig) -> KwaversResult<()> {
        config.beamforming.validate()?;

        let element_positions: Vec<[f64; 3]> = self.beamformer.sensor_positions().to_vec();
        let core_cfg: BeamformingCoreConfig = config.beamforming.clone().into();
        self.beamformer = BeamformingProcessor::new(core_cfg, element_positions);

        self.processor.set_config(config)?;
        Ok(())
    }
}

impl From<PamBeamformingConfig> for BeamformingCoreConfig {
    fn from(pam: PamBeamformingConfig) -> Self {
        let (f_min, f_max) = pam.frequency_range;
        let reference_frequency = 0.5 * (f_min + f_max);

        let mut core = pam.core;
        core.reference_frequency = reference_frequency;

        if let PamBeamformingMethod::CaponDiagonalLoading { diagonal_loading } = pam.method {
            core.diagonal_loading = diagonal_loading;
        }

        core
    }
}
