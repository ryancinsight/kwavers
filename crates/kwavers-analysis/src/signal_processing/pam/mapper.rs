use super::config::{PAMConfig, PamBeamformingConfig};
use super::processor::PAMProcessor;
use super::PamBeamformingMethod;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_domain::sensor::beamforming::processor::BeamformingProcessor;
use kwavers_domain::sensor::beamforming::BeamformingCoreConfig;
use kwavers_domain::sensor::passive_acoustic_mapping::geometry::PamArrayGeometry;
use ndarray::{Array3, Axis};

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

        let beamformed = match config.beamforming.method {
            PamBeamformingMethod::DelayAndSum => {
                let weights = vec![1.0; self.beamformer.num_sensors()];
                self.beamformer
                    .delay_and_sum_with(sensor_data, sample_rate, &delays, &weights)?
            }
            PamBeamformingMethod::CaponDiagonalLoading { diagonal_loading } => self
                .beamformer
                .mvdr_unsteered_weights_time_series(sensor_data, diagonal_loading)?,
            PamBeamformingMethod::Music { .. } => {
                return Err(KwaversError::InvalidInput(
                    "PAM beamforming: MUSIC is not yet wired to the shared subspace implementation. Use DelayAndSum or CaponDiagonalLoading for PAM mapping.".to_owned(),
                ));
            }
            PamBeamformingMethod::EigenspaceMinVariance { .. } => {
                return Err(KwaversError::InvalidInput(
                    "PAM beamforming: EigenspaceMinVariance is not yet wired to the shared subspace implementation. Use DelayAndSum or CaponDiagonalLoading for PAM mapping.".to_owned(),
                ));
            }
            PamBeamformingMethod::TimeExposureAcoustics => {
                let weights = vec![1.0; self.beamformer.num_sensors()];
                let das = self.beamformer.delay_and_sum_with(
                    sensor_data,
                    sample_rate,
                    &delays,
                    &weights,
                )?;

                let mut squared = das;
                squared.par_mapv_inplace(|x| x * x);

                let integrated = squared.sum_axis(Axis(2));
                let (nx, ny) = (integrated.shape()[0], integrated.shape()[1]);

                let mut tea = Array3::<f64>::zeros((nx, ny, 1));
                for ix in 0..nx {
                    for iy in 0..ny {
                        tea[[ix, iy, 0]] = integrated[[ix, iy]];
                    }
                }

                tea
            }
        };

        self.processor.process(&beamformed)
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
