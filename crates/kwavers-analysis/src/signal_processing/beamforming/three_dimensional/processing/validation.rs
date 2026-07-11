use super::super::processor::BeamformingProcessor3D;
#[cfg(feature = "gpu")]
use super::super::provider::BeamformingGpuProvider;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array4;

#[cfg(feature = "gpu")]
impl<P> BeamformingProcessor3D<P>
where
    P: BeamformingGpuProvider,
{
    /// Validate input RF data dimensions (GPU path).
    ///
    /// Invariants checked:
    /// - `rf_data` is non-empty.
    /// - `channels == nel_x × nel_y × nel_z`.
    /// - `samples ≥ 1`.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub(super) fn validate_input(&self, rf_data: &Array4<f32>) -> KwaversResult<()> {
        let rf_dims = rf_data.shape();
        let _frames = rf_dims[0];
        let channels = rf_dims[1];
        let samples = rf_dims[2];

        if rf_data.is_empty() {
            return Err(KwaversError::InvalidInput(
                "RF data array is empty".to_string(),
            ));
        }

        let expected_channels = self.config.num_elements_3d.0
            * self.config.num_elements_3d.1
            * self.config.num_elements_3d.2;

        if channels != expected_channels {
            return Err(KwaversError::InvalidInput(format!(
                "Channel count mismatch: expected {}, got {}",
                expected_channels, channels
            )));
        }

        if samples == 0 {
            return Err(KwaversError::InvalidInput(
                "RF data must contain at least one sample per channel".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(not(feature = "gpu"))]
impl BeamformingProcessor3D {
    /// Validate input RF data dimensions (CPU path — identical contract to GPU version).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub(super) fn validate_input(&self, rf_data: &Array4<f32>) -> KwaversResult<()> {
        if rf_data.is_empty() {
            return Err(KwaversError::InvalidInput(
                "RF data array is empty".to_owned(),
            ));
        }
        let [_, channels, samples, _] = rf_data.shape();
        let expected_channels = self.config.num_elements_3d.0
            * self.config.num_elements_3d.1
            * self.config.num_elements_3d.2;
        if channels != expected_channels {
            return Err(KwaversError::InvalidInput(format!(
                "3D beamforming: channel count {channels} ≠ element count \
                 {expected_channels} ({}×{}×{})",
                self.config.num_elements_3d.0,
                self.config.num_elements_3d.1,
                self.config.num_elements_3d.2,
            )));
        }
        if samples == 0 {
            return Err(KwaversError::InvalidInput(
                "RF data must contain at least one sample per channel".to_owned(),
            ));
        }
        Ok(())
    }
}
