use super::super::config::ApodizationWindow;
use super::super::processor::BeamformingProcessor3D;
use crate::core::error::KwaversResult;
use ndarray::{Array3, Array4};

impl BeamformingProcessor3D {
    /// Process delay-and-sum beamforming (GPU path).
    ///
    /// Coherent summation of delayed RF signals across all elements to form a focused image.
    /// Optional sub-volume processing for memory-constrained systems.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[cfg(feature = "gpu")]
    pub(super) fn process_delay_and_sum(
        &self,
        rf_data: &Array4<f32>,
        dynamic_focusing: bool,
        apodization: &ApodizationWindow,
        sub_volume_size: Option<(usize, usize, usize)>,
    ) -> KwaversResult<Array3<f32>> {
        let apodization_weights = self.create_apodization_weights(apodization);

        if let Some(sub_size) = sub_volume_size {
            self.delay_and_sum_subvolume_gpu(
                rf_data,
                dynamic_focusing,
                apodization,
                &apodization_weights,
                sub_size,
            )
        } else {
            self.delay_and_sum_gpu(rf_data, dynamic_focusing, apodization, &apodization_weights)
        }
    }

    /// CPU delay-and-sum dispatch.
    ///
    /// Forwards to [`super::super::cpu::delay_and_sum_cpu`].  The `dynamic_focusing`
    /// and `sub_volume_size` flags are accepted for API symmetry with the GPU path;
    /// the CPU kernel processes the whole volume in a single Rayon-parallel pass.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[cfg(not(feature = "gpu"))]
    pub(super) fn process_delay_and_sum(
        &self,
        rf_data: &Array4<f32>,
        _dynamic_focusing: bool,
        apodization: &ApodizationWindow,
        _sub_volume_size: Option<(usize, usize, usize)>,
    ) -> KwaversResult<Array3<f32>> {
        super::super::cpu::delay_and_sum_cpu(rf_data, &self.config, apodization)
    }

    /// Process MVDR 3D beamforming (GPU path — stub).
    ///
    /// # References
    /// - Van Veen & Buckley (1988) "Beamforming: A versatile approach to spatial filtering"
    /// - Synnevåg et al. (2009) "Adaptive beamforming applied to medical ultrasound imaging"
    /// # Errors
    /// - Returns [`KwaversError::System`] always (GPU MVDR not yet implemented).
    ///
    #[cfg(feature = "gpu")]
    pub(super) fn process_mvdr_3d(
        &mut self,
        rf_data: &Array4<f32>,
        diagonal_loading: f32,
        subarray_size: [usize; 3],
    ) -> KwaversResult<Array3<f32>> {
        let _ = (rf_data, diagonal_loading, subarray_size);
        Err(KwaversError::System(
            crate::core::error::SystemError::FeatureNotAvailable {
                feature: "mvdr-3d".to_string(),
                reason: "MVDR 3D reconstruction is not implemented".to_string(),
            },
        ))
    }

    /// CPU MVDR-3D adaptive beamforming.
    ///
    /// Forwards to [`super::super::cpu::mvdr_cpu`] which implements spatially-smoothed
    /// covariance estimation (Shan & Kailath 1985), relative diagonal loading,
    /// and Cholesky/LU solve via nalgebra.
    ///
    /// # References
    /// - Capon (1969): original MVDR
    /// - Synnevåg, Austeng & Holm (2007): medical ultrasound MVDR
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[cfg(not(feature = "gpu"))]
    pub(super) fn process_mvdr_3d(
        &mut self,
        rf_data: &Array4<f32>,
        diagonal_loading: f32,
        subarray_size: [usize; 3],
    ) -> KwaversResult<Array3<f32>> {
        super::super::cpu::mvdr_cpu(rf_data, &self.config, diagonal_loading, subarray_size)
    }
}
