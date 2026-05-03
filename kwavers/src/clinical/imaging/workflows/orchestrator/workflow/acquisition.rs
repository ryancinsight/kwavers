//! Ultrasound data acquisition for the clinical workflow orchestrator.

use super::ClinicalWorkflowOrchestrator;
use crate::core::error::KwaversResult;
use ndarray::Array3;

#[cfg(not(feature = "gpu"))]
use super::super::super::config::QualityPreference;
#[cfg(not(feature = "gpu"))]
use super::super::super::simulation::generate_realistic_rf_volume;
#[cfg(not(feature = "gpu"))]
use crate::physics::acoustics::imaging::modalities::ultrasound::{
    compute_bmode_image, UltrasoundConfig, UltrasoundMode,
};
#[cfg(not(feature = "gpu"))]
use ndarray::Array2;

#[cfg(feature = "gpu")]
use super::super::super::simulation::generate_realistic_rf_data;

impl ClinicalWorkflowOrchestrator {
    pub(super) fn acquire_ultrasound_data(&self) -> KwaversResult<Array3<f64>> {
        #[cfg(feature = "gpu")]
        {
            use crate::analysis::signal_processing::beamforming::three_dimensional::config::BeamformingConfig3D;
            use crate::physics::acoustics::imaging::modalities::ultrasound::{
                compute_bmode_image, UltrasoundConfig, UltrasoundMode,
            };
            use ndarray::Array2;

            let config = UltrasoundConfig {
                mode: UltrasoundMode::BMode,
                frequency: 5e6,
                sampling_frequency: 40e6,
                dynamic_range: 60.0,
                tgc_enabled: true,
            };

            let beamforming_config = BeamformingConfig3D {
                base_config: crate::domain::sensor::beamforming::BeamformingConfig {
                    sound_speed: 1540.0,
                    sampling_frequency: config.sampling_frequency,
                    reference_frequency: config.frequency,
                    diagonal_loading: 0.01,
                    num_snapshots: 100,
                    spatial_smoothing: None,
                },
                volume_dims: (256, 256, 128),
                voxel_spacing: (0.001, 0.001, 0.001),
                num_elements_3d: (64, 64, 1),
                element_spacing_3d: (0.0003, 0.0003, 0.0),
                center_frequency: config.frequency,
                sampling_frequency: config.sampling_frequency,
                sound_speed: 1540.0,
                gpu_device: None,
                enable_streaming: false,
                streaming_buffer_size: 1024,
            };

            let rf_data = generate_realistic_rf_data(&beamforming_config);
            let mut bmode_volume = Array3::zeros(beamforming_config.volume_dims);

            for elev in 0..beamforming_config.volume_dims.2 {
                let mut rf_slice = Array2::zeros((
                    beamforming_config.volume_dims.0,
                    beamforming_config.volume_dims.1,
                ));
                for depth in 0..beamforming_config.volume_dims.0 {
                    for lat in 0..beamforming_config.volume_dims.1 {
                        rf_slice[[depth, lat]] = rf_data[[depth, lat, elev]];
                    }
                }
                let bmode_slice = compute_bmode_image(&rf_slice, &config);
                for depth in 0..beamforming_config.volume_dims.0 {
                    for lat in 0..beamforming_config.volume_dims.1 {
                        bmode_volume[[depth, lat, elev]] = bmode_slice[[depth, lat]];
                    }
                }
            }

            Ok(bmode_volume)
        }

        #[cfg(not(feature = "gpu"))]
        {
            let config = UltrasoundConfig {
                mode: UltrasoundMode::BMode,
                frequency: 5e6,
                sampling_frequency: 40e6,
                dynamic_range: 60.0,
                tgc_enabled: true,
            };

            let volume_dims = match self.config.quality_preference {
                QualityPreference::Quality => (256, 256, 128),
                QualityPreference::Balanced => (128, 128, 64),
                QualityPreference::Speed => (64, 64, 32),
            };

            let rf_data = generate_realistic_rf_volume(
                volume_dims,
                1540.0,
                config.sampling_frequency,
                config.frequency,
            );
            let mut bmode_volume = Array3::zeros(volume_dims);

            for elev in 0..volume_dims.2 {
                let mut rf_slice = Array2::zeros((volume_dims.0, volume_dims.1));
                for depth in 0..volume_dims.0 {
                    for lat in 0..volume_dims.1 {
                        rf_slice[[depth, lat]] = rf_data[[depth, lat, elev]];
                    }
                }
                let bmode_slice = compute_bmode_image(&rf_slice, &config);
                for depth in 0..volume_dims.0 {
                    for lat in 0..volume_dims.1 {
                        bmode_volume[[depth, lat, elev]] = bmode_slice[[depth, lat]];
                    }
                }
            }

            Ok(bmode_volume)
        }
    }
}
