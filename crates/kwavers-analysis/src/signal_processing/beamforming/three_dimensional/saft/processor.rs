//! SaftProcessor struct and reconstruction logic.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array3, Array4};
use log::info;
use std::f64::consts::PI;

use super::super::{BeamformingAlgorithm3D, BeamformingConfig3D};
use super::config::SaftConfig;
use crate::signal_processing::beamforming::time_domain::coherence::amplitude_coherence_from_sums;
use kwavers_core::constants::numerical::TWO_PI;

/// Euclidean distance between two 3D points.
pub(super) fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
}

/// SAFT processor for 3D volumetric reconstruction.
#[derive(Debug)]
pub struct SaftProcessor {
    pub(super) config: SaftConfig,
    beamforming_config: super::super::config::BeamformingConfig3D,
}

impl SaftProcessor {
    /// Create new SAFT processor.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(saft_config: SaftConfig, beamforming_config: BeamformingConfig3D) -> Self {
        Self {
            config: saft_config,
            beamforming_config,
        }
    }

    /// Extract SAFT parameters from BeamformingAlgorithm3D::SAFT3D variant.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn from_algorithm(
        algorithm: &BeamformingAlgorithm3D,
        beamforming_config: BeamformingConfig3D,
    ) -> KwaversResult<Self> {
        match algorithm {
            BeamformingAlgorithm3D::SAFT3D { virtual_sources } => Ok(Self::new(
                SaftConfig {
                    virtual_sources: *virtual_sources,
                    ..Default::default()
                },
                beamforming_config,
            )),
            _ => Err(KwaversError::InvalidInput(
                "Expected SAFT3D algorithm variant".to_owned(),
            )),
        }
    }

    /// Compute round-trip time-of-flight from transmit → voxel → receive.
    pub(super) fn compute_time_of_flight(
        &self,
        tx_position: [f64; 3],
        rx_position: [f64; 3],
        voxel_position: [f64; 3],
        sound_speed: f64,
    ) -> f64 {
        let tx_to_voxel_dist = distance3(tx_position, voxel_position);
        let voxel_to_rx_dist = distance3(voxel_position, rx_position);
        (tx_to_voxel_dist + voxel_to_rx_dist) / sound_speed
    }

    /// Interpolate RF sample at time index (nearest-neighbour, bounds-checked).
    fn extract_rf_sample(rf_data: &Array4<f64>, element_idx: usize, time_idx: usize) -> f64 {
        rf_data[[0, element_idx, time_idx, 0]]
    }

    /// Demodulate a real RF sample at known time-of-flight.
    #[inline]
    pub(super) fn demodulate(
        sample: f64,
        center_frequency: f64,
        time_of_flight: f64,
    ) -> (f64, f64) {
        let phase = TWO_PI * center_frequency * time_of_flight;
        (sample * phase.cos(), sample * (-phase.sin()))
    }

    /// Hamming apodization weight for sidelobe suppression.
    pub(super) fn compute_apodization_weight(
        &self,
        tx_idx: usize,
        rx_idx: usize,
        total_elements: usize,
    ) -> f64 {
        let center = total_elements as f64 / 2.0;
        let pos = (tx_idx + rx_idx) as f64 / 2.0;
        let normalized_pos = (pos - center) / center.max(1.0);
        0.46f64.mul_add((PI * normalized_pos).cos(), 0.54)
    }

    /// Reconstruct a 3D volume using SAFT with carrier-phase demodulation.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn reconstruct_volume(&self, rf_data: &Array4<f32>) -> KwaversResult<Array3<f32>> {
        let start_time = std::time::Instant::now();

        self.validate_input(rf_data)?;

        let rf_data_f64 = rf_data.mapv(|x| x as f64);

        let (nx, ny, nz) = (
            self.beamforming_config.volume_dims.0,
            self.beamforming_config.volume_dims.1,
            self.beamforming_config.volume_dims.2,
        );
        let (dx, dy, dz) = (
            self.beamforming_config.voxel_spacing.0,
            self.beamforming_config.voxel_spacing.1,
            self.beamforming_config.voxel_spacing.2,
        );
        let sound_speed = self.beamforming_config.sound_speed;
        let sampling_frequency = self.beamforming_config.sampling_frequency;
        let center_frequency = self.beamforming_config.center_frequency;

        let mut volume = Array3::<f64>::zeros((nx, ny, nz));

        let (num_tx, num_rx) = (
            self.beamforming_config.num_elements_3d.0,
            self.beamforming_config.num_elements_3d.1,
        );
        let (tx_spacing, rx_spacing) = (
            self.beamforming_config.element_spacing_3d.0,
            self.beamforming_config.element_spacing_3d.1,
        );
        let n_rf_samples = rf_data_f64.shape()[2];

        for i in 0..nx {
            let x = i as f64 * dx;
            for j in 0..ny {
                let y = j as f64 * dy;
                for k in 0..nz {
                    let z = k as f64 * dz;
                    let voxel_position = [x, y, z];

                    let mut sum_i = 0.0_f64;
                    let mut sum_q = 0.0_f64;
                    // Σ per-element energies for the Mallart & Fink coherence
                    // factor. Demodulation is a phase rotation (I²+Q² = sample²),
                    // so the apodized per-element magnitude is `apod·|sample|`
                    // and its energy is `(apod·sample)²`.
                    let mut sum_of_squares = 0.0_f64;
                    let mut num_contributions = 0usize;

                    for tx_idx in 0..num_tx {
                        let tx_x = tx_idx as f64 * tx_spacing;
                        let tx_position = [tx_x, 0.0, 0.0];

                        for rx_idx in 0..num_rx {
                            let rx_x = rx_idx as f64 * rx_spacing;
                            let rx_position = [rx_x, 0.0, 0.0];

                            let tof = self.compute_time_of_flight(
                                tx_position,
                                rx_position,
                                voxel_position,
                                sound_speed,
                            );

                            let sample_idx = (tof * sampling_frequency).round() as usize;
                            if sample_idx >= n_rf_samples {
                                continue;
                            }

                            let element_idx = tx_idx * num_rx + rx_idx;
                            let sample =
                                Self::extract_rf_sample(&rf_data_f64, element_idx, sample_idx);

                            let apod =
                                self.compute_apodization_weight(tx_idx, rx_idx, num_tx * num_rx);

                            let (i_comp, q_comp) = Self::demodulate(sample, center_frequency, tof);

                            sum_i += apod * i_comp;
                            sum_q += apod * q_comp;
                            sum_of_squares += (apod * sample).powi(2);
                            num_contributions += 1;
                        }
                    }

                    let intensity = sum_q.mul_add(sum_q, sum_i.powi(2));

                    volume[[i, j, k]] =
                        if self.config.coherence_factor_enabled && num_contributions > 0 {
                            let coherent_mag = sum_i.hypot(sum_q);
                            let cf = amplitude_coherence_from_sums(
                                coherent_mag,
                                sum_of_squares,
                                num_contributions,
                            );
                            intensity * cf
                        } else {
                            intensity
                        };
                }
            }
        }

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        info!("SAFT reconstruction completed in {:.2} ms", processing_time);

        Ok(volume.mapv(|v| v as f32))
    }

    /// Validate input RF data dimensions.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub(super) fn validate_input(&self, rf_data: &Array4<f32>) -> KwaversResult<()> {
        if rf_data.is_empty() {
            return Err(KwaversError::InvalidInput(
                "RF data array is empty".to_owned(),
            ));
        }

        let channels = rf_data.shape()[1];
        let expected_channels = self.beamforming_config.num_elements_3d.0
            * self.beamforming_config.num_elements_3d.1
            * self.beamforming_config.num_elements_3d.2;

        if channels != expected_channels {
            return Err(KwaversError::InvalidInput(format!(
                "Channel count mismatch: expected {}, got {}",
                expected_channels, channels
            )));
        }

        if rf_data.shape()[2] == 0 {
            return Err(KwaversError::InvalidInput(
                "RF data must contain at least one sample per channel".to_owned(),
            ));
        }

        Ok(())
    }
}
