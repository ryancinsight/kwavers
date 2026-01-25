//! 3D Synthetic Aperture Focusing Technique (SAFT) Implementation
//!
//! This module implements the SAFT algorithm for volumetric ultrasound imaging.
//! SAFT synthesizes a large virtual aperture from multiple transmit-receive
//! events to achieve high-resolution imaging.
//!
//! # Mathematical Foundation
//! 
//! SAFT reconstruction for voxel r = (x, y, z):
//!   I_SAFT(r) = |Σᵢ Σⱼ wᵢⱼ · RF[i,j,t(i,j,r)]|²
//! where:
//!   t(i,j,r) = (|rᵢ - r| + |r - rⱼ|) / c
//!   wᵢⱼ = apodization weight for TX i, RX j
//!   i, j iterate over transmit and receive positions
//!
//! # References
//! - Frazier & O'Brien, "Synthetic Aperture Techniques with a Virtual Source Element" (1998)
//! - Karaman et al., "Synthetic aperture imaging for small scale systems" (1995)
//! - Nikolov & Jensen, "Virtual ultrasound sources in high-resolution ultrasound imaging" (2002)

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array3, Array4};
use std::f64::consts::PI;

use super::{ApodizationWindow, BeamformingAlgorithm3D, BeamformingConfig3D};

/// SAFT configuration parameters
#[derive(Debug, Clone)]
pub struct SaftConfig {
    /// Number of virtual sources for synthetic aperture
    pub virtual_sources: usize,
    /// Apodization window for sidelobe suppression
    pub apodization: ApodizationWindow,
    /// Coherence factor weighting enabled
    pub coherence_factor_enabled: bool,
    /// F-number for dynamic focusing
    pub f_number: f64,
}

impl Default for SaftConfig {
    fn default() -> Self {
        Self {
            virtual_sources: 100,
            apodization: super::config::ApodizationWindow::Hamming,
            coherence_factor_enabled: true,
            f_number: 1.5,
        }
    }
}

/// SAFT processor for 3D volumetric reconstruction
pub struct SaftProcessor {
    config: SaftConfig,
    beamforming_config: super::config::BeamformingConfig3D,
}

impl SaftProcessor {
    /// Create new SAFT processor
    pub fn new(
        saft_config: SaftConfig,
        beamforming_config: BeamformingConfig3D,
    ) -> Self {
        Self {
            config: saft_config,
            beamforming_config,
        }
    }

    /// Extract SAFT parameters from BeamformingAlgorithm3D::SAFT3D variant
    pub fn from_algorithm(
        algorithm: &BeamformingAlgorithm3D,
        beamforming_config: BeamformingConfig3D,
    ) -> KwaversResult<Self> {
        match algorithm {
            BeamformingAlgorithm3D::SAFT3D { virtual_sources } => {
                Ok(Self::new(
                    SaftConfig {
                        virtual_sources: *virtual_sources,
                        ..Default::default()
                    },
                    beamforming_config,
                ))
            }
            _ => Err(KwaversError::InvalidInput(
                "Expected SAFT3D algorithm variant".to_string(),
            )),
        }
    }

    /// Compute time-of-flight from transmit position to voxel to receive element
    ///
    /// # Arguments
    /// * `tx_position` - Transmit element position (x, y, z) in meters
    /// * `rx_position` - Receive element position (x, y, z) in meters
    /// * `voxel_position` - Voxel position (x, y, z) in meters
    /// * `sound_speed` - Sound speed in tissue (m/s)
    ///
    /// # Returns
    /// Time-of-flight in seconds
    fn compute_time_of_flight(
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

    /// Extract RF sample at computed time index for each TX-RX pair
    fn extract_rf_sample(
        &self,
        rf_data: &Array4<f64>,
        element_idx: usize,
        time_idx: usize,
    ) -> f64 {
        // Convert to f64 for processing
        rf_data[[0, element_idx, time_idx, 0]]
    }

    /// Apply phase correction for synthetic aperture coherence
    fn apply_phase_correction(&self, sample: f64, phase: f64) -> f64 {
        sample * phase.cos()
    }

    /// Compute apodization weight for sidelobe suppression
    fn compute_apodization_weight(
        &self,
        tx_idx: usize,
        rx_idx: usize,
        total_elements: usize,
    ) -> f64 {
        // Hamming window implementation
        let center = total_elements as f64 / 2.0;
        let pos = (tx_idx + rx_idx) as f64 / 2.0;
        let normalized_pos = (pos - center) / center;
        0.54 + 0.46 * (2.0 * PI * normalized_pos).cos()
    }

    /// Compute coherence factor for adaptive weighting
    fn compute_coherence_factor(
        &self,
        coherent_sum: f64,
        incoherent_sum: f64,
        num_contributions: usize,
    ) -> f64 {
        if num_contributions == 0 {
            return 0.0;
        }
        let denominator = num_contributions as f64 * incoherent_sum.powi(2);
        if denominator > 0.0 {
            (coherent_sum.abs().powi(2)) / denominator
        } else {
            0.0
        }
    }

    /// Main SAFT reconstruction algorithm
    ///
    /// # Arguments
    /// * `rf_data` - RF data array (frames × channels × samples × 1)
    ///
    /// # Returns
    /// Reconstructed 3D volume (x × y × z)
    pub fn reconstruct_volume(&self, rf_data: &Array4<f32>) -> KwaversResult<Array3<f32>> {
        let start_time = std::time::Instant::now();

        // Validate input dimensions
        self.validate_input(rf_data)?;

        // Convert RF data to f64 for processing
        let rf_data_f64 = rf_data.mapv(|x| x as f64);

        // Get configuration parameters
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

        // Initialize output volume
        let mut volume = Array3::zeros((nx, ny, nz));
        let mut coherence_volume = Array3::zeros((nx, ny, nz));

        // Get transducer element positions
        let (num_tx, num_rx) = (
            self.beamforming_config.num_elements_3d.0,
            self.beamforming_config.num_elements_3d.1,
        );
        let (tx_spacing, rx_spacing) = (
            self.beamforming_config.element_spacing_3d.0,
            self.beamforming_config.element_spacing_3d.1,
        );

        // For each voxel in reconstruction volume
        for i in 0..nx {
            let x = i as f64 * dx;
            for j in 0..ny {
                let y = j as f64 * dy;
                for k in 0..nz {
                    let z = k as f64 * dz;
                    let voxel_position = [x, y, z];

                    let mut coherent_sum = 0.0;
                    let mut incoherent_sum = 0.0;
                    let mut num_contributions = 0;

                    // Sum coherently across all virtual aperture positions
                    for tx_idx in 0..num_tx {
                        let tx_x = tx_idx as f64 * tx_spacing;
                        let tx_position = [tx_x, 0.0, 0.0]; // Simplified 1D array for now

                        for rx_idx in 0..num_rx {
                            let rx_x = rx_idx as f64 * rx_spacing;
                            let rx_position = [rx_x, 0.0, 0.0]; // Simplified 1D array for now

                            // Compute time-of-flight
                            let time_of_flight = self.compute_time_of_flight(
                                tx_position, rx_position, voxel_position, sound_speed,
                            );

                            // Convert time to sample index
                            let sample_idx = (time_of_flight * sampling_frequency).round() as usize;

                            // Check if sample index is within bounds
                            if sample_idx < rf_data_f64.dim().2 {
                                // Extract RF sample - for now, use a simple element index
                                let element_idx = tx_idx * num_rx + rx_idx;
                                let sample = self.extract_rf_sample(
                                    &rf_data_f64, element_idx, sample_idx,
                                );

                                // Apply apodization weight
                                let apod_weight = self.compute_apodization_weight(
                                    tx_idx, rx_idx, num_tx * num_rx,
                                );

                                // Apply phase correction (simplified for now)
                                let phase_corrected = self.apply_phase_correction(sample, 0.0);

                                // Accumulate coherent and incoherent sums
                                coherent_sum += apod_weight * phase_corrected;
                                incoherent_sum += apod_weight * sample.abs();
                                num_contributions += 1;
                            }
                        }
                    }

                    // Compute final intensity
                    let intensity = coherent_sum.abs().powi(2);

                    // Apply coherence factor if enabled
                    if self.config.coherence_factor_enabled && num_contributions > 0 {
                        let cf = self.compute_coherence_factor(
                            coherent_sum, incoherent_sum, num_contributions,
                        );
                        volume[[i, j, k]] = (intensity * cf) as f32;
                    } else {
                        volume[[i, j, k]] = intensity as f32;
                    }

                    // Store coherence factor for debugging
                    if self.config.coherence_factor_enabled {
                        coherence_volume[[i, j, k]] = self.compute_coherence_factor(
                            coherent_sum, incoherent_sum, num_contributions,
                        ) as f32;
                    }
                }
            }
        }

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        println!(
            "SAFT reconstruction completed in {:.2} ms",
            processing_time
        );

        Ok(volume)
    }

    /// Validate input RF data dimensions
    fn validate_input(&self, rf_data: &Array4<f32>) -> KwaversResult<()> {
        let rf_dims = rf_data.dim();
        let channels = rf_dims.1;
        let samples = rf_dims.2;

        // Basic validation - ensure data is not empty
        if rf_data.is_empty() {
            return Err(KwaversError::InvalidInput(
                "RF data array is empty".to_string(),
            ));
        }

        let expected_channels = self.beamforming_config.num_elements_3d.0
            * self.beamforming_config.num_elements_3d.1
            * self.beamforming_config.num_elements_3d.2;

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

/// Compute Euclidean distance between two 3D points
fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_saft_config_default() {
        let config = SaftConfig::default();
        assert_eq!(config.virtual_sources, 100);
        assert!(matches!(
            config.apodization,
            ApodizationWindow::Hamming
        ));
        assert!(config.coherence_factor_enabled);
        assert_eq!(config.f_number, 1.5);
    }

    #[test]
    fn test_saft_processor_creation() {
        let saft_config = SaftConfig::default();
        let beamforming_config = BeamformingConfig3D::default();
        let processor = SaftProcessor::new(saft_config, beamforming_config);

        assert_eq!(processor.config.virtual_sources, 100);
    }

    #[test]
    fn test_saft_from_algorithm() {
        let algorithm = BeamformingAlgorithm3D::SAFT3D {
            virtual_sources: 50,
        };
        let beamforming_config = BeamformingConfig3D::default();

        let processor = SaftProcessor::from_algorithm(&algorithm, beamforming_config).unwrap();
        assert_eq!(processor.config.virtual_sources, 50);
    }

    #[test]
    fn test_time_of_flight_computation() {
        let saft_config = SaftConfig::default();
        let beamforming_config = BeamformingConfig3D::default();
        let processor = SaftProcessor::new(saft_config, beamforming_config);

        let tx_pos = [0.0, 0.0, 0.0];
        let rx_pos = [0.0, 0.0, 0.0];
        let voxel_pos = [0.001, 0.0, 0.0]; // 1mm away
        let sound_speed = 1540.0; // m/s

        let tof = processor.compute_time_of_flight(tx_pos, rx_pos, voxel_pos, sound_speed);
        let expected_tof = (0.001 + 0.001) / 1540.0; // Round-trip: 2mm / 1540 m/s

        assert!((tof - expected_tof).abs() < 1e-9);
    }

    #[test]
    fn test_distance_computation() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.001, 0.0, 0.0]; // 1mm apart

        let dist = distance3(a, b);
        assert!((dist - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_apodization_weight() {
        let saft_config = SaftConfig::default();
        let beamforming_config = BeamformingConfig3D::default();
        let processor = SaftProcessor::new(saft_config, beamforming_config);

        let weight = processor.compute_apodization_weight(50, 50, 100);
        // Hamming window at center should be 1.0 (0.54 + 0.46 * cos(0) = 0.54 + 0.46 = 1.0)
        assert!(weight > 0.9 && weight <= 1.0);
    }

    #[test]
    fn test_coherence_factor() {
        let saft_config = SaftConfig::default();
        let beamforming_config = BeamformingConfig3D::default();
        let processor = SaftProcessor::new(saft_config, beamforming_config);

        let cf = processor.compute_coherence_factor(10.0, 5.0, 10);
        // CF = |sum|² / (N * sum|samples|²) = 100 / (10 * 25) = 0.4
        assert!((cf - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_input_validation() {
        let saft_config = SaftConfig::default();
        let beamforming_config = BeamformingConfig3D::default();
        let processor = SaftProcessor::new(saft_config, beamforming_config);

        // Test empty data
        let empty_data = Array4::<f32>::zeros((0, 0, 0, 0));
        assert!(processor.validate_input(&empty_data).is_err());

        // Test valid data
        let valid_data = Array4::<f32>::zeros((1, 16384, 1024, 1));
        assert!(processor.validate_input(&valid_data).is_ok());
    }

    #[test]
    fn test_reconstruct_volume_basic() {
        // Use smaller dimensions for faster testing
        let mut beamforming_config = BeamformingConfig3D::default();
        beamforming_config.volume_dims = (32, 32, 32); // Smaller volume
        beamforming_config.num_elements_3d = (8, 8, 4); // Fewer elements

        let saft_config = SaftConfig::default();
        let processor = SaftProcessor::new(saft_config, beamforming_config);

        // Create simple test data with a point target
        let num_elements = 8 * 8 * 4;
        let mut rf_data = Array4::<f32>::zeros((1, num_elements, 512, 1));
        // Add a simple impulse response at sample 256
        for elem in 0..num_elements {
            rf_data[[0, elem, 256, 0]] = 1.0;
        }

        let result = processor.reconstruct_volume(&rf_data);
        if let Err(e) = &result {
            println!("Reconstruction error: {:?}", e);
        }
        assert!(result.is_ok());
        let volume = result.unwrap();

        // Check output dimensions
        assert_eq!(volume.dim(), (32, 32, 32));

        // Check that some values are non-zero (basic functionality test)
        let max_value = volume.iter().fold(0.0, |max, &val| val.max(max));
        assert!(max_value > 0.0);
    }
}
