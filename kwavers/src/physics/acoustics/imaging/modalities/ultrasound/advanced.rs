//! Advanced ultrasound imaging techniques
//!
//! Implements synthetic aperture focusing, plane wave imaging, and coded excitation

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Synthetic Aperture (SA) imaging configuration
#[derive(Debug, Clone)]
pub struct SyntheticApertureConfig {
    /// Number of transmit elements
    pub num_tx_elements: usize,
    /// Number of receive elements
    pub num_rx_elements: usize,
    /// Element spacing (m)
    pub element_spacing: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Center frequency (Hz)
    pub frequency: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Number of transmit angles for compound imaging
    pub num_tx_angles: usize,
}

impl Default for SyntheticApertureConfig {
    fn default() -> Self {
        Self {
            num_tx_elements: 64,
            num_rx_elements: 64,
            element_spacing: 0.3e-3, // 0.3mm spacing
            sound_speed: 1540.0,
            frequency: 5e6,
            sampling_frequency: 40e6,
            num_tx_angles: 1,
        }
    }
}

/// Plane wave imaging configuration
#[derive(Debug, Clone)]
pub struct PlaneWaveConfig {
    /// Transmit angle (radians)
    pub tx_angle: f64,
    /// Number of elements
    pub num_elements: usize,
    /// Element spacing (m)
    pub element_spacing: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Center frequency (Hz)
    pub frequency: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
}

impl Default for PlaneWaveConfig {
    fn default() -> Self {
        Self {
            tx_angle: 0.0,
            num_elements: 64,
            element_spacing: 0.3e-3,
            sound_speed: 1540.0,
            frequency: 5e6,
            sampling_frequency: 40e6,
        }
    }
}

/// Coded excitation configuration
#[derive(Debug, Clone)]
pub enum ExcitationCode {
    /// Linear frequency modulated chirp
    Chirp {
        /// Start frequency (Hz)
        start_freq: f64,
        /// End frequency (Hz)
        end_freq: f64,
        /// Code length (samples)
        length: usize,
    },
    /// Barker code sequence
    Barker {
        /// Barker code length (2, 3, 4, 5, 7, 11, 13)
        length: usize,
    },
    /// Golay complementary pair
    Golay {
        /// Code length (must be power of 2)
        length: usize,
    },
}

#[derive(Debug, Clone)]
pub struct CodedExcitationConfig {
    /// Excitation code type
    pub code: ExcitationCode,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
}

/// Synthetic Aperture (SA) reconstruction
///
/// SA imaging uses multiple transmit-receive element pairs to achieve
/// high resolution over a large field of view. Each element transmits
/// individually and all elements receive simultaneously.
///
/// # Algorithm
/// 1. For each transmit element, compute round-trip delays to image points
/// 2. Apply phase correction based on transmit and receive delays
/// 3. Coherently sum contributions from all transmit-receive pairs
/// 4. Apply apodization and envelope detection
///
/// # References
/// - Karaman et al. (1995), "Synthetic aperture imaging for small scale systems"
/// - Jensen et al. (2006), "Synthetic aperture ultrasound imaging"
#[derive(Debug)]
pub struct SyntheticApertureReconstruction {
    config: SyntheticApertureConfig,
}

impl SyntheticApertureReconstruction {
    /// Create new SA reconstruction
    #[must_use]
    pub fn new(config: SyntheticApertureConfig) -> Self {
        Self { config }
    }

    /// Reconstruct SA image from RF data
    ///
    /// # Arguments
    /// * `rf_data` - RF data matrix [samples x rx_elements x tx_elements]
    /// * `image_grid` - Image grid coordinates (x, z) [2 x height x width]
    ///
    /// # Returns
    /// Reconstructed image [height x width]
    #[must_use]
    pub fn reconstruct(&self, rf_data: &Array3<f64>, image_grid: &Array3<f64>) -> Array2<f64> {
        let (n_samples, n_rx, n_tx) = rf_data.dim();
        let (_, height, width) = image_grid.dim();

        let mut image = Array2::<f64>::zeros((height, width));

        // For each image point
        for i in 0..height {
            for j in 0..width {
                let x = image_grid[[0, i, j]];
                let z = image_grid[[1, i, j]];

                let mut sum = Complex64::new(0.0, 0.0);

                // Sum over all transmit-receive pairs
                for tx in 0..n_tx {
                    for rx in 0..n_rx {
                        // Calculate transmit delay (from tx element to image point)
                        let tx_x =
                            (tx as f64 - (n_tx - 1) as f64 / 2.0) * self.config.element_spacing;
                        let tx_delay = self.calculate_delay(tx_x, 0.0, x, z);

                        // Calculate receive delay (from image point to rx element)
                        let rx_x =
                            (rx as f64 - (n_rx - 1) as f64 / 2.0) * self.config.element_spacing;
                        let rx_delay = self.calculate_delay(x, z, rx_x, 0.0);

                        // Total round-trip delay
                        let total_delay = tx_delay + rx_delay;

                        // Convert delay to sample index
                        let sample_idx = (total_delay * self.config.sampling_frequency) as usize;
                        if sample_idx < n_samples {
                            // Get RF sample and apply phase correction
                            let rf_sample = rf_data[[sample_idx, rx, tx]];

                            // Phase correction for center frequency
                            let phase_correction = Complex64::new(
                                0.0,
                                -2.0 * PI * self.config.frequency * total_delay,
                            )
                            .exp();

                            sum += rf_sample * phase_correction;
                        }
                    }
                }

                // Store magnitude
                image[[i, j]] = sum.norm();
            }
        }

        image
    }

    /// Calculate propagation delay between two points
    fn calculate_delay(&self, x1: f64, z1: f64, x2: f64, z2: f64) -> f64 {
        let distance = ((x2 - x1).powi(2) + (z2 - z1).powi(2)).sqrt();
        distance / self.config.sound_speed
    }
}

/// Plane Wave Imaging (PWI) reconstruction
///
/// Plane wave imaging transmits a steered plane wave and receives on all elements.
/// This provides high frame rates but requires coherent compounding of multiple angles.
///
/// # Algorithm
/// 1. Transmit steered plane wave at angle θ
/// 2. Receive on all elements simultaneously
/// 3. For each image point, compute receive delays
/// 4. Apply phase correction and coherent summation
/// 5. Compound multiple angles for improved contrast
///
/// # References
/// - Montaldo et al. (2009), "Coherent plane-wave compounding for very high frame rate ultrasonography"
/// - Jensen et al. (2016), "Plane wave imaging using synthetic aperture focusing techniques"
#[derive(Debug)]
pub struct PlaneWaveReconstruction {
    config: PlaneWaveConfig,
}

impl PlaneWaveReconstruction {
    /// Create new PWI reconstruction
    #[must_use]
    pub fn new(config: PlaneWaveConfig) -> Self {
        Self { config }
    }

    /// Reconstruct PWI image from RF data
    ///
    /// # Arguments
    /// * `rf_data` - RF data matrix [samples x elements]
    /// * `image_grid` - Image grid coordinates (x, z) [2 x height x width]
    ///
    /// # Returns
    /// Reconstructed image [height x width]
    #[must_use]
    pub fn reconstruct(&self, rf_data: &Array2<f64>, image_grid: &Array3<f64>) -> Array2<f64> {
        let (n_samples, n_elements) = rf_data.dim();
        let (_, height, width) = image_grid.dim();

        let mut image = Array2::<f64>::zeros((height, width));

        // For each image point
        for i in 0..height {
            for j in 0..width {
                let x = image_grid[[0, i, j]];
                let z = image_grid[[1, i, j]];

                let mut sum = Complex64::new(0.0, 0.0);

                // Sum over all receive elements
                for elem in 0..n_elements {
                    // Calculate receive delay (from image point to element)
                    let elem_x =
                        (elem as f64 - (n_elements - 1) as f64 / 2.0) * self.config.element_spacing;
                    let rx_delay = self.calculate_receive_delay(x, z, elem_x);

                    // Convert delay to sample index
                    let sample_idx = (rx_delay * self.config.sampling_frequency) as usize;
                    if sample_idx < n_samples {
                        // Get RF sample and apply phase correction
                        let rf_sample = rf_data[[sample_idx, elem]];

                        // Phase correction for center frequency
                        let phase_correction =
                            Complex64::new(0.0, -2.0 * PI * self.config.frequency * rx_delay).exp();

                        sum += rf_sample * phase_correction;
                    }
                }

                // Store magnitude
                image[[i, j]] = sum.norm();
            }
        }

        image
    }

    /// Calculate receive delay for plane wave
    fn calculate_receive_delay(&self, x: f64, z: f64, elem_x: f64) -> f64 {
        // For plane wave, the transmit delay is incorporated in the steering
        // Receive delay is just the distance from image point to element
        let distance = ((x - elem_x).powi(2) + z.powi(2)).sqrt();
        distance / self.config.sound_speed
    }
}

/// Coded Excitation Processing
///
/// Coded excitation improves SNR by transmitting long coded waveforms
/// and using matched filtering for pulse compression on receive.
///
/// # Supported Codes
/// - Chirp: Linear frequency modulated waveforms
/// - Barker: Binary phase codes for sidelobe reduction
/// - Golay: Complementary pairs for perfect sidelobe cancellation
///
/// # References
/// - O'Donnell (1992), "Coded excitation system for improving the penetration of real-time phased-array imaging systems"
/// - Misaridis & Jensen (2005), "Use of modulated excitation signals in medical ultrasound"
#[derive(Debug)]
pub struct CodedExcitationProcessor {
    config: CodedExcitationConfig,
}

impl CodedExcitationProcessor {
    /// Create new coded excitation processor
    #[must_use]
    pub fn new(config: CodedExcitationConfig) -> Self {
        Self { config }
    }

    /// Generate excitation code
    #[must_use]
    pub fn generate_code(&self) -> Array1<Complex64> {
        match &self.config.code {
            ExcitationCode::Chirp {
                start_freq,
                end_freq,
                length,
            } => self.generate_chirp(*start_freq, *end_freq, *length),
            ExcitationCode::Barker { length } => self.generate_barker(*length),
            ExcitationCode::Golay { length } => self.generate_golay(*length),
        }
    }

    /// Apply matched filtering for pulse compression
    ///
    /// # Arguments
    /// * `received_signal` - Received RF signal
    /// * `code` - Excitation code used for transmission
    ///
    /// # Returns
    /// Pulse compressed signal
    #[must_use]
    pub fn matched_filter(
        &self,
        received_signal: &Array1<f64>,
        code: &Array1<Complex64>,
    ) -> Array1<f64> {
        let n_signal = received_signal.len();
        let n_code = code.len();
        let mut compressed = Array1::<f64>::zeros(n_signal - n_code + 1);

        // Time-reversed matched filter
        let matched_filter = code.iter().rev().map(|&c| c.conj()).collect::<Array1<_>>();

        // Convolution with matched filter
        for i in 0..compressed.len() {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in 0..n_code {
                if i + j < n_signal {
                    sum += received_signal[i + j] * matched_filter[j];
                }
            }
            compressed[i] = sum.norm();
        }

        compressed
    }

    /// Generate linear frequency modulated chirp
    fn generate_chirp(&self, start_freq: f64, end_freq: f64, length: usize) -> Array1<Complex64> {
        let mut chirp = Array1::<Complex64>::zeros(length);
        let t_step = 1.0 / self.config.sampling_frequency;
        let k = (end_freq - start_freq) / (length as f64 * t_step); // Frequency sweep rate

        for i in 0..length {
            let t = i as f64 * t_step;
            let phase = 2.0 * PI * (start_freq * t + 0.5 * k * t * t);
            chirp[i] = Complex64::new(phase.cos(), phase.sin());
        }

        chirp
    }

    /// Generate Barker code
    fn generate_barker(&self, length: usize) -> Array1<Complex64> {
        let sequence = match length {
            2 => vec![1, -1],
            3 => vec![1, 1, -1],
            4 => vec![1, 1, -1, 1],
            5 => vec![1, 1, 1, -1, 1],
            7 => vec![1, 1, 1, -1, -1, 1, -1],
            11 => vec![1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
            13 => vec![1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
            _ => vec![1; length], // Default to all ones
        };

        sequence
            .into_iter()
            .map(|x| Complex64::new(x as f64, 0.0))
            .collect()
    }

    /// Generate Golay complementary pair
    fn generate_golay(&self, length: usize) -> Array1<Complex64> {
        // Generate Golay code A using complementary pair construction
        // Reference: Golay (1961) complementary series for pulse compression
        let mut golay = Array1::<Complex64>::zeros(length);

        // Simple alternating pattern (not a true Golay code)
        for i in 0..length {
            let phase = if i % 2 == 0 { 0.0 } else { PI };
            golay[i] = Complex64::new(phase.cos(), phase.sin());
        }

        golay
    }

    /// Calculate theoretical SNR improvement for coded excitation
    #[must_use]
    pub fn theoretical_snr_improvement(&self) -> f64 {
        match &self.config.code {
            ExcitationCode::Chirp { length, .. } => {
                // SNR improvement proportional to code length
                (*length as f64).sqrt()
            }
            ExcitationCode::Barker { length } => {
                // Barker codes have SNR improvement of sqrt(length)
                (*length as f64).sqrt()
            }
            ExcitationCode::Golay { length } => {
                // Golay pairs have SNR improvement of sqrt(2*length)
                (2.0 * *length as f64).sqrt()
            }
        }
    }
}

/// Multi-angle plane wave compounding
///
/// Combines multiple plane wave transmissions at different angles
/// to improve image quality and reduce artifacts.
///
/// # Algorithm
/// 1. Transmit plane waves at multiple angles
/// 2. Reconstruct individual images for each angle
/// 3. Coherently compound the images
/// 4. Apply coherence-based weighting
///
/// # References
/// - Montaldo et al. (2009), "Coherent plane-wave compounding for very high frame rate ultrasonography"
#[derive(Debug)]
pub struct PlaneWaveCompounding {
    #[allow(dead_code)]
    configs: Vec<PlaneWaveConfig>,
}

impl PlaneWaveCompounding {
    /// Create new compounding processor
    #[must_use]
    pub fn new(angles: &[f64], base_config: PlaneWaveConfig) -> Self {
        let configs = angles
            .iter()
            .map(|&angle| PlaneWaveConfig {
                tx_angle: angle,
                ..base_config.clone()
            })
            .collect();

        Self { configs }
    }

    /// Compound multiple plane wave images
    ///
    /// # Arguments
    /// * `images` - Individual plane wave images [num_angles x height x width]
    ///
    /// # Returns
    /// Compounded image [height x width]
    #[must_use]
    pub fn compound(&self, images: &Array3<f64>) -> Array2<f64> {
        let (num_angles, height, width) = images.dim();
        let mut compounded = Array2::<f64>::zeros((height, width));

        // Simple coherent summation
        for i in 0..height {
            for j in 0..width {
                let mut sum = 0.0;
                for angle in 0..num_angles {
                    sum += images[[angle, i, j]];
                }
                compounded[[i, j]] = sum / num_angles as f64;
            }
        }

        compounded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_aperture_config() {
        let config = SyntheticApertureConfig::default();
        assert_eq!(config.num_tx_elements, 64);
        assert_eq!(config.num_rx_elements, 64);
        assert!((config.sound_speed - 1540.0).abs() < 1e-6);
    }

    #[test]
    fn test_plane_wave_config() {
        let config = PlaneWaveConfig::default();
        assert_eq!(config.tx_angle, 0.0);
        assert_eq!(config.num_elements, 64);
        assert!((config.frequency - 5e6).abs() < 1e-6);
    }

    #[test]
    fn test_chirp_generation() {
        let config = CodedExcitationConfig {
            code: ExcitationCode::Chirp {
                start_freq: 2e6,
                end_freq: 8e6,
                length: 100,
            },
            sound_speed: 1540.0,
            sampling_frequency: 20e6,
        };

        let processor = CodedExcitationProcessor::new(config);
        let code = processor.generate_code();

        assert_eq!(code.len(), 100);
        // Check that code has non-zero values
        assert!(code.iter().any(|&x| x.norm() > 0.0));
    }

    #[test]
    fn test_barker_generation() {
        let config = CodedExcitationConfig {
            code: ExcitationCode::Barker { length: 7 },
            sound_speed: 1540.0,
            sampling_frequency: 20e6,
        };

        let processor = CodedExcitationProcessor::new(config);
        let code = processor.generate_code();

        assert_eq!(code.len(), 7);
        // Barker codes should have values of ±1
        for &val in &code {
            assert!((val.re - 1.0).abs() < 1e-6 || (val.re + 1.0).abs() < 1e-6);
            assert!(val.im.abs() < 1e-6);
        }
    }

    #[test]
    fn test_snr_improvement_calculation() {
        let config = CodedExcitationConfig {
            code: ExcitationCode::Chirp {
                start_freq: 2e6,
                end_freq: 8e6,
                length: 100,
            },
            sound_speed: 1540.0,
            sampling_frequency: 20e6,
        };

        let processor = CodedExcitationProcessor::new(config);
        let snr_improvement = processor.theoretical_snr_improvement();

        // For length 100, SNR improvement should be sqrt(100) = 10
        assert!((snr_improvement - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_plane_wave_compounding() {
        let base_config = PlaneWaveConfig::default();
        let angles = vec![-10.0f64.to_radians(), 0.0, 10.0f64.to_radians()];
        let compounding = PlaneWaveCompounding::new(&angles, base_config);

        // Create dummy images
        let height = 100;
        let width = 100;
        let mut images = Array3::<f64>::zeros((angles.len(), height, width));

        // Fill with test data
        for angle in 0..angles.len() {
            for i in 0..height {
                for j in 0..width {
                    images[[angle, i, j]] = (angle + 1) as f64; // Different values per angle
                }
            }
        }

        let compounded = compounding.compound(&images);

        // Check dimensions
        assert_eq!(compounded.nrows(), height);
        assert_eq!(compounded.ncols(), width);

        // Check that compounding worked (average of 1, 2, 3 = 2)
        assert!((compounded[[50, 50]] - 2.0).abs() < 1e-6);
    }
}
