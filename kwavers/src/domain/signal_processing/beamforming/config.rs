//! Beamforming Configuration
//!
//! Domain-level configuration for beamforming operations,
//! independent of implementation details.

use std::fmt::Debug;

/// Window function for beamforming (Hamming, Hann, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum WindowFunction {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hamming window
    #[default]
    Hamming,
    /// Hann (Hanning) window
    Hann,
    /// Blackman window
    Blackman,
    /// Kaiser window with beta parameter
    Kaiser(f64),
}

impl WindowFunction {
    /// Apply window to array
    pub fn apply(&self, data: &mut [f64]) {
        let n = data.len();
        match self {
            WindowFunction::Rectangular => {
                // No windowing
            }
            WindowFunction::Hamming => {
                for (i, item) in data.iter_mut().enumerate().take(n) {
                    *item *= 0.54
                        - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
                }
            }
            WindowFunction::Hann => {
                for (i, item) in data.iter_mut().enumerate().take(n) {
                    *item *= 0.5
                        * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());
                }
            }
            WindowFunction::Blackman => {
                let a0 = 0.42;
                let a1 = 0.5;
                let a2 = 0.08;
                for (i, item) in data.iter_mut().enumerate().take(n) {
                    let x = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
                    *item *= a0 - a1 * x.cos() + a2 * (2.0 * x).cos();
                }
            }
            WindowFunction::Kaiser(beta) => {
                // Kaiser window: w(n) = I₀(β √(1 − ((2n/(N−1)) − 1)²)) / I₀(β)
                // where I₀ is the zeroth-order modified Bessel function of the first kind
                let i0_beta = bessel_i0(*beta);
                for (i, item) in data.iter_mut().enumerate().take(n) {
                    let t = 2.0 * i as f64 / (n - 1) as f64 - 1.0; // −1..1
                    let arg = beta * (1.0 - t * t).max(0.0).sqrt();
                    *item *= bessel_i0(arg) / i0_beta;
                }
            }
        }
    }
}

/// Zeroth-order modified Bessel function of the first kind I₀(x).
///
/// Uses the power series: I₀(x) = Σ_{k=0}^∞ [(x/2)^k / k!]²
/// Converges very quickly — 25 terms gives >15 digits of precision for typical β values (0–40).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = x / 2.0;
    for k in 1..=25 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}

/// Beamforming configuration
///
/// Defines the parameters and settings for beamforming operations.
/// This is implementation-agnostic - the same config can be used with
/// different beamforming processors (physics-based, neural, adaptive, etc.)
#[derive(Debug, Clone)]
pub struct BeamformingConfig {
    /// Transducer array geometry (sensor positions [x, y, z])
    pub array_geometry: Vec<[f64; 3]>,

    /// Center frequency [Hz]
    pub frequency: f64,

    /// Sampling frequency [Hz]
    pub sampling_frequency: f64,

    /// Speed of sound [m/s]
    pub sound_speed: f64,

    /// Beamforming angle range: (azimuth_min, azimuth_max, elevation_min, elevation_max) [radians]
    pub angular_range: Option<(f64, f64, f64, f64)>,

    /// Window function for apodization
    pub window: WindowFunction,

    /// Number of spatial grid points in output
    pub num_beams: usize,

    /// Focal depth [m] (None = unfocused)
    pub focal_depth: Option<f64>,

    /// Process for aperture (fraction of array to use: 0.0-1.0)
    pub aperture_fraction: f64,

    /// Additional metadata (algorithm-specific)
    pub metadata: std::collections::HashMap<String, String>,
}

impl BeamformingConfig {
    /// Create a new beamforming configuration
    pub fn new(
        array_geometry: Vec<[f64; 3]>,
        frequency: f64,
        sampling_frequency: f64,
        sound_speed: f64,
    ) -> Self {
        Self {
            array_geometry,
            frequency,
            sampling_frequency,
            sound_speed,
            angular_range: None,
            window: WindowFunction::default(),
            num_beams: 64,
            focal_depth: None,
            aperture_fraction: 1.0,
            metadata: Default::default(),
        }
    }

    /// Set focal depth
    pub fn with_focal_depth(mut self, depth: f64) -> Self {
        self.focal_depth = Some(depth);
        self
    }

    /// Set angular range for beamforming
    pub fn with_angular_range(
        mut self,
        azimuth_min: f64,
        azimuth_max: f64,
        elevation_min: f64,
        elevation_max: f64,
    ) -> Self {
        self.angular_range = Some((azimuth_min, azimuth_max, elevation_min, elevation_max));
        self
    }

    /// Set window function
    pub fn with_window(mut self, window: WindowFunction) -> Self {
        self.window = window;
        self
    }

    /// Set number of output beams
    pub fn with_num_beams(mut self, num_beams: usize) -> Self {
        self.num_beams = num_beams;
        self
    }

    /// Get wavelength [m]
    pub fn wavelength(&self) -> f64 {
        self.sound_speed / self.frequency
    }

    /// Get time window for one period [s]
    pub fn period(&self) -> f64 {
        1.0 / self.frequency
    }

    /// Get spatial resolution at distance r [m]
    pub fn spatial_resolution(&self, distance: f64) -> f64 {
        // Simple estimate: wavelength * distance / array_aperture
        let aperture = self.array_aperture();
        self.wavelength() * distance / aperture
    }

    /// Get array aperture size [m]
    pub fn array_aperture(&self) -> f64 {
        if self.array_geometry.is_empty() {
            return 0.0;
        }
        let mut min_x = self.array_geometry[0][0];
        let mut max_x = min_x;
        let mut min_y = self.array_geometry[0][1];
        let mut max_y = min_y;

        for pos in &self.array_geometry {
            min_x = min_x.min(pos[0]);
            max_x = max_x.max(pos[0]);
            min_y = min_y.min(pos[1]);
            max_y = max_y.max(pos[1]);
        }

        let dx = max_x - min_x;
        let dy = max_y - min_y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Get number of active elements
    pub fn num_active_elements(&self) -> usize {
        ((self.array_geometry.len() as f64) * self.aperture_fraction).ceil() as usize
    }
}

impl Default for BeamformingConfig {
    fn default() -> Self {
        Self::new(
            vec![[0.0, 0.0, 0.0]],
            1.0e6,  // 1 MHz
            40.0e6, // 40 MHz sampling
            1540.0, // Tissue sound speed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_function_creation() {
        assert_eq!(WindowFunction::default(), WindowFunction::Hamming);
    }

    #[test]
    fn test_beamforming_config_creation() {
        let config = BeamformingConfig::new(
            vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]],
            1.0e6,
            40.0e6,
            1540.0,
        );

        assert_eq!(config.array_geometry.len(), 2);
        assert!((config.wavelength() - 1.54e-3).abs() < 1e-6);
    }

    #[test]
    fn test_beamforming_config_builder() {
        let config = BeamformingConfig::default()
            .with_focal_depth(0.05)
            .with_num_beams(128)
            .with_window(WindowFunction::Hann);

        assert_eq!(config.focal_depth, Some(0.05));
        assert_eq!(config.num_beams, 128);
        assert_eq!(config.window, WindowFunction::Hann);
    }

    #[test]
    fn test_window_function_apply() {
        let mut data = vec![1.0; 100];
        WindowFunction::Hamming.apply(&mut data);

        // First and last elements should be reduced
        assert!(data[0] < 1.0);
        assert!(data[99] < 1.0);
        // Middle should be closer to 1.0
        assert!(data[50] > 0.9);
    }

    #[test]
    fn test_array_aperture() {
        let config = BeamformingConfig::new(
            vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
            1.0e6,
            40.0e6,
            1540.0,
        );

        assert!((config.array_aperture() - 0.01).abs() < 1e-6);
    }
}
