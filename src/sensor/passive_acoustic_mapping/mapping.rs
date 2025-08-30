//! PAM processing and mapping algorithms

use crate::error::KwaversResult;
use ndarray::Array3;
use rustfft::{num_complex::Complex, FftPlanner};

/// PAM configuration
#[derive(Debug, Clone))]
pub struct PAMConfig {
    pub beamforming: super::beamforming::BeamformingConfig,
    pub frequency_bands: Vec<(f64, f64)>,
    pub integration_time: f64,
    pub threshold: f64,
    pub enable_harmonic_analysis: bool,
    pub enable_broadband_analysis: bool,
}

/// PAM processor for cavitation mapping
#[derive(Debug))]
pub struct PAMProcessor {
    config: PAMConfig,
    fft_planner: FftPlanner<f64>,
}

impl PAMProcessor {
    /// Create a new PAM processor
    pub fn new(config: PAMConfig) -> KwaversResult<Self> {
        Ok(Self {
            config,
            fft_planner: FftPlanner::new(),
        })
    }

    /// Process beamformed data to extract cavitation map
    pub fn process(&mut self, beamformed_data: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let shape = beamformed_data.shape();
        let (nx, ny, nt) = (shape[0], shape[1], shape[2]);

        // Initialize output map
        let mut cavitation_map = Array3::zeros((nx, ny, self.config.frequency_bands.len()));

        // Process each spatial point
        for ix in 0..nx {
            for iy in 0..ny {
                // Extract time series
                let time_series: Vec<f64> =
                    (0..nt).map(|it| beamformed_data[[ix, iy, it]).collect();

                // Compute spectrum
                let spectrum = self.compute_spectrum(&time_series)?;

                // Analyze frequency bands
                for (band_idx, &(f_min, f_max)) in self.config.frequency_bands.iter().enumerate() {
                    let power = self.integrate_band_power(&spectrum, f_min, f_max);

                    // Apply threshold
                    if power > self.config.threshold {
                        cavitation_map[[ix, iy, band_idx] = power;
                    }
                }

                // Harmonic analysis if enabled
                if self.config.enable_harmonic_analysis {
                    self.analyze_harmonics(&spectrum, ix, iy, &mut cavitation_map)?;
                }
            }
        }

        Ok(cavitation_map)
    }

    /// Compute frequency spectrum
    fn compute_spectrum(&mut self, time_series: &[f64]) -> KwaversResult<Vec<f64>> {
        let n = time_series.len();
        let mut complex_data: Vec<Complex<f64>> =
            time_series.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Perform FFT
        let fft = self.fft_planner.plan_fft_forward(n);
        fft.process(&mut complex_data);

        // Compute power spectrum
        let spectrum: Vec<f64> = complex_data
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        Ok(spectrum)
    }

    /// Integrate power in frequency band
    fn integrate_band_power(&self, spectrum: &[f64], f_min: f64, f_max: f64) -> f64 {
        // Convert frequencies to indices (simplified)
        let n = spectrum.len();
        let idx_min = ((f_min / 1e6) * n as f64) as usize;
        let idx_max = ((f_max / 1e6) * n as f64) as usize;

        spectrum[idx_min.min(n - 1)..idx_max.min(n)].iter().sum()
    }

    /// Analyze harmonic content
    fn analyze_harmonics(
        &self,
        spectrum: &[f64],
        ix: usize,
        iy: usize,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        // Find fundamental frequency
        let fundamental_idx = self.find_fundamental(spectrum);

        // Check for harmonics
        for harmonic in 2..5 {
            let harmonic_idx = fundamental_idx * harmonic;
            if harmonic_idx < spectrum.len() {
                let harmonic_power = spectrum[harmonic_idx];
                if harmonic_power > self.config.threshold * 0.5 {
                    // Store harmonic information
                    if harmonic - 2 < output.shape()[2] {
                        output[[ix, iy, harmonic - 2] += harmonic_power;
                    }
                }
            }
        }

        Ok(())
    }

    /// Find fundamental frequency
    fn find_fundamental(&self, spectrum: &[f64]) -> usize {
        spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get current configuration
    pub fn config(&self) -> &PAMConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: PAMConfig) -> KwaversResult<()> {
        self.config = config;
        Ok(())
    }
}

impl Default for PAMConfig {
    fn default() -> Self {
        Self {
            beamforming: super::beamforming::BeamformingConfig::default(),
            frequency_bands: vec![
                (20e3, 100e3),  // Sub-harmonic
                (100e3, 500e3), // Fundamental
                (500e3, 2e6),   // Harmonic
                (2e6, 10e6),    // Ultra-harmonic
            ],
            integration_time: 0.1, // 100 ms
            threshold: 1e-6,
            enable_harmonic_analysis: true,
            enable_broadband_analysis: true,
        }
    }
}
