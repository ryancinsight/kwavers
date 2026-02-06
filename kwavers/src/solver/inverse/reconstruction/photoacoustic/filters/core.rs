//! Core filter operations for photoacoustic reconstruction
//!
//! This module provides the main Filters struct and frequency-domain
//! filtering operations (bandpass, envelope detection, FBP filters).

use super::spatial;
use crate::core::error::KwaversResult;
use crate::domain::signal::{analytic, window_value, WindowType};
use crate::solver::reconstruction::FilterType;
use ndarray::{Array1, Array2, Array3};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

use crate::solver::reconstruction::photoacoustic::config::PhotoacousticConfig;

/// Filter operations for photoacoustic reconstruction
#[derive(Debug)]
pub struct Filters {
    filter_type: FilterType,
}

impl Filters {
    /// Create new filter operations
    pub fn new(_config: &PhotoacousticConfig) -> Self {
        Self {
            filter_type: FilterType::RamLak, // Default filter type
        }
    }

    /// Set filter type
    ///
    /// # Note
    /// This method is exposed for testing. Prefer setting filter type through configuration.
    #[doc(hidden)]
    pub fn set_filter_type(&mut self, filter_type: FilterType) {
        self.filter_type = filter_type;
    }

    /// Apply bandpass filter to sensor data
    pub fn apply_bandpass_filter(
        &self,
        data: &Array2<f64>,
        bandpass: [f64; 2],
        sampling_frequency: f64,
    ) -> KwaversResult<Array2<f64>> {
        let (n_samples, n_sensors) = data.dim();
        let mut filtered = Array2::zeros((n_samples, n_sensors));

        for sensor_idx in 0..n_sensors {
            let sensor_signal = data.column(sensor_idx);
            let filtered_signal = self.bandpass_filter_1d(
                sensor_signal.to_owned(),
                bandpass[0],
                bandpass[1],
                sampling_frequency,
            )?;

            for (i, val) in filtered_signal.iter().enumerate() {
                filtered[[i, sensor_idx]] = *val;
            }
        }

        Ok(filtered)
    }

    /// Apply 1D bandpass filter using FFT
    fn bandpass_filter_1d(
        &self,
        signal: Array1<f64>,
        low_freq: f64,
        high_freq: f64,
        sampling_freq: f64,
    ) -> KwaversResult<Array1<f64>> {
        let n = signal.len();
        if n == 0 {
            return Ok(signal);
        }

        let df = sampling_freq / n as f64;
        let mut filter = Array1::zeros(n);
        for i in 0..n {
            let freq = if i <= n / 2 {
                i as f64 * df
            } else {
                (n - i) as f64 * df
            };
            if freq >= low_freq && freq <= high_freq {
                filter[i] = 1.0;
            }
        }

        self.apply_filter_1d(signal, &filter)
    }

    /// Apply envelope detection using Hilbert transform
    pub fn apply_envelope_detection(&self, data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (n_samples, n_sensors) = data.dim();
        let mut envelope = Array2::zeros((n_samples, n_sensors));

        for sensor_idx in 0..n_sensors {
            let signal = data.column(sensor_idx);
            let analytic_signal = analytic::hilbert_transform(&signal.to_owned());

            for (i, val) in analytic_signal.iter().enumerate() {
                envelope[[i, sensor_idx]] = val.norm();
            }
        }

        Ok(envelope)
    }

    /// Apply FBP filter to data
    pub fn apply_fbp_filter(&self, data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let mut filtered = data.clone();

        match self.filter_type {
            FilterType::RamLak => self.apply_ram_lak_filter(&mut filtered)?,
            FilterType::SheppLogan => self.apply_shepp_logan_filter(&mut filtered)?,
            FilterType::Cosine => self.apply_cosine_filter(&mut filtered)?,
            FilterType::Hamming => self.apply_hamming_filter(&mut filtered)?,
            FilterType::Hann => self.apply_hann_filter(&mut filtered)?,
            FilterType::None => {} // No filtering applied
        }

        Ok(filtered)
    }

    /// Apply Ram-Lak filter
    fn apply_ram_lak_filter(&self, data: &mut Array2<f64>) -> KwaversResult<()> {
        let (n_samples, _) = data.dim();
        let filter = self.create_ram_lak_filter(n_samples);

        for mut col in data.columns_mut() {
            let filtered = self.apply_filter_1d(col.to_owned(), &filter)?;
            col.assign(&filtered);
        }

        Ok(())
    }

    /// Apply Shepp-Logan filter
    fn apply_shepp_logan_filter(&self, data: &mut Array2<f64>) -> KwaversResult<()> {
        let (n_samples, _) = data.dim();
        let filter = self.create_shepp_logan_filter(n_samples);

        for mut col in data.columns_mut() {
            let filtered = self.apply_filter_1d(col.to_owned(), &filter)?;
            col.assign(&filtered);
        }

        Ok(())
    }

    /// Apply Cosine filter
    fn apply_cosine_filter(&self, data: &mut Array2<f64>) -> KwaversResult<()> {
        let (n_samples, _) = data.dim();
        let filter = self.create_cosine_filter(n_samples);

        for mut col in data.columns_mut() {
            let filtered = self.apply_filter_1d(col.to_owned(), &filter)?;
            col.assign(&filtered);
        }

        Ok(())
    }

    /// Create Ram-Lak filter
    fn create_ram_lak_filter(&self, n: usize) -> Array1<f64> {
        let mut filter = Array1::zeros(n);
        for i in 0..n {
            let freq = i as f64 / n as f64;
            filter[i] = if i <= n / 2 { freq } else { 1.0 - freq };
        }
        filter
    }

    /// Create Shepp-Logan filter
    fn create_shepp_logan_filter(&self, n: usize) -> Array1<f64> {
        let mut filter = Array1::zeros(n);
        for i in 0..n {
            let freq = i as f64 / n as f64;
            let ram_lak = if i <= n / 2 { freq } else { 1.0 - freq };
            filter[i] = ram_lak * (PI * freq).sin() / (PI * freq).max(1e-10);
        }
        filter
    }

    /// Create Cosine filter
    fn create_cosine_filter(&self, n: usize) -> Array1<f64> {
        let mut filter = Array1::zeros(n);
        for i in 0..n {
            let freq = i as f64 / n as f64;
            let ram_lak = if i <= n / 2 { freq } else { 1.0 - freq };
            filter[i] = ram_lak * (PI * freq / 2.0).cos();
        }
        filter
    }

    /// Apply Hamming filter
    ///
    /// Hamming window: w(n) = 0.54 - 0.46*cos(2πn/(N-1))
    /// Applied to Ram-Lak filter for improved noise reduction
    fn apply_hamming_filter(&self, data: &mut Array2<f64>) -> KwaversResult<()> {
        let (n_samples, _) = data.dim();
        let filter = self.create_hamming_filter(n_samples);

        for mut col in data.columns_mut() {
            let filtered = self.apply_filter_1d(col.to_owned(), &filter)?;
            col.assign(&filtered);
        }

        Ok(())
    }

    /// Apply Hann filter
    ///
    /// Hann window: w(n) = 0.5 * (1 - cos(2πn/(N-1)))
    /// Applied to Ram-Lak filter for smooth frequency response
    fn apply_hann_filter(&self, data: &mut Array2<f64>) -> KwaversResult<()> {
        let (n_samples, _) = data.dim();
        let filter = self.create_hann_filter(n_samples);

        for mut col in data.columns_mut() {
            let filtered = self.apply_filter_1d(col.to_owned(), &filter)?;
            col.assign(&filtered);
        }

        Ok(())
    }

    /// Create Hamming filter
    ///
    /// Combines Ram-Lak filter with Hamming window for improved sidelobe suppression
    /// Literature: Hamming, R. W. (1989). Digital Filters, 3rd ed.
    ///
    /// # Note
    /// This method is exposed as `pub` for integration testing. Not part of stable public API.
    #[doc(hidden)]
    pub fn create_hamming_filter(&self, n: usize) -> Array1<f64> {
        let mut filter = Array1::zeros(n);
        for i in 0..n {
            let freq = i as f64 / n as f64;
            let ram_lak = if i <= n / 2 { freq } else { 1.0 - freq };
            let hamming_window = if n > 1 {
                window_value(WindowType::Hamming, i as f64 / (n - 1) as f64)
            } else {
                1.0
            };
            filter[i] = ram_lak * hamming_window;
        }
        filter
    }

    /// Create Hann filter
    ///
    /// Combines Ram-Lak filter with Hann window for smooth frequency rolloff
    /// Literature: Blackman, R. B., & Tukey, J. W. (1958). The Measurement of Power Spectra
    ///
    /// # Note
    /// This method is exposed as `pub` for integration testing. Not part of stable public API.
    #[doc(hidden)]
    pub fn create_hann_filter(&self, n: usize) -> Array1<f64> {
        let mut filter = Array1::zeros(n);
        for i in 0..n {
            let freq = i as f64 / n as f64;
            let ram_lak = if i <= n / 2 { freq } else { 1.0 - freq };
            let hann_window = if n > 1 {
                window_value(WindowType::Hann, i as f64 / (n - 1) as f64)
            } else {
                1.0
            };
            filter[i] = ram_lak * hann_window;
        }
        filter
    }

    /// Apply filter in frequency domain
    fn apply_filter_1d(
        &self,
        signal: Array1<f64>,
        filter: &Array1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let n = signal.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        // Convert to complex
        let mut complex_signal: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Forward FFT
        fft.process(&mut complex_signal);

        // Apply filter
        for (i, val) in complex_signal.iter_mut().enumerate() {
            *val *= filter[i];
        }

        // Inverse FFT
        ifft.process(&mut complex_signal);

        // Extract real part and normalize
        let norm_factor = 1.0 / n as f64;
        Ok(Array1::from_vec(
            complex_signal.iter().map(|c| c.re * norm_factor).collect(),
        ))
    }

    /// Apply reconstruction filter for regularization and denoising
    pub fn apply_reconstruction_filter(&self, image: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        // Apply 3D Gaussian filter for noise reduction
        const GAUSSIAN_SIGMA: f64 = 1.0; // Standard deviation in voxels
        const KERNEL_RADIUS: usize = 3; // 7x7x7 kernel
        const WINDOW_RADIUS: usize = 2;
        const INTENSITY_SIGMA: f64 = 0.1; // Relative to data range

        // Apply Gaussian filtering using separable implementation
        let gaussian_filtered =
            spatial::apply_gaussian_filter(image, GAUSSIAN_SIGMA, KERNEL_RADIUS)?;

        // Apply edge-preserving bilateral filter for better feature preservation
        let bilateral_filtered = spatial::apply_bilateral_filter(
            &gaussian_filtered,
            GAUSSIAN_SIGMA,
            WINDOW_RADIUS,
            INTENSITY_SIGMA,
        )?;

        Ok(bilateral_filtered)
    }
}
