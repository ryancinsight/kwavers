//! Filter operations for photoacoustic reconstruction
//!
//! This module provides various filtering operations used in
//! photoacoustic image reconstruction.

use crate::error::KwaversResult;
use crate::solver::reconstruction::FilterType;
use ndarray::{Array1, Array2, Array3};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

use super::config::PhotoacousticConfig;

/// Filter operations for photoacoustic reconstruction
pub struct Filters {
    filter_type: FilterType,
}

impl Filters {
    /// Create new filter operations
    pub fn new(config: &PhotoacousticConfig) -> Self {
        Self {
            filter_type: FilterType::RamLak, // Default filter type
        }
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
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        // Convert to complex
        let mut complex_signal: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Forward FFT
        fft.process(&mut complex_signal);

        // Apply bandpass filter
        let freq_resolution = sampling_freq / n as f64;
        for (i, val) in complex_signal.iter_mut().enumerate() {
            let freq = if i <= n / 2 {
                i as f64 * freq_resolution
            } else {
                (n - i) as f64 * freq_resolution
            };

            if freq < low_freq || freq > high_freq {
                *val = Complex::new(0.0, 0.0);
            }
        }

        // Inverse FFT
        ifft.process(&mut complex_signal);

        // Extract real part and normalize
        let norm_factor = 1.0 / n as f64;
        Ok(Array1::from_vec(
            complex_signal.iter().map(|c| c.re * norm_factor).collect(),
        ))
    }

    /// Apply envelope detection using Hilbert transform
    pub fn apply_envelope_detection(&self, data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (n_samples, n_sensors) = data.dim();
        let mut envelope = Array2::zeros((n_samples, n_sensors));

        for sensor_idx in 0..n_sensors {
            let signal = data.column(sensor_idx);
            let analytic = self.hilbert_transform_1d(signal.to_owned())?;

            for (i, val) in analytic.iter().enumerate() {
                envelope[[i, sensor_idx]] = val.norm();
            }
        }

        Ok(envelope)
    }

    /// Compute 1D Hilbert transform
    fn hilbert_transform_1d(&self, signal: Array1<f64>) -> KwaversResult<Vec<Complex<f64>>> {
        let n = signal.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        // Convert to complex
        let mut complex_signal: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Forward FFT
        fft.process(&mut complex_signal);

        // Apply Hilbert transform in frequency domain
        for i in 0..n {
            if i == 0 || i == n / 2 {
                // DC and Nyquist components unchanged
            } else if i < n / 2 {
                // Positive frequencies doubled
                complex_signal[i] *= 2.0;
            } else {
                // Negative frequencies zeroed
                complex_signal[i] = Complex::new(0.0, 0.0);
            }
        }

        // Inverse FFT
        ifft.process(&mut complex_signal);

        // Normalize
        let norm_factor = 1.0 / n as f64;
        for val in &mut complex_signal {
            *val *= norm_factor;
        }

        Ok(complex_signal)
    }

    /// Apply FBP filter to data
    pub fn apply_fbp_filter(&self, data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let mut filtered = data.clone();

        match self.filter_type {
            FilterType::RamLak => self.apply_ram_lak_filter(&mut filtered)?,
            FilterType::SheppLogan => self.apply_shepp_logan_filter(&mut filtered)?,
            FilterType::Cosine => self.apply_cosine_filter(&mut filtered)?,
            _ => {} // Other filter types not implemented
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

    /// Apply reconstruction filter for regularization
    pub fn apply_reconstruction_filter(&self, image: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        // Simple Gaussian smoothing for regularization
        // This is a placeholder - actual implementation would use proper regularization
        Ok(image.clone())
    }
}
