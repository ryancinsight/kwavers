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

    /// Apply reconstruction filter for regularization and denoising
    pub fn apply_reconstruction_filter(&self, image: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = image.dim();
        let mut filtered = Array3::zeros((nx, ny, nz));

        // Apply 3D Gaussian filter for noise reduction
        // Using separable implementation for efficiency
        const GAUSSIAN_SIGMA: f64 = 1.0; // Standard deviation in voxels
        const KERNEL_RADIUS: usize = 3; // 7x7x7 kernel

        // Generate 1D Gaussian kernel
        let kernel = self.create_gaussian_kernel(KERNEL_RADIUS, GAUSSIAN_SIGMA);

        // Apply separable filtering in each dimension
        // First pass: filter along X
        let mut temp1 = Array3::zeros((nx, ny, nz));
        for j in 0..ny {
            for k in 0..nz {
                for i in 0..nx {
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;

                    for ki in 0..kernel.len() {
                        let ii = (i as i32 + ki as i32 - KERNEL_RADIUS as i32) as usize;
                        if ii < nx {
                            sum += image[[ii, j, k]] * kernel[ki];
                            weight_sum += kernel[ki];
                        }
                    }

                    if weight_sum > 0.0 {
                        temp1[[i, j, k]] = sum / weight_sum;
                    }
                }
            }
        }

        // Second pass: filter along Y
        let mut temp2 = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for k in 0..nz {
                for j in 0..ny {
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;

                    for kj in 0..kernel.len() {
                        let jj = (j as i32 + kj as i32 - KERNEL_RADIUS as i32) as usize;
                        if jj < ny {
                            sum += temp1[[i, jj, k]] * kernel[kj];
                            weight_sum += kernel[kj];
                        }
                    }

                    if weight_sum > 0.0 {
                        temp2[[i, j, k]] = sum / weight_sum;
                    }
                }
            }
        }

        // Third pass: filter along Z
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;

                    for kk in 0..kernel.len() {
                        let zz = (k as i32 + kk as i32 - KERNEL_RADIUS as i32) as usize;
                        if zz < nz {
                            sum += temp2[[i, j, zz]] * kernel[kk];
                            weight_sum += kernel[kk];
                        }
                    }

                    if weight_sum > 0.0 {
                        filtered[[i, j, k]] = sum / weight_sum;
                    }
                }
            }
        }

        // Apply edge-preserving bilateral filter for better feature preservation
        let bilateral_filtered = self.apply_bilateral_filter(&filtered, GAUSSIAN_SIGMA)?;

        Ok(bilateral_filtered)
    }

    /// Create 1D Gaussian kernel
    fn create_gaussian_kernel(&self, radius: usize, sigma: f64) -> Vec<f64> {
        let size = 2 * radius + 1;
        let mut kernel = vec![0.0; size];
        let norm = 1.0 / (sigma * (2.0 * PI).sqrt());
        let sigma2 = 2.0 * sigma * sigma;

        for i in 0..size {
            let x = f64::from(i as i32 - radius as i32);
            kernel[i] = norm * (-x * x / sigma2).exp();
        }

        // Normalize
        let sum: f64 = kernel.iter().sum();
        for val in &mut kernel {
            *val /= sum;
        }

        kernel
    }

    /// Apply bilateral filter for edge preservation
    fn apply_bilateral_filter(
        &self,
        image: &Array3<f64>,
        spatial_sigma: f64,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = image.dim();
        let mut filtered = image.clone();

        const WINDOW_RADIUS: usize = 2;
        const INTENSITY_SIGMA: f64 = 0.1; // Relative to data range

        // Estimate intensity range for normalization
        let max_val = image.iter().copied().fold(0.0_f64, f64::max);
        let min_val = image.iter().copied().fold(f64::INFINITY, f64::min);
        let range = (max_val - min_val).max(1e-10);
        let intensity_sigma = INTENSITY_SIGMA * range;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let center_val = image[[i, j, k]];
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;

                    // Apply bilateral filter in local window
                    for di in -(WINDOW_RADIUS as i32)..=(WINDOW_RADIUS as i32) {
                        for dj in -(WINDOW_RADIUS as i32)..=(WINDOW_RADIUS as i32) {
                            for dk in -(WINDOW_RADIUS as i32)..=(WINDOW_RADIUS as i32) {
                                let ii = (i as i32 + di) as usize;
                                let jj = (j as i32 + dj) as usize;
                                let kk = (k as i32 + dk) as usize;

                                if ii < nx && jj < ny && kk < nz {
                                    let neighbor_val = image[[ii, jj, kk]];

                                    // Spatial weight
                                    let spatial_dist2 = f64::from(di * di + dj * dj + dk * dk);
                                    let spatial_weight = (-spatial_dist2
                                        / (2.0 * spatial_sigma * spatial_sigma))
                                        .exp();

                                    // Intensity weight
                                    let intensity_diff = neighbor_val - center_val;
                                    let intensity_weight = (-(intensity_diff * intensity_diff)
                                        / (2.0 * intensity_sigma * intensity_sigma))
                                        .exp();

                                    let weight = spatial_weight * intensity_weight;
                                    sum += neighbor_val * weight;
                                    weight_sum += weight;
                                }
                            }
                        }
                    }

                    if weight_sum > 0.0 {
                        filtered[[i, j, k]] = sum / weight_sum;
                    }
                }
            }
        }

        Ok(filtered)
    }
}
