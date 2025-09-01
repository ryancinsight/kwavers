//! Fourier domain reconstruction for photoacoustic imaging
//!
//! This module implements Fourier-domain reconstruction algorithms
//! based on the Fourier slice theorem and k-space methods.
//!
//! References:
//! - Norton (1980) "Reconstruction from projections"
//! - Kostli et al. (2001) "Temporal backward projection of optoacoustic pressure"

use crate::error::KwaversResult;
use ndarray::{Array1, Array2, Array3, ArrayView2};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

/// Fourier domain reconstruction algorithm
#[derive(Debug)]
pub struct FourierReconstructor {
    grid_size: [usize; 3],
    sound_speed: f64,
    sampling_frequency: f64,
}

impl FourierReconstructor {
    /// Create new Fourier domain reconstructor
    pub fn new(grid_size: [usize; 3], sound_speed: f64, sampling_frequency: f64) -> Self {
        Self {
            grid_size,
            sound_speed,
            sampling_frequency,
        }
    }

    /// Perform Fourier domain reconstruction using the projection theorem
    ///
    /// This method reconstructs the initial pressure distribution from
    /// sensor data using the Fourier slice theorem, which states that
    /// the Fourier transform of a projection equals a slice through
    /// the Fourier transform of the object.
    pub fn reconstruct(
        &self,
        sensor_data: ArrayView2<f64>,
        sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<Array3<f64>> {
        let (n_time, n_sensors) = sensor_data.dim();
        let [nx, ny, nz] = self.grid_size;

        // Initialize k-space representation
        let mut k_space = Array3::zeros((nx, ny, nz));

        // Process each sensor
        for (sensor_idx, sensor_pos) in sensor_positions.iter().enumerate() {
            // Get sensor signal
            let signal = sensor_data.column(sensor_idx);

            // Apply ramp filter for derivative (d/dt -> multiplication by iω in Fourier)
            let filtered_signal = self.apply_ramp_filter(signal.to_owned())?;

            // Compute angular spectrum for this sensor
            let angular_spectrum = self.compute_angular_spectrum(&filtered_signal, sensor_pos)?;

            // Add to k-space using projection theorem
            self.add_to_k_space(&mut k_space, &angular_spectrum, sensor_pos)?;
        }

        // Apply k-space filter for regularization
        self.apply_k_space_filter(&mut k_space);

        // Inverse Fourier transform to get spatial image
        self.inverse_fourier_transform(&k_space)
    }

    /// Apply ramp filter (derivative in time domain)
    fn apply_ramp_filter(&self, signal: Array1<f64>) -> KwaversResult<Array1<f64>> {
        let n = signal.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        // Convert to complex
        let mut complex_signal: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Forward FFT
        fft.process(&mut complex_signal);

        // Apply ramp filter: H(ω) = iω
        let df = self.sampling_frequency / n as f64;
        for (i, val) in complex_signal.iter_mut().enumerate() {
            let freq = if i <= n / 2 {
                i as f64 * df
            } else {
                (i as i32 - n as i32) as f64 * df
            };

            // Multiply by iω (derivative in frequency domain)
            let omega = 2.0 * PI * freq;
            *val *= Complex::new(0.0, omega);
        }

        // Inverse FFT
        ifft.process(&mut complex_signal);

        // Extract real part and normalize
        let norm = 1.0 / n as f64;
        Ok(Array1::from_vec(
            complex_signal.iter().map(|c| c.re * norm).collect(),
        ))
    }

    /// Compute angular spectrum from filtered sensor data
    fn compute_angular_spectrum(
        &self,
        filtered_signal: &Array1<f64>,
        sensor_pos: &[f64; 3],
    ) -> KwaversResult<Array2<Complex<f64>>> {
        let n_time = filtered_signal.len();
        let n_freq = n_time / 2 + 1;

        // FFT of filtered signal
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_time);

        let mut complex_signal: Vec<Complex<f64>> = filtered_signal
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut complex_signal);

        // Create angular spectrum (frequency vs angle)
        let n_angles = 180; // Angular resolution
        let mut angular_spectrum = Array2::zeros((n_freq, n_angles));

        // Map frequency components to angular components
        for f_idx in 0..n_freq {
            let freq = f_idx as f64 * self.sampling_frequency / n_time as f64;
            let k = 2.0 * PI * freq / self.sound_speed; // Wave number

            for angle_idx in 0..n_angles {
                let theta = angle_idx as f64 * PI / n_angles as f64;

                // Project onto this angle
                let projection_factor = Complex::new(
                    (k * sensor_pos[0] * theta.cos()).cos(),
                    (k * sensor_pos[0] * theta.cos()).sin(),
                );

                angular_spectrum[[f_idx, angle_idx]] = complex_signal[f_idx] * projection_factor;
            }
        }

        Ok(angular_spectrum)
    }

    /// Add angular spectrum to k-space representation
    fn add_to_k_space(
        &self,
        k_space: &mut Array3<Complex<f64>>,
        angular_spectrum: &Array2<Complex<f64>>,
        sensor_pos: &[f64; 3],
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = self.grid_size;
        let (n_freq, n_angles) = angular_spectrum.dim();

        // Map angular spectrum to k-space using projection theorem
        for f_idx in 0..n_freq {
            let freq = f_idx as f64 * self.sampling_frequency / (2.0 * n_freq as f64);
            let k_mag = 2.0 * PI * freq / self.sound_speed;

            for angle_idx in 0..n_angles {
                let theta = angle_idx as f64 * PI / n_angles as f64;
                let phi = sensor_pos[1].atan2(sensor_pos[0]); // Azimuthal angle

                // Convert spherical to Cartesian k-space coordinates
                let kx = k_mag * theta.sin() * phi.cos();
                let ky = k_mag * theta.sin() * phi.sin();
                let kz = k_mag * theta.cos();

                // Find nearest k-space grid point
                let ix = ((kx / (2.0 * PI) + 0.5) * nx as f64) as usize % nx;
                let iy = ((ky / (2.0 * PI) + 0.5) * ny as f64) as usize % ny;
                let iz = ((kz / (2.0 * PI) + 0.5) * nz as f64) as usize % nz;

                // Add contribution (with proper weighting for spherical integration)
                let weight = theta.sin(); // Jacobian for spherical coordinates
                k_space[[ix, iy, iz]] += angular_spectrum[[f_idx, angle_idx]] * weight;
            }
        }

        Ok(())
    }

    /// Apply k-space filter for noise suppression and regularization
    fn apply_k_space_filter(&self, k_space: &mut Array3<Complex<f64>>) {
        let [nx, ny, nz] = self.grid_size;

        // Apply Hann window for smooth roll-off
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let kx = (ix as f64 - nx as f64 / 2.0) / nx as f64;
                    let ky = (iy as f64 - ny as f64 / 2.0) / ny as f64;
                    let kz = (iz as f64 - nz as f64 / 2.0) / nz as f64;

                    let k_norm = (kx * kx + ky * ky + kz * kz).sqrt();

                    // Hann window
                    let window = if k_norm < 0.5 {
                        0.5 * (1.0 + (2.0 * PI * k_norm).cos())
                    } else {
                        0.0
                    };

                    k_space[[ix, iy, iz]] *= window;
                }
            }
        }
    }

    /// Inverse Fourier transform to get spatial image
    fn inverse_fourier_transform(
        &self,
        k_space: &Array3<Complex<f64>>,
    ) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = self.grid_size;
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(nx * ny * nz);

        // Flatten k-space data
        let mut complex_data: Vec<Complex<f64>> = Vec::with_capacity(nx * ny * nz);
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    complex_data.push(k_space[[ix, iy, iz]]);
                }
            }
        }

        // Inverse FFT
        ifft.process(&mut complex_data);

        // Extract real part and reshape
        let norm = 1.0 / (nx * ny * nz) as f64;
        let mut result = Array3::zeros((nx, ny, nz));
        for (idx, val) in complex_data.iter().enumerate() {
            let iz = idx % nz;
            let iy = (idx / nz) % ny;
            let ix = idx / (ny * nz);
            result[[ix, iy, iz]] = val.re * norm;
        }

        // Apply positivity constraint (pressure should be non-negative)
        result.mapv_inplace(|x| x.max(0.0));

        Ok(result)
    }
}
