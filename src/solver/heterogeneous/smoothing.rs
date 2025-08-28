//! Smoothing algorithms for interface treatment
//!
//! Implements various smoothing methods to mitigate Gibbs phenomenon

use super::config::SmoothingMethod;
use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::Array3;
use std::f64::consts::PI;

/// Smoothing processor for heterogeneous media
#[derive(Debug)]
pub struct Smoother {
    method: SmoothingMethod,
    width: f64,
    grid: Grid,
}

impl Smoother {
    /// Create a new smoother
    pub fn new(method: SmoothingMethod, width: f64, grid: Grid) -> Self {
        Self {
            method,
            width,
            grid,
        }
    }

    /// Apply smoothing to medium properties
    pub fn smooth(
        &self,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        interface_mask: &Array3<bool>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        match self.method {
            SmoothingMethod::None => Ok((density.clone(), sound_speed.clone())),
            SmoothingMethod::Gaussian => self.gaussian_smooth(density, sound_speed, interface_mask),
            SmoothingMethod::Tanh => self.tanh_smooth(density, sound_speed, interface_mask),
            SmoothingMethod::Polynomial => {
                self.polynomial_smooth(density, sound_speed, interface_mask)
            }
            SmoothingMethod::SpectralFilter => self.spectral_filter(density, sound_speed),
        }
    }

    /// Apply Gaussian smoothing
    fn gaussian_smooth(
        &self,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        interface_mask: &Array3<bool>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        let mut density_smooth = density.clone();
        let mut sound_speed_smooth = sound_speed.clone();

        let sigma = self.width;
        let kernel_size = (3.0 * sigma).ceil() as usize;
        let kernel = self.create_gaussian_kernel(sigma, kernel_size);

        // Apply smoothing only at interfaces
        for i in kernel_size..self.grid.nx - kernel_size {
            for j in kernel_size..self.grid.ny - kernel_size {
                for k in kernel_size..self.grid.nz - kernel_size {
                    if interface_mask[[i, j, k]] {
                        let (rho, c) =
                            self.convolve_3d(density, sound_speed, i, j, k, &kernel, kernel_size);
                        density_smooth[[i, j, k]] = rho;
                        sound_speed_smooth[[i, j, k]] = c;
                    }
                }
            }
        }

        Ok((density_smooth, sound_speed_smooth))
    }

    /// Apply hyperbolic tangent smoothing
    fn tanh_smooth(
        &self,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        interface_mask: &Array3<bool>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        let mut density_smooth = density.clone();
        let mut sound_speed_smooth = sound_speed.clone();

        for i in 1..self.grid.nx - 1 {
            for j in 1..self.grid.ny - 1 {
                for k in 1..self.grid.nz - 1 {
                    if interface_mask[[i, j, k]] {
                        // Use tanh transition between neighbors
                        let transition = |x: f64| 0.5 * (1.0 + (x / self.width).tanh());

                        let t = transition(0.0);

                        // Average with neighbors using tanh weighting
                        let rho_avg = self.weighted_average(density, i, j, k, t);
                        let c_avg = self.weighted_average(sound_speed, i, j, k, t);

                        density_smooth[[i, j, k]] = rho_avg;
                        sound_speed_smooth[[i, j, k]] = c_avg;
                    }
                }
            }
        }

        Ok((density_smooth, sound_speed_smooth))
    }

    /// Apply polynomial (cubic) smoothing
    fn polynomial_smooth(
        &self,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        interface_mask: &Array3<bool>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        let mut density_smooth = density.clone();
        let mut sound_speed_smooth = sound_speed.clone();

        for i in 2..self.grid.nx - 2 {
            for j in 2..self.grid.ny - 2 {
                for k in 2..self.grid.nz - 2 {
                    if interface_mask[[i, j, k]] {
                        // Cubic polynomial interpolation
                        let rho_smooth = self.cubic_interpolate(density, i, j, k);
                        let c_smooth = self.cubic_interpolate(sound_speed, i, j, k);

                        density_smooth[[i, j, k]] = rho_smooth;
                        sound_speed_smooth[[i, j, k]] = c_smooth;
                    }
                }
            }
        }

        Ok((density_smooth, sound_speed_smooth))
    }

    /// Apply spectral filtering
    fn spectral_filter(
        &self,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        // This would require FFT implementation
        // For now, return original (placeholder for proper implementation)
        Ok((density.clone(), sound_speed.clone()))
    }

    /// Create Gaussian kernel
    fn create_gaussian_kernel(&self, sigma: f64, size: usize) -> Vec<f64> {
        let mut kernel = Vec::with_capacity(size * size * size);
        let center = size as f64 / 2.0;
        let norm = 1.0 / ((2.0 * PI).sqrt() * sigma).powi(3);

        for i in 0..size {
            for j in 0..size {
                for k in 0..size {
                    let r2 = ((i as f64 - center).powi(2)
                        + (j as f64 - center).powi(2)
                        + (k as f64 - center).powi(2))
                        / (2.0 * sigma * sigma);
                    kernel.push(norm * (-r2).exp());
                }
            }
        }

        // Normalize
        let sum: f64 = kernel.iter().sum();
        kernel.iter_mut().for_each(|k| *k /= sum);

        kernel
    }

    /// 3D convolution at a point
    fn convolve_3d(
        &self,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        kernel: &[f64],
        kernel_size: usize,
    ) -> (f64, f64) {
        let mut rho_sum = 0.0;
        let mut c_sum = 0.0;
        let half = kernel_size / 2;

        for di in 0..kernel_size {
            for dj in 0..kernel_size {
                for dk in 0..kernel_size {
                    let ki = i + di - half;
                    let kj = j + dj - half;
                    let kk = k + dk - half;

                    let kernel_val = kernel[di * kernel_size * kernel_size + dj * kernel_size + dk];

                    rho_sum += density[[ki, kj, kk]] * kernel_val;
                    c_sum += sound_speed[[ki, kj, kk]] * kernel_val;
                }
            }
        }

        (rho_sum, c_sum)
    }

    /// Weighted average with neighbors
    fn weighted_average(
        &self,
        field: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        weight: f64,
    ) -> f64 {
        let center = field[[i, j, k]];
        let neighbors = [
            field[[i - 1, j, k]],
            field[[i + 1, j, k]],
            field[[i, j - 1, k]],
            field[[i, j + 1, k]],
            field[[i, j, k - 1]],
            field[[i, j, k + 1]],
        ];

        let neighbor_avg = neighbors.iter().sum::<f64>() / neighbors.len() as f64;
        center * weight + neighbor_avg * (1.0 - weight)
    }

    /// Cubic interpolation at a point
    fn cubic_interpolate(&self, field: &Array3<f64>, i: usize, j: usize, k: usize) -> f64 {
        // Cubic interpolation using surrounding points
        let points = [
            field[[i - 2, j, k]],
            field[[i - 1, j, k]],
            field[[i, j, k]],
            field[[i + 1, j, k]],
            field[[i + 2, j, k]],
        ];

        // Cubic polynomial coefficients (simplified)
        let a =
            (-points[0] + 4.0 * points[1] - 6.0 * points[2] + 4.0 * points[3] - points[4]) / 24.0;
        let b = (points[0] - 2.0 * points[1] + points[3]) / 2.0;
        let c = (-points[0] + 16.0 * points[1] - 30.0 * points[2] + 16.0 * points[3] - points[4])
            / 24.0;

        points[2] + a + b + c
    }
}
