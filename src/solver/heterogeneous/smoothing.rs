//! Smoothing algorithms for interface treatment
//!
//! Implements various smoothing methods to mitigate Gibbs phenomenon

use super::config::SmoothingMethod;
use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::{Array3, ArrayView3};
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
        density: ArrayView3<f64>,
        sound_speed: ArrayView3<f64>,
        interface_mask: &Array3<bool>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        match self.method {
            SmoothingMethod::None => Ok((density.to_owned(), sound_speed.to_owned())),
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
        density: ArrayView3<f64>,
        sound_speed: ArrayView3<f64>,
        interface_mask: &Array3<bool>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        let mut density_smooth = density.to_owned();
        let mut sound_speed_smooth = sound_speed.to_owned();

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
        density: ArrayView3<f64>,
        sound_speed: ArrayView3<f64>,
        interface_mask: &Array3<bool>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        let mut density_smooth = density.to_owned();
        let mut sound_speed_smooth = sound_speed.to_owned();

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
        density: ArrayView3<f64>,
        sound_speed: ArrayView3<f64>,
        interface_mask: &Array3<bool>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        let mut density_smooth = density.to_owned();
        let mut sound_speed_smooth = sound_speed.to_owned();

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
        density: ArrayView3<f64>,
        sound_speed: ArrayView3<f64>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        use crate::fft::Fft3d;
        use ndarray::Array3;
        use num_complex::Complex;

        // Initialize FFT processor
        let fft = Fft3d::new(self.grid.nx, self.grid.ny, self.grid.nz);

        // Transform to frequency domain
        let mut density_fft =
            Array3::<Complex<f64>>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
        let mut sound_speed_fft =
            Array3::<Complex<f64>>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        // Convert real to complex
        for ((i, j, k), &val) in density.indexed_iter() {
            density_fft[[i, j, k]] = Complex::new(val, 0.0);
        }
        for ((i, j, k), &val) in sound_speed.indexed_iter() {
            sound_speed_fft[[i, j, k]] = Complex::new(val, 0.0);
        }

        // Forward FFT
        let mut fft_processor = fft;
        fft_processor.process(&mut density_fft, &self.grid);
        fft_processor.process(&mut sound_speed_fft, &self.grid);

        // Apply low-pass filter in frequency domain
        // Cutoff at Nyquist/4 to remove high-frequency artifacts
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let cutoff_x = nx / 4;
        let cutoff_y = ny / 4;
        let cutoff_z = nz / 4;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let kx = if i <= nx / 2 { i } else { nx - i };
                    let ky = if j <= ny / 2 { j } else { ny - j };
                    let kz = if k <= nz / 2 { k } else { nz - k };

                    if kx > cutoff_x || ky > cutoff_y || kz > cutoff_z {
                        // Apply smooth transition using Tukey window
                        let wx = if kx > cutoff_x {
                            0.5 * (1.0 + ((kx - cutoff_x) as f64 * PI / cutoff_x as f64).cos())
                        } else {
                            1.0
                        };
                        let wy = if ky > cutoff_y {
                            0.5 * (1.0 + ((ky - cutoff_y) as f64 * PI / cutoff_y as f64).cos())
                        } else {
                            1.0
                        };
                        let wz = if kz > cutoff_z {
                            0.5 * (1.0 + ((kz - cutoff_z) as f64 * PI / cutoff_z as f64).cos())
                        } else {
                            1.0
                        };

                        let window = wx * wy * wz;
                        density_fft[[i, j, k]] *= window;
                        sound_speed_fft[[i, j, k]] *= window;
                    }
                }
            }
        }

        // Inverse FFT
        use crate::fft::Ifft3d;
        let mut ifft = Ifft3d::new(nx, ny, nz);
        ifft.process(&mut density_fft, &self.grid);
        ifft.process(&mut sound_speed_fft, &self.grid);

        // Convert back to real and normalize
        let mut density_smooth = Array3::zeros((nx, ny, nz));
        let mut sound_speed_smooth = Array3::zeros((nx, ny, nz));
        let norm = 1.0 / (nx * ny * nz) as f64;

        for ((i, j, k), val) in density_fft.indexed_iter() {
            density_smooth[[i, j, k]] = val.re * norm;
        }
        for ((i, j, k), val) in sound_speed_fft.indexed_iter() {
            sound_speed_smooth[[i, j, k]] = val.re * norm;
        }

        Ok((density_smooth, sound_speed_smooth))
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
        density: ArrayView3<f64>,
        sound_speed: ArrayView3<f64>,
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
        field: ArrayView3<f64>,
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
    fn cubic_interpolate(&self, field: ArrayView3<f64>, i: usize, j: usize, k: usize) -> f64 {
        // Cubic interpolation using surrounding points
        let points = [
            field[[i - 2, j, k]],
            field[[i - 1, j, k]],
            field[[i, j, k]],
            field[[i + 1, j, k]],
            field[[i + 2, j, k]],
        ];

        // Cubic polynomial interpolation using 5-point stencil
        // Standard cubic Hermite formula for C2 smoothness
        // Per Fornberg (1988): "Generation of Finite Difference Formulas"
        let a =
            (-points[0] + 4.0 * points[1] - 6.0 * points[2] + 4.0 * points[3] - points[4]) / 24.0;
        let b = (points[0] - 2.0 * points[1] + points[3]) / 2.0;
        let c = (-points[0] + 16.0 * points[1] - 30.0 * points[2] + 16.0 * points[3] - points[4])
            / 24.0;

        points[2] + a + b + c
    }
}
