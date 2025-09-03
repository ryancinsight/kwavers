//! Modern FFT implementation using rustfft 6.2 with optimizations
//!
//! References:
//! - rustfft documentation: <https://docs.rs/rustfft/latest/rustfft>/
//! - "Numerical Recipes" by Press et al. (2007) for FFT algorithms

use ndarray::{s, Array2, Array3, ArrayView1, Axis, Zip};
use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// 3D FFT implementation with parallelization
pub struct Fft3d {
    planner: FftPlanner<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    // Cache FFT instances for reuse
    fft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,
    fft_z: Arc<dyn Fft<f64>>,
    ifft_x: Arc<dyn Fft<f64>>,
    ifft_y: Arc<dyn Fft<f64>>,
    ifft_z: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for Fft3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fft3d")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .finish()
    }
}

impl Fft3d {
    /// Create a new 3D FFT processor with cached plans
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let mut planner = FftPlanner::new();

        // Pre-plan all FFTs for efficiency
        let fft_x = planner.plan_fft_forward(nx);
        let fft_y = planner.plan_fft_forward(ny);
        let fft_z = planner.plan_fft_forward(nz);
        let ifft_x = planner.plan_fft_inverse(nx);
        let ifft_y = planner.plan_fft_inverse(ny);
        let ifft_z = planner.plan_fft_inverse(nz);

        Self {
            planner,
            nx,
            ny,
            nz,
            fft_x,
            fft_y,
            fft_z,
            ifft_x,
            ifft_y,
            ifft_z,
        }
    }

    /// Forward 3D FFT with parallel execution
    pub fn forward(&mut self, input: &Array3<f64>) -> Array3<Complex64> {
        self.transform_3d(input, true)
    }

    /// Inverse 3D FFT with parallel execution
    pub fn inverse(&mut self, input: &Array3<Complex64>) -> Array3<f64> {
        let mut data = input.clone();
        self.transform_3d_complex(&mut data, false);

        // Extract real part and normalize
        let norm = 1.0 / (self.nx * self.ny * self.nz) as f64;
        data.mapv(|c| c.re * norm)
    }

    /// Core 3D transform for real input
    fn transform_3d(&mut self, input: &Array3<f64>, forward: bool) -> Array3<Complex64> {
        // Convert to complex
        let mut data = input.mapv(|x| Complex64::new(x, 0.0));

        self.transform_3d_complex(&mut data, forward);
        data
    }

    /// Core 3D transform for complex data
    fn transform_3d_complex(
        &mut self,
        data: &Array3<Complex64>,
        forward: bool,
    ) -> Array3<Complex64> {
        let mut result = data.clone();

        // Transform along X axis (parallelized over Y-Z planes)
        let x_fft = if forward { &self.fft_x } else { &self.ifft_x };
        result
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut yz_slice| {
                yz_slice.axis_iter_mut(Axis(1)).for_each(|mut x_line| {
                    // Process FFT in-place without allocation
                    if let Some(slice) = x_line.as_slice_mut() {
                        x_fft.process(slice);
                    } else {
                        // Non-contiguous slice, need to copy
                        let mut temp: Vec<Complex64> = x_line.iter().cloned().collect();
                        x_fft.process(&mut temp);
                        x_line.iter_mut().zip(temp.iter()).for_each(|(dst, src)| *dst = *src);
                    }
                });
            });

        // Transform along Y axis (parallelized over X-Z planes)
        let y_fft = if forward { &self.fft_y } else { &self.ifft_y };
        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut xz_slice| {
                xz_slice.axis_iter_mut(Axis(1)).for_each(|mut y_line| {
                    // Process FFT in-place without allocation
                    if let Some(slice) = y_line.as_slice_mut() {
                        y_fft.process(slice);
                    } else {
                        // Non-contiguous slice, need to copy
                        let mut temp: Vec<Complex64> = y_line.iter().cloned().collect();
                        y_fft.process(&mut temp);
                        y_line.iter_mut().zip(temp.iter()).for_each(|(dst, src)| *dst = *src);
                    }
                });
            });

        // Transform along Z axis (parallelized over X-Y planes)
        let z_fft = if forward { &self.fft_z } else { &self.ifft_z };
        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut xy_slice| {
                xy_slice.axis_iter_mut(Axis(0)).for_each(|mut z_line| {
                    let mut buffer: Vec<Complex64> = z_line.to_vec();
                    z_fft.process(&mut buffer);
                    z_line.assign(&ArrayView1::from(&buffer));
                });
            });

        result
    }

    /// Apply spectral derivative (k-space multiplication)
    pub fn spectral_derivative(&mut self, field: &Array3<f64>, axis: usize) -> Array3<f64> {
        // Forward FFT
        let mut spectrum = self.forward(field);

        // Get k-values for the specified axis
        let k_values = match axis {
            0 => self.get_kx(),
            1 => self.get_ky(),
            2 => self.get_kz(),
            _ => panic!("Invalid axis"),
        };

        // Apply ik multiplication in k-space
        Zip::from(&mut spectrum)
            .and(&k_values)
            .par_for_each(|s, &k| {
                *s *= Complex64::new(0.0, k);
            });

        // Inverse FFT
        self.inverse(&spectrum)
    }

    /// Get k-space coordinates for X axis
    fn get_kx(&self) -> Array3<f64> {
        let mut kx = Array3::zeros((self.nx, self.ny, self.nz));
        let dk = 2.0 * std::f64::consts::PI / self.nx as f64;

        for i in 0..self.nx {
            let k = if i <= self.nx / 2 {
                i as f64 * dk
            } else {
                (i as f64 - self.nx as f64) * dk
            };
            kx.slice_mut(s![i, .., ..]).fill(k);
        }

        kx
    }

    /// Get k-space coordinates for Y axis
    fn get_ky(&self) -> Array3<f64> {
        let mut ky = Array3::zeros((self.nx, self.ny, self.nz));
        let dk = 2.0 * std::f64::consts::PI / self.ny as f64;

        for j in 0..self.ny {
            let k = if j <= self.ny / 2 {
                j as f64 * dk
            } else {
                (j as f64 - self.ny as f64) * dk
            };
            ky.slice_mut(s![.., j, ..]).fill(k);
        }

        ky
    }

    /// Get k-space coordinates for Z axis
    fn get_kz(&self) -> Array3<f64> {
        let mut kz = Array3::zeros((self.nx, self.ny, self.nz));
        let dk = 2.0 * std::f64::consts::PI / self.nz as f64;

        for k in 0..self.nz {
            let kval = if k <= self.nz / 2 {
                k as f64 * dk
            } else {
                (k as f64 - self.nz as f64) * dk
            };
            kz.slice_mut(s![.., .., k]).fill(kval);
        }

        kz
    }
}

/// 2D FFT processor for grid-based operations
pub struct Fft2d {
    planner: FftPlanner<f64>,
    nx: usize,
    ny: usize,
    fft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,
    ifft_x: Arc<dyn Fft<f64>>,
    ifft_y: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for Fft2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fft2d")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .finish()
    }
}

impl Fft2d {
    #[must_use]
    pub fn new(nx: usize, ny: usize) -> Self {
        let mut planner = FftPlanner::new();

        Self {
            nx,
            ny,
            fft_x: planner.plan_fft_forward(nx),
            fft_y: planner.plan_fft_forward(ny),
            ifft_x: planner.plan_fft_inverse(nx),
            ifft_y: planner.plan_fft_inverse(ny),
            planner,
        }
    }

    /// Forward 2D FFT
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<Complex64> {
        let mut data = input.mapv(|x| Complex64::new(x, 0.0));

        // Transform rows
        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                let mut buffer: Vec<Complex64> = row.to_vec();
                self.fft_x.process(&mut buffer);
                row.assign(&ArrayView1::from(&buffer));
            });

        // Transform columns
        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut col| {
                let mut buffer: Vec<Complex64> = col.to_vec();
                self.fft_y.process(&mut buffer);
                col.assign(&ArrayView1::from(&buffer));
            });

        data
    }

    /// Inverse 2D FFT
    pub fn inverse(&mut self, input: &Array2<Complex64>) -> Array2<f64> {
        let mut data = input.clone();

        // Transform columns
        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut col| {
                let mut buffer: Vec<Complex64> = col.to_vec();
                self.ifft_y.process(&mut buffer);
                col.assign(&ArrayView1::from(&buffer));
            });

        // Transform rows
        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                let mut buffer: Vec<Complex64> = row.to_vec();
                self.ifft_x.process(&mut buffer);
                row.assign(&ArrayView1::from(&buffer));
            });

        // Extract real part and normalize
        let norm = 1.0 / (self.nx * self.ny) as f64;
        data.mapv(|c| c.re * norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    #[test]
    fn test_fft_1d_forward_inverse() {
        let n = 64;
        let mut data: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect();

        let original = data.clone();

        // Forward FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut data);

        // Inverse FFT
        let ifft = planner.plan_fft_inverse(n);
        ifft.process(&mut data);

        // Normalize
        let norm = 1.0 / n as f64;
        data.iter_mut().for_each(|x| *x *= norm);

        // Check round-trip
        for (orig, result) in original.iter().zip(data.iter()) {
            assert_relative_eq!(orig.re, result.re, epsilon = 1e-10);
            assert_relative_eq!(orig.im, result.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_2d_gaussian() {
        let mut fft2d = Fft2d::new(64, 64);
        let mut data = Array2::zeros((64, 64));

        // Create Gaussian (real-valued)
        let sigma = 5.0;
        for i in 0..64 {
            for j in 0..64 {
                let x = (i as f64 - 32.0) / sigma;
                let y = (j as f64 - 32.0) / sigma;
                data[[i, j]] = (-0.5 * (x * x + y * y)).exp();
            }
        }

        let original = data.clone();

        // Forward and inverse transform
        let complex_data = fft2d.forward(&data);
        let reconstructed = fft2d.inverse(&complex_data);

        // Check reconstruction
        for ((i, j), &val) in reconstructed.indexed_iter() {
            assert_relative_eq!(val, original[[i, j]], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_3d_energy_conservation() {
        let mut fft3d = Fft3d::new(32, 32, 32);
        let mut data = Array3::zeros((32, 32, 32));

        // Create test signal (real-valued)
        let freq = 2.0 * PI / 32.0;
        for i in 0..32 {
            for j in 0..32 {
                for k in 0..32 {
                    data[[i, j, k]] = (freq * i as f64).cos() * (freq * j as f64).cos();
                }
            }
        }

        // Compute energy before
        let energy_before: f64 = data.iter().map(|x| x * x).sum();

        // FFT and IFFT
        let spectrum = fft3d.forward(&data);
        let reconstructed = fft3d.inverse(&spectrum);

        // Compute energy after
        let energy_after: f64 = reconstructed.iter().map(|x| x * x).sum();

        // Energy should be conserved (Parseval's theorem)
        assert_relative_eq!(energy_before, energy_after, epsilon = 1e-10 * energy_before);
    }
}
