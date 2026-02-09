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

use crate::core::error::{KwaversError, ValidationError};

/// 3D FFT implementation with parallelization
pub struct Fft3d {
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
    pub fn forward(&self, input: &Array3<f64>) -> Array3<Complex64> {
        self.transform_3d(input, true)
    }

    /// Inverse 3D FFT with parallel execution
    pub fn inverse(&self, input: &Array3<Complex64>) -> Array3<f64> {
        let mut output = Array3::zeros((self.nx, self.ny, self.nz));
        let mut scratch = Array3::zeros((self.nx, self.ny, self.nz));
        self.inverse_into(input, &mut output, &mut scratch);
        output
    }

    /// Forward 3D FFT with complex input
    pub fn forward_complex(&self, input: &Array3<Complex64>) -> Array3<Complex64> {
        self.transform_3d_complex(input, true)
    }

    /// Inverse 3D FFT with complex input
    pub fn inverse_complex(&self, input: &Array3<Complex64>) -> Array3<Complex64> {
        self.transform_3d_complex(input, false)
    }

    pub fn forward_into(&self, input: &Array3<f64>, output: &mut Array3<Complex64>) {
        assert_eq!(
            input.dim(),
            (self.nx, self.ny, self.nz),
            "input shape mismatch"
        );
        assert_eq!(
            output.dim(),
            (self.nx, self.ny, self.nz),
            "output shape mismatch"
        );

        Zip::from(&mut *output).and(input).for_each(|out, &val| {
            *out = Complex64::new(val, 0.0);
        });

        self.transform_3d_complex_inplace(output, true);
    }

    pub fn inverse_into(
        &self,
        input: &Array3<Complex64>,
        output: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    ) {
        assert_eq!(input.dim(), (self.nx, self.ny, self.nz));
        assert_eq!(output.dim(), (self.nx, self.ny, self.nz));
        assert_eq!(scratch.dim(), (self.nx, self.ny, self.nz));

        scratch.assign(input);
        self.transform_3d_complex_inplace(scratch, false);

        let norm = 1.0 / (self.nx * self.ny * self.nz) as f64;
        Zip::from(output).and(scratch).for_each(|out, val| {
            *out = val.re * norm;
        });
    }

    /// Forward 3D FFT in-place (complex input/output)
    pub fn forward_complex_inplace(&self, data: &mut Array3<Complex64>) {
        self.transform_3d_complex_inplace(data, true);
    }

    /// Inverse 3D FFT in-place (complex input/output)
    /// Note: Result is unnormalized. Caller must scale by 1/N.
    pub fn inverse_complex_inplace(&self, data: &mut Array3<Complex64>) {
        self.transform_3d_complex_inplace(data, false);
    }

    /// Core 3D transform for real input
    fn transform_3d(&self, input: &Array3<f64>, forward: bool) -> Array3<Complex64> {
        // Convert to complex
        let data = input.mapv(|x| Complex64::new(x, 0.0));

        self.transform_3d_complex(&data, forward)
    }

    /// Core 3D transform for complex data
    fn transform_3d_complex(&self, data: &Array3<Complex64>, forward: bool) -> Array3<Complex64> {
        let mut result = data.clone();
        self.transform_3d_complex_inplace(&mut result, forward);
        result
    }

    fn transform_3d_complex_inplace(&self, data: &mut Array3<Complex64>, forward: bool) {
        let fft_x = if forward {
            Arc::clone(&self.fft_x)
        } else {
            Arc::clone(&self.ifft_x)
        };
        let fft_y = if forward {
            Arc::clone(&self.fft_y)
        } else {
            Arc::clone(&self.ifft_y)
        };
        let fft_z = if forward {
            Arc::clone(&self.fft_z)
        } else {
            Arc::clone(&self.ifft_z)
        };

        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;

        // Transform along Z axis first (contiguous in memory for C-order arrays)
        // Each row along z is contiguous, so process in-place via slicing
        let raw = data.as_slice_mut().expect("Array3 must be contiguous");
        raw.par_chunks_mut(nz).for_each(|chunk| {
            fft_z.process(chunk);
        });

        // Transform along Y axis (axis 1) - parallelized over X dimension
        // Each x-slice (shape ny × nz) is independent
        {
            data.axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut slice| {
                    let mut line = vec![Complex64::default(); ny];
                    for k in 0..nz {
                        for j in 0..ny {
                            line[j] = slice[[j, k]];
                        }
                        fft_y.process(&mut line);
                        for j in 0..ny {
                            slice[[j, k]] = line[j];
                        }
                    }
                });
        }

        // Transform along X axis (axis 0) - parallelized over Y dimension
        // Each y-slice (shape nx × nz) is independent
        {
            data.axis_iter_mut(Axis(1))
                .into_par_iter()
                .for_each(|mut slice| {
                    let mut line = vec![Complex64::default(); nx];
                    for k in 0..nz {
                        for i in 0..nx {
                            line[i] = slice[[i, k]];
                        }
                        fft_x.process(&mut line);
                        for i in 0..nx {
                            slice[[i, k]] = line[i];
                        }
                    }
                });
        }
    }

    /// Apply spectral derivative (k-space multiplication)
    pub fn spectral_derivative(
        &self,
        field: &Array3<f64>,
        axis: usize,
    ) -> Result<Array3<f64>, KwaversError> {
        let mut spectrum = self.forward(field);
        let k_values = match axis {
            0 => self.get_kx(),
            1 => self.get_ky(),
            2 => self.get_kz(),
            _ => {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "axis".to_string(),
                    value: axis.to_string(),
                    constraint: "Axis must be 0, 1, or 2".to_string(),
                }))
            }
        };

        Zip::from(&mut spectrum)
            .and(&k_values)
            .par_for_each(|s, &k| {
                *s *= Complex64::new(0.0, k);
            });

        Ok(self.inverse(&spectrum))
    }

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

/// 1D FFT implementation
pub struct Fft1d {
    n: usize,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for Fft1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fft1d").field("n", &self.n).finish()
    }
}

impl Fft1d {
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            n,
            fft: planner.plan_fft_forward(n),
            ifft: planner.plan_fft_inverse(n),
        }
    }

    pub fn forward(&self, input: &ndarray::Array1<f64>) -> ndarray::Array1<Complex64> {
        let mut data = input.mapv(|x| Complex64::new(x, 0.0));
        self.fft
            .process(data.as_slice_mut().expect("Array must be contiguous"));
        data
    }

    pub fn inverse(&self, input: &ndarray::Array1<Complex64>) -> ndarray::Array1<f64> {
        let mut data = input.clone();
        self.ifft
            .process(data.as_slice_mut().expect("Array must be contiguous"));
        let norm = 1.0 / self.n as f64;
        data.mapv(|c| c.re * norm)
    }

    pub fn forward_into(
        &self,
        input: &ndarray::Array1<f64>,
        output: &mut ndarray::Array1<Complex64>,
    ) {
        Zip::from(&mut *output).and(input).for_each(|out, &val| {
            *out = Complex64::new(val, 0.0);
        });
        self.fft
            .process(output.as_slice_mut().expect("Array must be contiguous"));
    }

    pub fn inverse_into(
        &self,
        input: &ndarray::Array1<Complex64>,
        output: &mut ndarray::Array1<f64>,
        scratch: &mut ndarray::Array1<Complex64>,
    ) {
        scratch.assign(input);
        self.ifft
            .process(scratch.as_slice_mut().expect("Array must be contiguous"));
        let norm = 1.0 / self.n as f64;
        Zip::from(output).and(scratch).for_each(|out, val| {
            *out = val.re * norm;
        });
    }

    pub fn forward_complex_inplace(&self, data: &mut ndarray::Array1<Complex64>) {
        self.fft
            .process(data.as_slice_mut().expect("Array must be contiguous"));
    }

    pub fn inverse_complex_inplace(&self, data: &mut ndarray::Array1<Complex64>) {
        self.ifft
            .process(data.as_slice_mut().expect("Array must be contiguous"));
    }
}

/// 2D FFT processor for grid-based operations
pub struct Fft2d {
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
        }
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<Complex64> {
        let mut data = input.mapv(|x| Complex64::new(x, 0.0));
        self.forward_complex_inplace(&mut data);
        data
    }

    pub fn inverse(&self, input: &Array2<Complex64>) -> Array2<f64> {
        let mut data = input.clone();
        self.inverse_complex_inplace(&mut data);
        let norm = 1.0 / (self.nx * self.ny) as f64;
        data.mapv(|c| c.re * norm)
    }

    pub fn forward_into(&self, input: &Array2<f64>, output: &mut Array2<Complex64>) {
        Zip::from(&mut *output).and(input).for_each(|out, &val| {
            *out = Complex64::new(val, 0.0);
        });
        self.forward_complex_inplace(output);
    }

    pub fn inverse_into(
        &self,
        input: &Array2<Complex64>,
        output: &mut Array2<f64>,
        scratch: &mut Array2<Complex64>,
    ) {
        scratch.assign(input);
        self.inverse_complex_inplace(scratch);
        let norm = 1.0 / (self.nx * self.ny) as f64;
        Zip::from(output).and(scratch).for_each(|out, val| {
            *out = val.re * norm;
        });
    }

    pub fn forward_complex_inplace(&self, data: &mut Array2<Complex64>) {
        let fft_x = Arc::clone(&self.fft_x);
        let fft_y = Arc::clone(&self.fft_y);
        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                let mut buffer = row.to_vec();
                fft_x.process(&mut buffer);
                row.assign(&ArrayView1::from(&buffer));
            });
        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut col| {
                let mut buffer = col.to_vec();
                fft_y.process(&mut buffer);
                col.assign(&ArrayView1::from(&buffer));
            });
    }

    pub fn inverse_complex_inplace(&self, data: &mut Array2<Complex64>) {
        let ifft_x = Arc::clone(&self.ifft_x);
        let ifft_y = Arc::clone(&self.ifft_y);
        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut col| {
                let mut buffer = col.to_vec();
                ifft_y.process(&mut buffer);
                col.assign(&ArrayView1::from(&buffer));
            });
        data.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                let mut buffer = row.to_vec();
                ifft_x.process(&mut buffer);
                row.assign(&ArrayView1::from(&buffer));
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_fft_1d_forward_inverse() {
        let n = 64;
        let fft = Fft1d::new(n);
        let data = Array1::from_shape_fn(n, |i| (i as f64 * 0.1).sin());
        let spectrum = fft.forward(&data);
        let reconstructed = fft.inverse(&spectrum);
        for i in 0..n {
            assert_relative_eq!(data[i], reconstructed[i], epsilon = 1e-10);
        }
    }
}
