//! Diffraction operator using angular spectrum method
//!
//! Reference: Goodman (2005) "Introduction to Fourier Optics"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::mechanics::acoustic_wave::hybrid_angular_spectrum::HASConfig;
use ndarray::Array3;
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

/// Diffraction operator using FFT-based angular spectrum
pub struct DiffractionOperator {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    k: f64,
    fft_forward: Arc<dyn Fft<f64>>,
    fft_inverse: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for DiffractionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiffractionOperator")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("dx", &self.dx)
            .field("dy", &self.dy)
            .field("k", &self.k)
            .finish()
    }
}

impl DiffractionOperator {
    /// Create new diffraction operator
    pub fn new(grid: &Grid, config: &HASConfig) -> KwaversResult<Self> {
        let k = 2.0 * PI * config.reference_frequency / config.sound_speed;

        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(grid.nx);
        let fft_inverse = planner.plan_fft_inverse(grid.nx);

        Ok(Self {
            nx: grid.nx,
            ny: grid.ny,
            dx: grid.dx,
            dy: grid.dy,
            k,
            fft_forward,
            fft_inverse,
        })
    }

    /// Apply diffraction step
    ///
    /// Propagates field by distance dz using angular spectrum
    pub fn apply(&self, pressure: &Array3<f64>, dz: f64) -> KwaversResult<Array3<f64>> {
        let nz = pressure.shape()[2];
        let mut result = pressure.clone();

        // Process each z-plane
        for k_plane in 0..nz {
            // Extract 2D slice
            let mut plane = Array3::zeros((self.nx, self.ny, 1));
            for i in 0..self.nx {
                for j in 0..self.ny {
                    plane[[i, j, 0]] = pressure[[i, j, k_plane]];
                }
            }

            // Apply angular spectrum
            let propagated = self.propagate_2d_plane(&plane, dz)?;

            // Copy back
            for i in 0..self.nx {
                for j in 0..self.ny {
                    result[[i, j, k_plane]] = propagated[[i, j, 0]];
                }
            }
        }

        Ok(result)
    }

    fn propagate_2d_plane(&self, plane: &Array3<f64>, dz: f64) -> KwaversResult<Array3<f64>> {
        // Convert to complex
        let mut field: Vec<Complex64> = plane
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();

        // 2D FFT
        self.fft_2d_forward(&mut field)?;

        // Apply propagation phase
        let kx_max = PI / self.dx;
        let ky_max = PI / self.dy;

        for i in 0..self.nx {
            for j in 0..self.ny {
                let idx = i * self.ny + j;

                let kx = if i < self.nx / 2 {
                    2.0 * PI * i as f64 / (self.nx as f64 * self.dx)
                } else {
                    2.0 * PI * (i as f64 - self.nx as f64) / (self.nx as f64 * self.dx)
                };

                let ky = if j < self.ny / 2 {
                    2.0 * PI * j as f64 / (self.ny as f64 * self.dy)
                } else {
                    2.0 * PI * (j as f64 - self.ny as f64) / (self.ny as f64 * self.dy)
                };

                if kx.abs() <= kx_max && ky.abs() <= ky_max {
                    let kz_sq = self.k * self.k - kx * kx - ky * ky;
                    if kz_sq >= 0.0 {
                        let kz = kz_sq.sqrt();
                        let phase = Complex64::from_polar(1.0, kz * dz);
                        field[idx] *= phase;
                    } else {
                        // Evanescent waves decay
                        let kz = (-kz_sq).sqrt();
                        field[idx] *= (-kz * dz).exp();
                    }
                }
            }
        }

        // 2D IFFT
        self.fft_2d_inverse(&mut field)?;

        // Convert back to real
        let mut result = Array3::zeros((self.nx, self.ny, 1));
        for i in 0..self.nx {
            for j in 0..self.ny {
                let idx = i * self.ny + j;
                result[[i, j, 0]] = field[idx].re;
            }
        }

        Ok(result)
    }

    fn fft_2d_forward(&self, data: &mut [Complex64]) -> KwaversResult<()> {
        // Row-wise FFT
        for i in 0..self.nx {
            let start = i * self.ny;
            let end = start + self.ny;
            let mut row = data[start..end].to_vec();
            self.fft_forward.process(&mut row);
            data[start..end].copy_from_slice(&row);
        }

        // Column-wise FFT
        for j in 0..self.ny {
            let mut col = Vec::with_capacity(self.nx);
            for i in 0..self.nx {
                col.push(data[i * self.ny + j]);
            }
            self.fft_forward.process(&mut col);
            for i in 0..self.nx {
                data[i * self.ny + j] = col[i];
            }
        }

        Ok(())
    }

    fn fft_2d_inverse(&self, data: &mut [Complex64]) -> KwaversResult<()> {
        // Row-wise IFFT
        for i in 0..self.nx {
            let start = i * self.ny;
            let end = start + self.ny;
            let mut row = data[start..end].to_vec();
            self.fft_inverse.process(&mut row);
            // Normalize
            for val in &mut row {
                *val /= self.ny as f64;
            }
            data[start..end].copy_from_slice(&row);
        }

        // Column-wise IFFT
        for j in 0..self.ny {
            let mut col = Vec::with_capacity(self.nx);
            for i in 0..self.nx {
                col.push(data[i * self.ny + j]);
            }
            self.fft_inverse.process(&mut col);
            // Normalize
            for val in &mut col {
                *val /= self.nx as f64;
            }
            for i in 0..self.nx {
                data[i * self.ny + j] = col[i];
            }
        }

        Ok(())
    }
}
