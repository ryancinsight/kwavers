//! Diffraction operator using angular spectrum method
//!
//! Reference: Goodman (2005) "Introduction to Fourier Optics"

use crate::error::KwaversResult;
use crate::fft::{fft_2d_complex, ifft_2d_complex, Complex64};
use crate::grid::Grid;
use crate::physics::mechanics::acoustic_wave::hybrid_angular_spectrum::HASConfig;
use ndarray::{Array2, Array3};
use std::f64::consts::PI;

/// Diffraction operator using FFT-based angular spectrum
pub struct DiffractionOperator {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    k: f64,
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

        Ok(Self {
            nx: grid.nx,
            ny: grid.ny,
            dx: grid.dx,
            dy: grid.dy,
            k,
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
        // Convert to complex Array2
        let plane_2d = plane.index_axis(ndarray::Axis(2), 0).to_owned();
        let mut field = plane_2d.mapv(|v| Complex64::new(v, 0.0));

        // 2D FFT
        let mut spectrum = fft_2d_complex(&field);

        // Apply propagation phase
        let kx_max = PI / self.dx;
        let ky_max = PI / self.dy;

        for i in 0..self.nx {
            for j in 0..self.ny {
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
                        spectrum[[i, j]] *= phase;
                    } else {
                        // Evanescent waves decay
                        let kz = (-kz_sq).sqrt();
                        spectrum[[i, j]] *= (-kz * dz).exp();
                    }
                }
            }
        }

        // 2D IFFT
        let propagated = ifft_2d_complex(&spectrum);

        // Convert back to real Array3
        let mut result = Array3::zeros((self.nx, self.ny, 1));
        for i in 0..self.nx {
            for j in 0..self.ny {
                result[[i, j, 0]] = propagated[[i, j]].re;
            }
        }

        Ok(result)
    }
}
