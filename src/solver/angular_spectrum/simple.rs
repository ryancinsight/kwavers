//! Angular spectrum method for acoustic field calculations

use ndarray::{Array2, Array3};
use num_complex::Complex;
use rustfft::{FftPlanner, num_complex::Complex64};
use std::f64::consts::PI;

use crate::error::{KwaversError, KwaversResult};
use crate::grid::Grid;

/// Angular spectrum method for acoustic field calculations
#[derive(Debug)]
pub struct AngularSpectrum {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Grid spacing
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Wave number
    pub k: f64,
}

impl AngularSpectrum {
    /// Create new angular spectrum calculator
    pub fn new(grid: &Grid, frequency: f64, sound_speed: f64) -> Self {
        let k = 2.0 * PI * frequency / sound_speed;
        Self {
            nx: grid.nx,
            ny: grid.ny,
            nz: grid.nz,
            dx: grid.dx,
            dy: grid.dy,
            dz: grid.dz,
            k,
        }
    }

    /// Propagate field using angular spectrum method
    pub fn propagate(
        &self,
        field: &Array2<Complex64>,
        z_distance: f64,
    ) -> KwaversResult<Array2<Complex64>> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.nx);
        let ifft = planner.plan_fft_inverse(self.nx);
        
        // Convert to frequency domain
        let mut spectrum = field.clone();
        for mut row in spectrum.rows_mut() {
            let mut row_vec: Vec<Complex64> = row.to_vec();
            fft.process(&mut row_vec);
            let row_array = Array2::from_shape_vec((1, self.nx), row_vec)
                .map_err(|e| KwaversError::InvalidDimensions(format!("FFT row reshape failed: {}", e)))?;
            row.assign(&row_array.row(0));
        }
        
        // Apply propagation in frequency domain
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
                        let phase = Complex64::from_polar(1.0, kz * z_distance);
                        spectrum[[i, j]] *= phase;
                    } else {
                        // Evanescent waves decay exponentially
                        let kz = (-kz_sq).sqrt();
                        spectrum[[i, j]] *= (-kz * z_distance).exp();
                    }
                }
            }
        }
        
        // Convert back to spatial domain
        for mut row in spectrum.rows_mut() {
            let mut row_vec: Vec<Complex64> = row.to_vec();
            ifft.process(&mut row_vec);
            let row_array = Array2::from_shape_vec((1, self.nx), row_vec)
                .map_err(|e| KwaversError::InvalidDimensions(format!("IFFT row reshape failed: {}", e)))?;
            row.assign(&row_array.row(0));
        }
        
        // Normalize
        let scale = 1.0 / (self.nx as f64);
        spectrum.mapv_inplace(|x| x * scale);
        
        Ok(spectrum)
    }

    /// Calculate 3D angular spectrum propagation
    pub fn propagate_3d(
        &self,
        field: &Array3<Complex64>,
        z_distance: f64,
    ) -> KwaversResult<Array3<Complex64>> {
        let mut result = Array3::zeros(field.dim());
        
        // Process each z-slice independently
        for z in 0..self.nz {
            let slice = field.index_axis(ndarray::Axis(2), z);
            let propagated = self.propagate(&slice.to_owned(), z_distance)?;
            result.index_axis_mut(ndarray::Axis(2), z).assign(&propagated);
        }
        
        Ok(result)
    }

    /// Calculate transfer function for angular spectrum method
    pub fn transfer_function(&self, z_distance: f64) -> Array2<Complex64> {
        let mut h = Array2::zeros((self.nx, self.ny));
        
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
                
                let kz_sq = self.k * self.k - kx * kx - ky * ky;
                if kz_sq >= 0.0 {
                    let kz = kz_sq.sqrt();
                    h[[i, j]] = Complex64::from_polar(1.0, kz * z_distance);
                } else {
                    let kz = (-kz_sq).sqrt();
                    h[[i, j]] = Complex64::from((-kz * z_distance).exp());
                }
            }
        }
        
        h
    }
}