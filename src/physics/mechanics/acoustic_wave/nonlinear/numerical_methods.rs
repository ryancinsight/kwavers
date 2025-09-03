//! Numerical methods for nonlinear acoustic wave simulation
//!
//! This module contains the core numerical algorithms for solving nonlinear acoustic equations.

use crate::constants::numerical;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::utils::fft_operations::{fft_3d_array, ifft_3d_array};
use crate::KwaversResult;

use log::{debug, warn};
use ndarray::{Array3, Zip};
use rustfft::num_complex::Complex;
use std::f64;
use std::time::Instant;

use super::wave_model::NonlinearWave;

impl NonlinearWave {
    /// Updates the wave field using the nonlinear acoustic wave equation.
    ///
    /// This method implements a pseudo-spectral time-domain (PSTD) solver for the
    /// nonlinear acoustic wave equation. It uses FFT for spatial derivatives and
    /// includes nonlinear terms for accurate modeling of high-intensity acoustics.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Current pressure field [Pa]
    /// * `source` - Source term array [Pa/s²]
    /// * `medium` - Medium properties
    /// * `grid` - Computational grid
    /// * `time_step` - Current time step index
    ///
    /// # Returns
    ///
    /// Updated pressure field (internal implementation)
    pub fn update_wave_inner(
        &mut self,
        pressure: &Array3<f64>,
        source: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        time_step: usize,
    ) -> KwaversResult<Array3<f64>> {
        let start_total = Instant::now();
        self.call_count += 1;

        // Validate inputs
        if pressure.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(crate::error::KwaversError::InvalidInput(format!(
                "Pressure array shape [{}, {}, {}] doesn't match grid dimensions [{}, {}, {}]",
                pressure.shape()[0],
                pressure.shape()[1],
                pressure.shape()[2],
                grid.nx,
                grid.ny,
                grid.nz
            )));
        }

        // Compute nonlinear term
        let start_nonlinear = Instant::now();
        let nonlinear_term = self.compute_nonlinear_term(pressure, medium, grid)?;
        self.nonlinear_time += start_nonlinear.elapsed().as_secs_f64();

        // Apply k-space correction
        let start_fft = Instant::now();
        let linear_term = self.apply_k_space_correction(pressure, medium, grid)?;
        self.fft_time += start_fft.elapsed().as_secs_f64();

        // Add source term
        let start_source = Instant::now();
        let source_contribution = source * self.dt.powi(2);
        self.source_time += start_source.elapsed().as_secs_f64();

        // Combine terms
        let start_combination = Instant::now();
        let mut updated_pressure =
            linear_term + nonlinear_term * self.nonlinearity_scaling + source_contribution;

        // Apply stability constraints if needed
        if self.clamp_gradients {
            self.apply_stability_constraints(&mut updated_pressure);
        }

        self.combination_time += start_combination.elapsed().as_secs_f64();

        debug!(
            "Step {}: max pressure = {:.2e} Pa, update time = {:.3} ms",
            time_step,
            updated_pressure
                .iter()
                .fold(0.0_f64, |max, &val| max.max(val.abs())),
            start_total.elapsed().as_secs_f64() * 1000.0
        );

        Ok(updated_pressure)
    }

    /// Computes the nonlinear term for the acoustic wave equation.
    ///
    /// The nonlinear term accounts for finite-amplitude effects in acoustic propagation.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Current pressure field [Pa]
    /// * `medium` - Medium properties
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// The nonlinear term contribution
    fn compute_nonlinear_term(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Get average medium properties (simplified for homogeneous media)
        // For heterogeneous media, this would need position-dependent calculations
        let x = grid.nx as f64 * grid.dx / 2.0;
        let y = grid.ny as f64 * grid.dy / 2.0;
        let z = grid.nz as f64 * grid.dz / 2.0;
        let density = crate::medium::density_at(medium, x, y, z, grid);
        let sound_speed = crate::medium::sound_speed_at(medium, x, y, z, grid);
        let nonlinearity = 3.5; // Default B/A for water (would need to be added to Medium trait)

        // Compute pressure gradients using spectral differentiation
        let (grad_x, grad_y, grad_z) = self.compute_spectral_gradient(pressure, grid)?;

        // Compute nonlinear term for Westervelt equation
        // The full nonlinear term includes both (∇p)² and p∇²p terms
        // N = (β/ρ₀c₀⁴) * [p * ∇²p + (∇p)²]
        // where β = 1 + B/2A is the nonlinearity parameter
        let beta = 1.0 + nonlinearity / 2.0;
        let prefactor = beta / (density * sound_speed.powi(4));

        // Compute Laplacian
        let laplacian = self.compute_spectral_laplacian(pressure, grid)?;

        // Compute gradient squared term: (∇p)² = (∂p/∂x)² + (∂p/∂y)² + (∂p/∂z)²
        let grad_squared = &grad_x * &grad_x + &grad_y * &grad_y + &grad_z * &grad_z;

        // Full nonlinear term
        let nonlinear = (pressure * laplacian + grad_squared) * prefactor;

        Ok(nonlinear)
    }

    /// Applies k-space correction for the linear wave propagation.
    ///
    /// This implements the k-space pseudo-spectral method for accurate
    /// wave propagation without numerical dispersion.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Current pressure field
    /// * `medium` - Medium properties
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// The k-space corrected pressure field
    fn apply_k_space_correction(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Transform to k-space
        let pressure_k = fft_3d_array(pressure);

        // Get k-space grid
        let kx = grid.compute_kx();
        let ky = grid.compute_ky();
        let kz = grid.compute_kz();

        // Get spatially-varying sound speed
        let c_array = medium.sound_speed_array();
        let c = c_array.mean().unwrap_or(self.max_sound_speed);
        let mut result_k = Array3::<Complex<f64>>::zeros(pressure_k.raw_dim());

        // Use pre-computed k_squared if available
        if let Some(ref k_squared) = self.k_squared {
            Zip::from(&mut result_k)
                .and(&pressure_k)
                .and(k_squared)
                .for_each(|r, &p, &k2| {
                    let k = k2.sqrt();
                    let sinc_factor = if k > numerical::EPSILON {
                        (c * k * self.dt / 2.0).sin() / (c * k * self.dt / 2.0)
                    } else {
                        1.0
                    };
                    *r = p * Complex::new(sinc_factor * (-c.powi(2) * k2 * self.dt.powi(2)), 0.0)
                        .exp();
                });
        } else {
            // Compute k-squared on the fly
            result_k.indexed_iter_mut().for_each(|((i, j, k), val)| {
                let k_mag_sq = kx[i].powi(2) + ky[j].powi(2) + kz[k].powi(2);
                let k_mag = k_mag_sq.sqrt();

                let sinc_factor = if k_mag > numerical::EPSILON {
                    (c * k_mag * self.dt / 2.0).sin() / (c * k_mag * self.dt / 2.0)
                } else {
                    1.0
                };

                *val = pressure_k[(i, j, k)]
                    * Complex::new(sinc_factor * (-c.powi(2) * k_mag_sq * self.dt.powi(2)), 0.0)
                        .exp();
            });
        }

        // Transform back to spatial domain
        Ok(ifft_3d_array(&result_k))
    }

    /// Computes the spectral gradient of a field.
    ///
    /// # Arguments
    ///
    /// * `field` - Input field
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// Tuple of (`grad_x`, `grad_y`, `grad_z`)
    fn compute_spectral_gradient(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        // Transform to k-space
        let field_k = fft_3d_array(field);

        // Get k-space grid
        let kx = grid.compute_kx();
        let ky = grid.compute_ky();
        let kz = grid.compute_kz();

        // Compute gradients in k-space
        let mut grad_x_k = Array3::<Complex<f64>>::zeros(field_k.raw_dim());
        let mut grad_y_k = Array3::<Complex<f64>>::zeros(field_k.raw_dim());
        let mut grad_z_k = Array3::<Complex<f64>>::zeros(field_k.raw_dim());

        grad_x_k.indexed_iter_mut().for_each(|((i, j, k), val)| {
            *val = field_k[(i, j, k)] * Complex::new(0.0, kx[i]);
        });

        grad_y_k.indexed_iter_mut().for_each(|((i, j, k), val)| {
            *val = field_k[(i, j, k)] * Complex::new(0.0, ky[j]);
        });

        grad_z_k.indexed_iter_mut().for_each(|((i, j, k), val)| {
            *val = field_k[(i, j, k)] * Complex::new(0.0, kz[k]);
        });

        // Transform back to spatial domain
        Ok((
            ifft_3d_array(&grad_x_k),
            ifft_3d_array(&grad_y_k),
            ifft_3d_array(&grad_z_k),
        ))
    }

    /// Computes the spectral Laplacian of a field.
    ///
    /// # Arguments
    ///
    /// * `field` - Input field
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// The Laplacian of the field
    fn compute_spectral_laplacian(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Transform to k-space
        let field_k = fft_3d_array(field);

        // Apply Laplacian operator in k-space
        let mut laplacian_k = Array3::<Complex<f64>>::zeros(field_k.raw_dim());

        if let Some(ref k_squared) = self.k_squared {
            // Use pre-computed k-squared
            Zip::from(&mut laplacian_k)
                .and(&field_k)
                .and(k_squared)
                .for_each(|l, &f, &k2| {
                    *l = f * Complex::new(-k2, 0.0);
                });
        } else {
            // Compute k-squared on the fly
            let kx = grid.compute_kx();
            let ky = grid.compute_ky();
            let kz = grid.compute_kz();

            laplacian_k.indexed_iter_mut().for_each(|((i, j, k), val)| {
                let k_mag_sq = kx[i].powi(2) + ky[j].powi(2) + kz[k].powi(2);
                *val = field_k[(i, j, k)] * Complex::new(-k_mag_sq, 0.0);
            });
        }

        // Transform back to spatial domain
        Ok(ifft_3d_array(&laplacian_k))
    }

    /// Applies stability constraints to prevent numerical instabilities.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to constrain (modified in place)
    fn apply_stability_constraints(&self, field: &mut Array3<f64>) {
        // Clamp extreme values
        field.iter_mut().for_each(|val| {
            if val.abs() > self.max_pressure {
                *val = val.signum() * self.max_pressure;
            }
            // Remove NaN or Inf values
            if !val.is_finite() {
                *val = 0.0;
                warn!("Non-finite value detected and zeroed in pressure field");
            }
        });
    }

    /// Computes adaptive time step based on CFL condition.
    ///
    /// # Arguments
    ///
    /// * `medium` - Medium properties
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// Recommended time step [s]
    pub fn compute_adaptive_timestep(&self, medium: &dyn Medium, grid: &Grid) -> f64 {
        // Get actual maximum sound speed from medium
        let mut max_c: f64 = 0.0;
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    let c = crate::medium::sound_speed_at(medium, x, y, z, grid);
                    max_c = max_c.max(c);
                }
            }
        }

        // Fall back to stored value if medium returns zero
        if max_c <= 0.0 {
            max_c = self.max_sound_speed;
        }

        let min_dx = grid.dx.min(grid.dy).min(grid.dz);

        // CFL condition for PSTD
        let dt_cfl = self.cfl_safety_factor * min_dx / (f64::consts::PI * max_c);

        // Additional constraint for nonlinear terms
        let dt_nonlinear = if self.nonlinearity_scaling > 0.0 {
            // Get typical B/A from center of grid
            let cx = grid.nx / 2;
            let cy = grid.ny / 2;
            let cz = grid.nz / 2;
            let (x, y, z) = grid.indices_to_coordinates(cx, cy, cz);
            let beta = medium.nonlinearity_coefficient(x, y, z, grid);
            min_dx / (beta * max_c)
        } else {
            f64::INFINITY
        };

        dt_cfl.min(dt_nonlinear)
    }
}
