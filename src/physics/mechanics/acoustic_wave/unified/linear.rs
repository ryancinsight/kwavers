//! Linear acoustic wave equation solver

use crate::{
    error::KwaversResult,
    fft::{Fft3d, Ifft3d},
    grid::Grid,
    medium::Medium,
};
use ndarray::Array3;
use num_complex::Complex;

use super::config::AcousticSolverConfig;
use super::solver::AcousticSolver;

/// Linear acoustic wave equation solver
pub struct LinearSolver {
    config: AcousticSolverConfig,
    grid: Grid,
    k_space_filter: Option<Array3<f64>>,
}

impl LinearSolver {
    /// Create a new linear solver
    pub fn new(config: AcousticSolverConfig, grid: Grid) -> KwaversResult<Self> {
        let k_space_filter = if config.k_space_correction {
            Some(Self::create_k_space_filter(&grid, config.k_space_order)?)
        } else {
            None
        };

        Ok(Self {
            config,
            grid,
            k_space_filter,
        })
    }

    /// Create k-space correction filter
    fn create_k_space_filter(grid: &Grid, order: usize) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut filter = Array3::ones((nx, ny, nz));

        // Apply k-space correction based on order
        // Reference: Tabei et al., "A k-space method for coupled first-order acoustic propagation equations"
        // J. Acoust. Soc. Am. 111, 53-63 (2002)

        let kx_max = std::f64::consts::PI / grid.dx;
        let ky_max = std::f64::consts::PI / grid.dy;
        let kz_max = std::f64::consts::PI / grid.dz;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let kx = if i <= nx / 2 {
                        i as f64
                    } else {
                        (i as i32 - nx as i32) as f64
                    } * kx_max
                        / (nx as f64 / 2.0);
                    let ky = if j <= ny / 2 {
                        j as f64
                    } else {
                        (j as i32 - ny as i32) as f64
                    } * ky_max
                        / (ny as f64 / 2.0);
                    let kz = if k <= nz / 2 {
                        k as f64
                    } else {
                        (k as i32 - nz as i32) as f64
                    } * kz_max
                        / (nz as f64 / 2.0);

                    let k_norm = (kx * kx + ky * ky + kz * kz).sqrt();

                    if order == 1 {
                        // First-order correction
                        filter[[i, j, k]] = sinc(k_norm * grid.dx / 2.0);
                    } else {
                        // Second-order correction
                        let sinc_val = sinc(k_norm * grid.dx / 2.0);
                        filter[[i, j, k]] = sinc_val * sinc_val;
                    }
                }
            }
        }

        Ok(filter)
    }
}

impl AcousticSolver for LinearSolver {
    fn update(
        &mut self,
        pressure: &mut Array3<f64>,
        medium: &dyn Medium,
        source_term: &Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        // Add source term
        *pressure = &*pressure + source_term;

        // Apply k-space propagation if enabled
        if self.config.k_space_correction {
            // Transform to k-space
            let mut pressure_complex = pressure.mapv(|x| Complex::new(x, 0.0));
            let mut fft = Fft3d::new(grid.nx, grid.ny, grid.nz);
            fft.process(&mut pressure_complex, grid);

            // Apply k-space filter
            if let Some(ref filter) = self.k_space_filter {
                for (c, f) in pressure_complex.iter_mut().zip(filter.iter()) {
                    *c *= Complex::new(*f, 0.0);
                }
            }

            // Transform back
            let mut ifft = Ifft3d::new(grid.nx, grid.ny, grid.nz);
            ifft.process(&mut pressure_complex, grid);

            // Extract real part
            pressure.assign(&pressure_complex.mapv(|c| c.re));
        } else {
            // Standard finite difference update
            apply_finite_difference_update(pressure, medium, grid, dt)?;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "Linear Acoustic Solver"
    }

    fn check_stability(&self, dt: f64, grid: &Grid, max_sound_speed: f64) -> KwaversResult<()> {
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = max_sound_speed * dt / dx_min;

        if cfl > self.config.cfl_safety_factor {
            return Err(crate::error::ValidationError::RangeValidation {
                field: "CFL".to_string(),
                value: cfl.to_string(),
                min: "0".to_string(),
                max: self.config.cfl_safety_factor.to_string(),
            }
            .into());
        }

        Ok(())
    }
}

/// Sinc function for k-space correction
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        x.sin() / x
    }
}

/// Apply standard finite difference update
fn apply_finite_difference_update(
    pressure: &mut Array3<f64>,
    medium: &dyn Medium,
    grid: &Grid,
    dt: f64,
) -> KwaversResult<()> {
    let (nx, ny, nz) = pressure.dim();
    let mut laplacian = Array3::zeros((nx, ny, nz));

    // Compute Laplacian with 2nd order central differences
    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                laplacian[[i, j, k]] = (pressure[[i + 1, j, k]] - 2.0 * pressure[[i, j, k]]
                    + pressure[[i - 1, j, k]])
                    / (grid.dx * grid.dx)
                    + (pressure[[i, j + 1, k]] - 2.0 * pressure[[i, j, k]]
                        + pressure[[i, j - 1, k]])
                        / (grid.dy * grid.dy)
                    + (pressure[[i, j, k + 1]] - 2.0 * pressure[[i, j, k]]
                        + pressure[[i, j, k - 1]])
                        / (grid.dz * grid.dz);
            }
        }
    }

    // Update pressure using wave equation: ∂²p/∂t² = c² ∇²p
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = medium.sound_speed(
                    i as f64 * grid.dx,
                    j as f64 * grid.dy,
                    k as f64 * grid.dz,
                    grid,
                );
                pressure[[i, j, k]] += dt * dt * c * c * laplacian[[i, j, k]];
            }
        }
    }

    Ok(())
}
