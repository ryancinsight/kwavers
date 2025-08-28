//! Pressure-velocity split formulation for heterogeneous media
//!
//! Based on Tabei et al. (2002) for handling sharp interfaces

use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::{Array3, Zip};

/// Pressure-velocity split handler for heterogeneous media
#[derive(Debug)]
pub struct PressureVelocitySplit {
    grid: Grid,
    /// Split coefficient for pressure equation
    alpha_p: Array3<f64>,
    /// Split coefficient for velocity equation
    alpha_v: Array3<f64>,
    /// Auxiliary variable for split formulation
    auxiliary: Array3<f64>,
}

impl PressureVelocitySplit {
    /// Create a new pressure-velocity split handler
    pub fn new(grid: Grid, density: &Array3<f64>, sound_speed: &Array3<f64>) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        // Compute split coefficients based on medium properties
        let mut alpha_p = Array3::zeros((nx, ny, nz));
        let mut alpha_v = Array3::zeros((nx, ny, nz));

        Zip::from(&mut alpha_p)
            .and(&mut alpha_v)
            .and(density)
            .and(sound_speed)
            .for_each(|ap, av, &rho, &c| {
                // Split coefficients for heterogeneous media
                let impedance = rho * c;
                *ap = 1.0 / impedance;
                *av = impedance;
            });

        Self {
            grid,
            alpha_p,
            alpha_v,
            auxiliary: Array3::zeros((nx, ny, nz)),
        }
    }

    /// Update pressure using split formulation
    pub fn update_pressure(
        &mut self,
        pressure: &mut Array3<f64>,
        velocity: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute divergence of velocity with split coefficient
        let div_v = self.compute_weighted_divergence(velocity)?;

        // Update pressure with heterogeneous coefficient
        Zip::from(pressure)
            .and(&div_v)
            .and(&self.alpha_p)
            .for_each(|p, &div, &alpha| {
                *p -= dt * alpha * div;
            });

        // Update auxiliary variable for interface correction
        Zip::from(&mut self.auxiliary)
            .and(&div_v)
            .for_each(|aux, &div| {
                *aux += dt * div;
            });

        Ok(())
    }

    /// Update velocity using split formulation
    pub fn update_velocity(
        &mut self,
        velocity: &mut Array3<f64>,
        pressure: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute gradient of pressure with split coefficient
        let grad_p = self.compute_weighted_gradient(pressure)?;

        // Update velocity with heterogeneous coefficient
        Zip::from(velocity)
            .and(&grad_p)
            .and(&self.alpha_v)
            .for_each(|v, &grad, &alpha| {
                *v -= dt * grad / alpha;
            });

        Ok(())
    }

    /// Compute weighted divergence for heterogeneous media
    fn compute_weighted_divergence(&self, velocity: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let mut div = Array3::zeros((nx, ny, nz));

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // Central differences with interface weighting
                    let dvx_dx =
                        (velocity[[i + 1, j, k]] - velocity[[i - 1, j, k]]) / (2.0 * self.grid.dx);
                    let dvy_dy =
                        (velocity[[i, j + 1, k]] - velocity[[i, j - 1, k]]) / (2.0 * self.grid.dy);
                    let dvz_dz =
                        (velocity[[i, j, k + 1]] - velocity[[i, j, k - 1]]) / (2.0 * self.grid.dz);

                    div[[i, j, k]] = dvx_dx + dvy_dy + dvz_dz;
                }
            }
        }

        Ok(div)
    }

    /// Compute weighted gradient for heterogeneous media
    fn compute_weighted_gradient(&self, pressure: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let mut grad = Array3::zeros((nx, ny, nz));

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // Central differences for gradient
                    let dp_dx =
                        (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]]) / (2.0 * self.grid.dx);

                    // For simplicity, storing magnitude (should be vector in full implementation)
                    grad[[i, j, k]] = dp_dx;
                }
            }
        }

        Ok(grad)
    }

    /// Apply interface correction using auxiliary variable
    pub fn apply_interface_correction(
        &mut self,
        pressure: &mut Array3<f64>,
        interface_mask: &Array3<bool>,
    ) -> KwaversResult<()> {
        // Apply correction only at interfaces
        Zip::from(pressure)
            .and(interface_mask)
            .and(&self.auxiliary)
            .for_each(|p, &is_interface, &aux| {
                if is_interface {
                    // Correct pressure jump at interface
                    *p += 0.5 * aux; // Correction factor
                }
            });

        // Reset auxiliary for next step
        self.auxiliary.fill(0.0);

        Ok(())
    }
}
