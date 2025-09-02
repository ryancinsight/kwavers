//! Cattaneo-Vernotte Hyperbolic Heat Transfer
//!
//! Reference: Cattaneo, C. (1958). "A form of heat conduction equation which eliminates
//! the paradox of instantaneous propagation." Comptes Rendus, 247, 431-433.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array3, Zip};

/// Cattaneo-Vernotte model parameters
#[derive(Debug, Clone)]
pub struct HyperbolicParameters {
    /// Thermal relaxation time [s]
    pub relaxation_time: f64,
    /// Thermal wave speed [m/s]
    pub thermal_wave_speed: f64,
}

impl Default for HyperbolicParameters {
    fn default() -> Self {
        Self {
            relaxation_time: 20.0,    // 20s typical for biological tissue
            thermal_wave_speed: 10.0, // Typical for soft tissue
        }
    }
}

/// Hyperbolic heat transfer solver
#[derive(Debug)]
pub struct CattaneoVernotte {
    params: HyperbolicParameters,
    /// Heat flux components
    heat_flux_x: Array3<f64>,
    heat_flux_y: Array3<f64>,
    heat_flux_z: Array3<f64>,
    /// Previous heat flux for time derivatives
    prev_flux_x: Array3<f64>,
    prev_flux_y: Array3<f64>,
    prev_flux_z: Array3<f64>,
}

impl CattaneoVernotte {
    pub fn new(params: HyperbolicParameters, grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            params,
            heat_flux_x: Array3::zeros(shape),
            heat_flux_y: Array3::zeros(shape),
            heat_flux_z: Array3::zeros(shape),
            prev_flux_x: Array3::zeros(shape),
            prev_flux_y: Array3::zeros(shape),
            prev_flux_z: Array3::zeros(shape),
        }
    }

    /// Update heat flux using Cattaneo-Vernotte equation
    /// τ ∂q/∂t + q = -k∇T
    pub fn update_heat_flux(
        &mut self,
        temperature: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        let tau = self.params.relaxation_time;
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        // Store previous flux
        self.prev_flux_x.assign(&self.heat_flux_x);
        self.prev_flux_y.assign(&self.heat_flux_y);
        self.prev_flux_z.assign(&self.heat_flux_z);

        // Update x-component of heat flux
        for i in 1..nx - 1 {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    let k_thermal = medium.thermal_conductivity(x, y, z, grid);
                    let grad_t =
                        (temperature[[i + 1, j, k]] - temperature[[i - 1, j, k]]) / (2.0 * grid.dx);

                    // Cattaneo-Vernotte equation for heat flux
                    let q_previous = self.heat_flux_x[[i, j, k]];
                    self.heat_flux_x[[i, j, k]] = (q_previous
                        - dt / tau * (q_previous + k_thermal * grad_t))
                        / (1.0 + dt / tau);
                }
            }
        }

        // Similar updates for y and z components
        for i in 0..nx {
            for j in 1..ny - 1 {
                for k in 0..nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    let k_thermal = medium.thermal_conductivity(x, y, z, grid);
                    let grad_t =
                        (temperature[[i, j + 1, k]] - temperature[[i, j - 1, k]]) / (2.0 * grid.dy);

                    let q_previous = self.heat_flux_y[[i, j, k]];
                    self.heat_flux_y[[i, j, k]] = (q_previous
                        - dt / tau * (q_previous + k_thermal * grad_t))
                        / (1.0 + dt / tau);
                }
            }
        }

        for i in 0..nx {
            for j in 0..ny {
                for k in 1..nz - 1 {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    let k_thermal = medium.thermal_conductivity(x, y, z, grid);
                    let grad_t =
                        (temperature[[i, j, k + 1]] - temperature[[i, j, k - 1]]) / (2.0 * grid.dz);

                    let q_previous = self.heat_flux_z[[i, j, k]];
                    self.heat_flux_z[[i, j, k]] = (q_previous
                        - dt / tau * (q_previous + k_thermal * grad_t))
                        / (1.0 + dt / tau);
                }
            }
        }

        Ok(())
    }

    /// Calculate divergence of heat flux
    pub fn heat_flux_divergence(&self, grid: &Grid) -> Array3<f64> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut div = Array3::zeros((nx, ny, nz));

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let div_x = (self.heat_flux_x[[i + 1, j, k]] - self.heat_flux_x[[i - 1, j, k]])
                        / (2.0 * grid.dx);
                    let div_y = (self.heat_flux_y[[i, j + 1, k]] - self.heat_flux_y[[i, j - 1, k]])
                        / (2.0 * grid.dy);
                    let div_z = (self.heat_flux_z[[i, j, k + 1]] - self.heat_flux_z[[i, j, k - 1]])
                        / (2.0 * grid.dz);

                    div[[i, j, k]] = div_x + div_y + div_z;
                }
            }
        }

        div
    }

    /// Update temperature using hyperbolic heat equation
    pub fn update_temperature(
        &mut self,
        temperature: &mut Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        // Update heat flux first
        self.update_heat_flux(temperature, medium, grid, dt)?;

        // Calculate divergence of heat flux
        let div_q = self.heat_flux_divergence(grid);

        // Update temperature: ∂T/∂t = -∇·q/(ρc)
        Zip::indexed(temperature)
            .and(&div_q)
            .for_each(|(i, j, k), t, &div| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = crate::medium::density_at(medium, x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);

                *t -= dt * div / (rho * cp);
            });

        Ok(())
    }
}
