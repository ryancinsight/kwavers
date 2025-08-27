//! Unified thermal diffusion solver

use crate::{error::KwaversResult, grid::Grid, medium::Medium};
use ndarray::{Array3, Zip};

use super::{
    bioheat::{BioheatParameters, PennesBioheat},
    dose::ThermalDoseCalculator,
    hyperbolic::{CattaneoVernotte, HyperbolicParameters},
    ThermalDiffusionConfig,
};

/// Unified thermal diffusion solver
#[derive(Debug)]
pub struct ThermalDiffusionSolver {
    config: ThermalDiffusionConfig,
    temperature: Array3<f64>,
    temperature_prev: Option<Array3<f64>>,
    bioheat_solver: Option<PennesBioheat>,
    hyperbolic_solver: Option<CattaneoVernotte>,
    dose_calculator: Option<ThermalDoseCalculator>,
    /// Workspace for Laplacian calculation
    laplacian_workspace: Array3<f64>,
    /// Current simulation time
    current_time: f64,
}

impl ThermalDiffusionSolver {
    pub fn new(config: ThermalDiffusionConfig, grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        let temperature = Array3::from_elem(shape, config.arterial_temperature);

        let bioheat_solver = if config.enable_bioheat {
            Some(PennesBioheat::new(BioheatParameters {
                perfusion_rate: config.perfusion_rate,
                blood_density: config.blood_density,
                blood_specific_heat: config.blood_specific_heat,
                arterial_temperature: config.arterial_temperature,
            }))
        } else {
            None
        };

        let hyperbolic_solver = if config.enable_hyperbolic {
            Some(CattaneoVernotte::new(
                HyperbolicParameters {
                    relaxation_time: config.relaxation_time,
                    thermal_wave_speed: 10.0, // Default thermal wave speed
                },
                grid,
            ))
        } else {
            None
        };

        let dose_calculator = if config.track_thermal_dose {
            Some(ThermalDoseCalculator::new(shape))
        } else {
            None
        };

        Self {
            config,
            temperature,
            temperature_prev: None,
            bioheat_solver,
            hyperbolic_solver,
            dose_calculator,
            laplacian_workspace: Array3::zeros(shape),
            current_time: 0.0,
        }
    }

    /// Calculate the Laplacian of temperature field
    fn calculate_laplacian(&mut self, grid: &Grid) -> KwaversResult<()> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        match self.config.spatial_order {
            2 => {
                // Second-order accurate Laplacian
                let dx2_inv = 1.0 / (grid.dx * grid.dx);
                let dy2_inv = 1.0 / (grid.dy * grid.dy);
                let dz2_inv = 1.0 / (grid.dz * grid.dz);

                for i in 1..nx - 1 {
                    for j in 1..ny - 1 {
                        for k in 1..nz - 1 {
                            let d2_dx2 = (self.temperature[[i + 1, j, k]]
                                - 2.0 * self.temperature[[i, j, k]]
                                + self.temperature[[i - 1, j, k]])
                                * dx2_inv;
                            let d2_dy2 = (self.temperature[[i, j + 1, k]]
                                - 2.0 * self.temperature[[i, j, k]]
                                + self.temperature[[i, j - 1, k]])
                                * dy2_inv;
                            let d2_dz2 = (self.temperature[[i, j, k + 1]]
                                - 2.0 * self.temperature[[i, j, k]]
                                + self.temperature[[i, j, k - 1]])
                                * dz2_inv;

                            self.laplacian_workspace[[i, j, k]] = d2_dx2 + d2_dy2 + d2_dz2;
                        }
                    }
                }
            }
            4 => {
                // Fourth-order accurate Laplacian
                const C0: f64 = -1.0 / 12.0;
                const C1: f64 = 4.0 / 3.0;
                const C2: f64 = -5.0 / 2.0;

                let dx2_inv = 1.0 / (grid.dx * grid.dx);
                let dy2_inv = 1.0 / (grid.dy * grid.dy);
                let dz2_inv = 1.0 / (grid.dz * grid.dz);

                for i in 2..nx - 2 {
                    for j in 2..ny - 2 {
                        for k in 2..nz - 2 {
                            let d2_dx2 = (C0 * self.temperature[[i - 2, j, k]]
                                + C1 * self.temperature[[i - 1, j, k]]
                                + C2 * self.temperature[[i, j, k]]
                                + C1 * self.temperature[[i + 1, j, k]]
                                + C0 * self.temperature[[i + 2, j, k]])
                                * dx2_inv;

                            let d2_dy2 = (C0 * self.temperature[[i, j - 2, k]]
                                + C1 * self.temperature[[i, j - 1, k]]
                                + C2 * self.temperature[[i, j, k]]
                                + C1 * self.temperature[[i, j + 1, k]]
                                + C0 * self.temperature[[i, j + 2, k]])
                                * dy2_inv;

                            let d2_dz2 = (C0 * self.temperature[[i, j, k - 2]]
                                + C1 * self.temperature[[i, j, k - 1]]
                                + C2 * self.temperature[[i, j, k]]
                                + C1 * self.temperature[[i, j, k + 1]]
                                + C0 * self.temperature[[i, j, k + 2]])
                                * dz2_inv;

                            self.laplacian_workspace[[i, j, k]] = d2_dx2 + d2_dy2 + d2_dz2;
                        }
                    }
                }
            }
            _ => {
                // Default to second-order
                self.config.spatial_order = 2;
                self.calculate_laplacian(grid)?;
            }
        }

        Ok(())
    }

    /// Update temperature field
    pub fn update(
        &mut self,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
        external_source: Option<&Array3<f64>>,
    ) -> KwaversResult<()> {
        // Store previous temperature
        self.temperature_prev = Some(self.temperature.clone());

        if self.config.enable_hyperbolic {
            // Use hyperbolic heat transfer (Cattaneo-Vernotte)
            if let Some(ref mut solver) = self.hyperbolic_solver {
                solver.update_temperature(&mut self.temperature, medium, grid, dt)?;
            }
        } else {
            // Calculate Laplacian for diffusion
            self.calculate_laplacian(grid)?;

            if self.config.enable_bioheat {
                // Use Pennes bioheat equation
                if let Some(ref solver) = self.bioheat_solver {
                    solver.update(
                        &mut self.temperature,
                        &self.laplacian_workspace,
                        external_source,
                        medium,
                        grid,
                        dt,
                    )?;
                }
            } else {
                // Standard heat diffusion
                self.update_standard_diffusion(external_source, medium, grid, dt)?;
            }
        }

        // Update thermal dose if tracking
        if let Some(ref mut calc) = self.dose_calculator {
            self.current_time += dt;
            calc.update_dose(&self.temperature, dt, self.current_time)?;
        }

        Ok(())
    }

    /// Standard heat diffusion update
    fn update_standard_diffusion(
        &mut self,
        external_source: Option<&Array3<f64>>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        Zip::indexed(&mut self.temperature)
            .and(&self.laplacian_workspace)
            .for_each(|(i, j, k), temp, &lap| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let alpha = medium.thermal_diffusivity(x, y, z, grid);
                let source = external_source.map(|s| s[[i, j, k]]).unwrap_or(0.0);

                *temp += dt * (alpha * lap + source);
            });

        Ok(())
    }

    /// Get current temperature field
    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }

    /// Set temperature field
    pub fn set_temperature(&mut self, temperature: Array3<f64>) {
        self.temperature = temperature;
    }

    /// Get thermal dose if tracking
    pub fn thermal_dose(&self) -> Option<&Array3<f64>> {
        self.dose_calculator.as_ref().map(|c| c.get_dose())
    }

    /// Get maximum thermal dose
    pub fn max_thermal_dose(&self) -> Option<f64> {
        self.dose_calculator.as_ref().map(|c| c.max_dose())
    }

    /// Get necrosis fraction
    pub fn necrosis_fraction(&self) -> Option<f64> {
        self.dose_calculator.as_ref().map(|c| c.necrosis_fraction())
    }

    /// Reset the solver
    pub fn reset(&mut self, grid: &Grid) {
        let shape = (grid.nx, grid.ny, grid.nz);
        self.temperature = Array3::from_elem(shape, self.config.arterial_temperature);
        self.temperature_prev = None;
        self.laplacian_workspace.fill(0.0);
        self.current_time = 0.0;

        if let Some(ref mut calc) = self.dose_calculator {
            calc.reset();
        }
    }
}
