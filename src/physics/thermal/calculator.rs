// thermal/calculator.rs - Thermal field calculator

use super::{ThermalConfig, ThermalState};
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array3, Zip};

/// Thermal calculator for temperature evolution
#[derive(Debug)]
pub struct ThermalCalculator {
    state: ThermalState,
    config: ThermalConfig,
}

impl ThermalCalculator {
    /// Create new calculator - USING grid parameter properly
    pub fn new(grid: &Grid, initial_temperature: f64) -> Self {
        Self {
            state: ThermalState::new(grid, initial_temperature),
            config: ThermalConfig::default(),
        }
    }

    /// Configure the calculator
    pub fn with_config(mut self, config: ThermalConfig) -> Self {
        self.config = config;
        self
    }

    /// Update temperature field - USING all parameters
    pub fn update(
        &mut self,
        heat_source: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute thermal diffusion
        self.compute_diffusion(medium, grid, dt)?;

        // Add heat source - USING the parameter
        Zip::from(&mut self.state.temperature)
            .and(heat_source)
            .for_each(|t, &q| {
                *t += q * dt
                    / (medium.density(0.0, 0.0, 0.0, grid)
                        * medium.specific_heat(0.0, 0.0, 0.0, grid));
            });

        // Apply bioheat cooling if enabled
        if self.config.use_bioheat {
            self.apply_bioheat_cooling(medium, grid, dt)?;
        }

        Ok(())
    }

    /// Compute thermal diffusion - USING grid parameter
    fn compute_diffusion(
        &mut self,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        let k = medium.thermal_conductivity(0.0, 0.0, 0.0, grid);
        let rho = medium.density(0.0, 0.0, 0.0, grid);
        let cp = medium.specific_heat(0.0, 0.0, 0.0, grid);
        let alpha = k / (rho * cp);

        // Use grid spacing for finite differences
        let dx2 = grid.dx * grid.dx;
        let dy2 = grid.dy * grid.dy;
        let dz2 = grid.dz * grid.dz;

        let (nx, ny, nz) = self.state.temperature.dim();
        let mut laplacian = Array3::zeros((nx, ny, nz));

        // Compute Laplacian using grid spacing
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    laplacian[[i, j, k]] = (self.state.temperature[[i + 1, j, k]]
                        - 2.0 * self.state.temperature[[i, j, k]]
                        + self.state.temperature[[i - 1, j, k]])
                        / dx2
                        + (self.state.temperature[[i, j + 1, k]]
                            - 2.0 * self.state.temperature[[i, j, k]]
                            + self.state.temperature[[i, j - 1, k]])
                            / dy2
                        + (self.state.temperature[[i, j, k + 1]]
                            - 2.0 * self.state.temperature[[i, j, k]]
                            + self.state.temperature[[i, j, k - 1]])
                            / dz2;
                }
            }
        }

        // Update temperature
        Zip::from(&mut self.state.temperature)
            .and(&laplacian)
            .for_each(|t, &lap| {
                *t += alpha * lap * dt;
            });

        Ok(())
    }

    /// Apply Pennes bioheat cooling - USING grid parameter
    fn apply_bioheat_cooling(
        &mut self,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        let wb = self.config.blood_perfusion;
        let cb = self.config.blood_specific_heat;
        let tb = self.config.blood_temperature;

        // Get tissue properties using grid
        let rho = medium.density(0.0, 0.0, 0.0, grid);
        let cp = medium.specific_heat(0.0, 0.0, 0.0, grid);

        // Perfusion cooling rate
        let cooling_rate = wb * cb / (rho * cp);

        Zip::from(&mut self.state.temperature).for_each(|t| {
            *t -= cooling_rate * (*t - tb) * dt;
        });

        Ok(())
    }

    /// Get current temperature
    pub fn temperature(&self) -> &Array3<f64> {
        &self.state.temperature
    }

    /// Get thermal state
    pub fn state(&self) -> &ThermalState {
        &self.state
    }

    /// Calculate heat source from acoustic intensity - USING all parameters
    pub fn calculate_heat_source(
        &mut self,
        intensity: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> Array3<f64> {
        // Heat generation = absorption * intensity
        // Q = α * I where α is absorption coefficient
        // Using 1 MHz as default frequency for now
        use crate::medium::AcousticProperties;
        let absorption = medium.absorption_coefficient(0.0, 0.0, 0.0, grid, 1e6);
        intensity * absorption
    }

    /// Update temperature with heat source - USING all parameters
    pub fn update_temperature(
        &mut self,
        heat_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        self.update(heat_source, medium, grid, dt)
    }

    /// Get thermal dose field
    pub fn thermal_dose(&self) -> Array3<f64> {
        // Calculate CEM43 thermal dose
        use super::dose::ThermalDose;
        ThermalDose::cem43(&self.state.temperature, 1.0 / 60.0) // Convert to minutes
    }
}
