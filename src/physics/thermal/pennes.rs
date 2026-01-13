//! Pennes bioheat equation solver
//!
//! The Pennes equation models heat transfer in perfused tissue:
//! ρc ∂T/∂t = ∇·(k∇T) + `w_b` `c_b` (`T_a` - T) + `Q_m` + Q
//!
//! Where Q is the heat source from ultrasound absorption.

use crate::domain::medium::properties::ThermalPropertyData;
use ndarray::Array3;

/// Pennes bioheat equation solver
#[derive(Debug)]
pub struct PennesSolver {
    /// Temperature field (°C)
    temperature: Array3<f64>,
    /// Previous temperature for time stepping
    temperature_prev: Array3<f64>,
    /// Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,
    /// Grid spacing (m)
    dx: f64,
    dy: f64,
    dz: f64,
    /// Time step (s)
    dt: f64,
    /// Material thermal properties
    properties: ThermalPropertyData,
    /// Arterial blood temperature (°C) - simulation parameter
    arterial_temperature: f64,
    /// Metabolic heat generation (W/m³) - simulation parameter
    metabolic_heat: f64,
}

impl PennesSolver {
    /// Create new Pennes solver
    ///
    /// # Arguments
    ///
    /// * `properties` - Material thermal properties (conductivity, specific heat, density, perfusion)
    /// * `arterial_temperature` - Arterial blood temperature (°C) for perfusion term
    /// * `metabolic_heat` - Metabolic heat generation rate (W/m³)
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        properties: ThermalPropertyData,
        arterial_temperature: f64,
        metabolic_heat: f64,
    ) -> Result<Self, String> {
        // Validate bio-heat parameters if needed
        if !properties.has_bioheat_parameters() {
            return Err(
                "ThermalPropertyData must have blood_perfusion and blood_specific_heat for Pennes solver".to_string()
            );
        }

        // Check stability (3D diffusion)
        let alpha = properties.thermal_diffusivity();
        let stability = alpha * dt * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));

        if stability > 0.5 {
            return Err(format!(
                "Unstable time step: stability number {stability:.3} > 0.5"
            ));
        }

        // Initialize with arterial temperature
        let temperature = Array3::from_elem((nx, ny, nz), arterial_temperature);
        let temperature_prev = temperature.clone();

        Ok(Self {
            temperature,
            temperature_prev,
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            properties,
            arterial_temperature,
            metabolic_heat,
        })
    }

    /// Update temperature for one time step
    /// `heat_source`: volumetric heat deposition rate (W/m³)
    pub fn step(&mut self, heat_source: &Array3<f64>) {
        let alpha = self.properties.thermal_diffusivity();

        // Perfusion coefficient: w_b * c_b / (ρ * c)
        let w_b = self
            .properties
            .blood_perfusion
            .expect("blood_perfusion validated in constructor");
        let c_b = self
            .properties
            .blood_specific_heat
            .expect("blood_specific_heat validated in constructor");
        let perfusion_term = w_b * c_b / (self.properties.density * self.properties.specific_heat);

        // Store current temperature
        self.temperature_prev.assign(&self.temperature);

        // Update interior points using explicit finite differences
        for k in 1..self.nz - 1 {
            for j in 1..self.ny - 1 {
                for i in 1..self.nx - 1 {
                    // Compute Laplacian
                    let laplacian_x = (self.temperature_prev[[i + 1, j, k]]
                        - 2.0 * self.temperature_prev[[i, j, k]]
                        + self.temperature_prev[[i - 1, j, k]])
                        / (self.dx * self.dx);

                    let laplacian_y = (self.temperature_prev[[i, j + 1, k]]
                        - 2.0 * self.temperature_prev[[i, j, k]]
                        + self.temperature_prev[[i, j - 1, k]])
                        / (self.dy * self.dy);

                    let laplacian_z = (self.temperature_prev[[i, j, k + 1]]
                        - 2.0 * self.temperature_prev[[i, j, k]]
                        + self.temperature_prev[[i, j, k - 1]])
                        / (self.dz * self.dz);

                    let laplacian = laplacian_x + laplacian_y + laplacian_z;

                    // Pennes equation: ρc ∂T/∂t = ∇·(k∇T) + w_b c_b (T_a - T) + Q_m + Q
                    let t = self.temperature_prev[[i, j, k]];
                    let dt_dt = alpha * laplacian
                        - perfusion_term * (t - self.arterial_temperature)
                        + self.metabolic_heat
                            / (self.properties.density * self.properties.specific_heat)
                        + heat_source[[i, j, k]]
                            / (self.properties.density * self.properties.specific_heat);

                    self.temperature[[i, j, k]] = t + self.dt * dt_dt;
                }
            }
        }

        // Apply insulated (zero-flux Neumann) boundary conditions
        // Standard for biological tissue per Pennes (1948) thermal modeling
        self.apply_boundary_conditions();
    }

    /// Apply boundary conditions (zero flux)
    fn apply_boundary_conditions(&mut self) {
        // X boundaries
        for k in 0..self.nz {
            for j in 0..self.ny {
                self.temperature[[0, j, k]] = self.temperature[[1, j, k]];
                self.temperature[[self.nx - 1, j, k]] = self.temperature[[self.nx - 2, j, k]];
            }
        }

        // Y boundaries
        for k in 0..self.nz {
            for i in 0..self.nx {
                self.temperature[[i, 0, k]] = self.temperature[[i, 1, k]];
                self.temperature[[i, self.ny - 1, k]] = self.temperature[[i, self.ny - 2, k]];
            }
        }

        // Z boundaries
        for j in 0..self.ny {
            for i in 0..self.nx {
                self.temperature[[i, j, 0]] = self.temperature[[i, j, 1]];
                self.temperature[[i, j, self.nz - 1]] = self.temperature[[i, j, self.nz - 2]];
            }
        }
    }

    /// Get current temperature field
    #[must_use]
    pub fn get_temperature(&self) -> &Array3<f64> {
        &self.temperature
    }

    /// Get maximum temperature
    #[must_use]
    pub fn get_max_temperature(&self) -> f64 {
        self.temperature.iter().fold(0.0_f64, |a, &b| a.max(b))
    }

    /// Get temperature rise above baseline
    #[must_use]
    pub fn get_temperature_rise(&self) -> Array3<f64> {
        &self.temperature - self.arterial_temperature
    }

    /// Calculate heat source from acoustic intensity
    /// Q = 2αI where α is absorption coefficient and I is intensity
    #[must_use]
    pub fn acoustic_heat_source(
        intensity: &Array3<f64>,
        absorption: f64, // Np/m
    ) -> Array3<f64> {
        intensity.mapv(|i| 2.0 * absorption * i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pennes_solver_creation() {
        let properties = ThermalPropertyData::soft_tissue();
        let arterial_temp = 37.0;
        let metabolic_heat = 400.0;
        let solver = PennesSolver::new(
            32,
            32,
            32,
            1e-3,
            1e-3,
            1e-3,
            0.01,
            properties,
            arterial_temp,
            metabolic_heat,
        );
        assert!(solver.is_ok());
    }

    #[test]
    fn test_steady_state_temperature() {
        let properties = ThermalPropertyData::soft_tissue();
        let arterial_temp = 37.0;
        let metabolic_heat = 400.0;
        let mut solver = PennesSolver::new(
            16,
            16,
            16,
            1e-3,
            1e-3,
            1e-3,
            0.001,
            properties,
            arterial_temp,
            metabolic_heat,
        )
        .unwrap();

        // No heat source - should remain at body temperature
        let zero_source = Array3::zeros((16, 16, 16));

        for _ in 0..100 {
            solver.step(&zero_source);
        }

        let max_temp = solver.get_max_temperature();
        assert!(
            (max_temp - 37.0).abs() < 0.01,
            "Temperature drift: {}",
            max_temp
        );
    }

    #[test]
    fn test_heating_with_source() {
        // Create properties without perfusion for simple test
        let properties = ThermalPropertyData::new(
            0.5,          // conductivity
            3600.0,       // specific_heat
            1050.0,       // density
            Some(0.0),    // No perfusion
            Some(3617.0), // blood_specific_heat (required for validation)
        )
        .unwrap();

        let arterial_temp = 37.0;
        let metabolic_heat = 0.0; // No metabolic heat for simple test

        // Use larger time step for faster heating
        let mut solver = PennesSolver::new(
            16,
            16,
            16,
            1e-3,
            1e-3,
            1e-3,
            0.01,
            properties,
            arterial_temp,
            metabolic_heat,
        )
        .unwrap();

        // Apply heat source at center
        let mut heat_source = Array3::zeros((16, 16, 16));
        heat_source[[8, 8, 8]] = 1e7; // 10 MW/m³ for significant heating

        let initial_temp = solver.get_temperature()[[8, 8, 8]];

        // Run for 1 second (100 steps × 0.01s)
        for _ in 0..100 {
            solver.step(&heat_source);
        }

        // Temperature should rise significantly at source
        // Expected: dT/dt = Q/(ρc) = 1e7/(1050*3600) ≈ 2.65 K/s
        // After 1s: ΔT ≈ 2.65 K (accounting for diffusion)
        let center_temp = solver.get_temperature()[[8, 8, 8]];
        assert!(
            center_temp > 38.0,
            "Insufficient heating: initial={:.2}°C, final={:.2}°C",
            initial_temp,
            center_temp
        );
    }
}
