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
        if !properties.has_bioheat_parameters() {
            return Err(
                "ThermalPropertyData must have blood_perfusion and blood_specific_heat for Pennes solver".to_string()
            );
        }

        let alpha = properties.thermal_diffusivity();
        let stability = alpha * dt * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));

        if stability > 0.5 {
            return Err(format!(
                "Unstable time step: stability number {stability:.3} > 0.5"
            ));
        }

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

        let w_b = self
            .properties
            .blood_perfusion
            .expect("blood_perfusion validated in constructor");
        let c_b = self
            .properties
            .blood_specific_heat
            .expect("blood_specific_heat validated in constructor");
        let perfusion_term = w_b * c_b / (self.properties.density * self.properties.specific_heat);

        self.temperature_prev.assign(&self.temperature);

        for k in 1..self.nz - 1 {
            for j in 1..self.ny - 1 {
                for i in 1..self.nx - 1 {
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

        self.apply_boundary_conditions();
    }

    fn apply_boundary_conditions(&mut self) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                self.temperature[[0, j, k]] = self.temperature[[1, j, k]];
                self.temperature[[self.nx - 1, j, k]] = self.temperature[[self.nx - 2, j, k]];
            }
        }

        for k in 0..self.nz {
            for i in 0..self.nx {
                self.temperature[[i, 0, k]] = self.temperature[[i, 1, k]];
                self.temperature[[i, self.ny - 1, k]] = self.temperature[[i, self.ny - 2, k]];
            }
        }

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
    pub fn acoustic_heat_source(intensity: &Array3<f64>, absorption: f64) -> Array3<f64> {
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
            0.01,
            properties,
            arterial_temp,
            metabolic_heat,
        )
        .unwrap();

        let heat_source = Array3::zeros((16, 16, 16));
        for _ in 0..100 {
            solver.step(&heat_source);
        }

        let max_temp = solver.get_max_temperature();
        assert!(max_temp >= arterial_temp);
    }
}
