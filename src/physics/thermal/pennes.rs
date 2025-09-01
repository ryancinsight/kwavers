//! Pennes bioheat equation solver
//!
//! The Pennes equation models heat transfer in perfused tissue:
//! ρc ∂T/∂t = ∇·(k∇T) + w_b c_b (T_a - T) + Q_m + Q
//!
//! Where Q is the heat source from ultrasound absorption.

use super::ThermalProperties;
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
    /// Tissue properties
    properties: ThermalProperties,
}

impl PennesSolver {
    /// Create new Pennes solver
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        properties: ThermalProperties,
    ) -> Result<Self, String> {
        // Check stability (3D diffusion)
        let alpha = properties.k / (properties.rho * properties.c);
        let stability = alpha * dt * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));

        if stability > 0.5 {
            return Err(format!(
                "Unstable time step: stability number {:.3} > 0.5",
                stability
            ));
        }

        // Initialize with body temperature
        let temperature = Array3::from_elem((nx, ny, nz), properties.T_a);
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
        })
    }

    /// Update temperature for one time step
    /// heat_source: volumetric heat deposition rate (W/m³)
    pub fn step(&mut self, heat_source: &Array3<f64>) {
        let alpha = self.properties.k / (self.properties.rho * self.properties.c);
        let perfusion_term =
            self.properties.w_b * self.properties.c_b / (self.properties.rho * self.properties.c);

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

                    // Pennes equation
                    let T = self.temperature_prev[[i, j, k]];
                    let dT_dt = alpha * laplacian - perfusion_term * (T - self.properties.T_a)
                        + self.properties.Q_m / (self.properties.rho * self.properties.c)
                        + heat_source[[i, j, k]] / (self.properties.rho * self.properties.c);

                    self.temperature[[i, j, k]] = T + self.dt * dT_dt;
                }
            }
        }

        // Apply boundary conditions (insulated for now)
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
    pub fn get_temperature(&self) -> &Array3<f64> {
        &self.temperature
    }

    /// Get maximum temperature
    pub fn get_max_temperature(&self) -> f64 {
        self.temperature.iter().fold(0.0_f64, |a, &b| a.max(b))
    }

    /// Get temperature rise above baseline
    pub fn get_temperature_rise(&self) -> Array3<f64> {
        &self.temperature - self.properties.T_a
    }

    /// Calculate heat source from acoustic intensity
    /// Q = 2αI where α is absorption coefficient and I is intensity
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
        let properties = ThermalProperties::default();
        let solver = PennesSolver::new(32, 32, 32, 1e-3, 1e-3, 1e-3, 0.01, properties);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_steady_state_temperature() {
        let properties = ThermalProperties::default();
        let mut solver =
            PennesSolver::new(16, 16, 16, 1e-3, 1e-3, 1e-3, 0.001, properties).unwrap();

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
        let properties = ThermalProperties {
            w_b: 0.0, // No perfusion for simple test
            ..Default::default()
        };

        let mut solver =
            PennesSolver::new(16, 16, 16, 1e-3, 1e-3, 1e-3, 0.001, properties).unwrap();

        // Apply heat source at center
        let mut heat_source = Array3::zeros((16, 16, 16));
        heat_source[[8, 8, 8]] = 1e6; // 1 MW/m³

        for _ in 0..100 {
            solver.step(&heat_source);
        }

        // Temperature should rise at source
        let center_temp = solver.get_temperature()[[8, 8, 8]];
        assert!(center_temp > 37.5, "No heating detected: {}", center_temp);
    }
}
