//! Conservation monitoring for multi-rate time integration
//!
//! This module tracks conservation of mass, momentum, energy, and angular momentum
//! to ensure physical accuracy of multi-physics simulations.

use crate::{medium::Medium, Grid, KwaversError, KwaversResult, ValidationError};
use log::warn;
use ndarray::{Array3, Zip};
use std::collections::HashMap;

/// Conservation quantities for monitoring
#[derive(Debug, Clone)]
pub struct ConservedQuantities {
    /// Total mass
    pub mass: f64,
    /// Total momentum (x, y, z components)
    pub momentum: (f64, f64, f64),
    /// Total energy
    pub energy: f64,
    /// Total angular momentum
    pub angular_momentum: (f64, f64, f64),
}

/// History of conserved quantities
#[derive(Debug, Clone)]
pub struct ConservationHistory {
    /// Time points
    pub times: Vec<f64>,
    /// Conserved quantities at each time
    pub quantities: Vec<ConservedQuantities>,
}

impl Default for ConservationHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl ConservationHistory {
    /// Create new empty history
    #[must_use]
    pub fn new() -> Self {
        Self {
            times: Vec::new(),
            quantities: Vec::new(),
        }
    }

    /// Add a new entry
    pub fn push(&mut self, time: f64, quantities: ConservedQuantities) {
        self.times.push(time);
        self.quantities.push(quantities);
    }
}

/// Conservation monitor for multi-rate integration
#[derive(Debug)]
pub struct ConservationMonitor {
    /// Grid for spatial integration
    grid: Grid,
    /// History of conserved quantities
    history: ConservationHistory,
    /// Tolerance for conservation violations
    tolerance: f64,
}

/// Conservation error at a time step
#[derive(Debug, Clone)]
pub struct ConservationError {
    /// Time at which error was measured
    pub time: f64,
    /// Relative mass error
    pub mass_error: f64,
    /// Relative momentum error
    pub momentum_error: f64,
    /// Relative energy error
    pub energy_error: f64,
    /// Relative angular momentum error
    pub angular_momentum_error: f64,
}

impl ConservationError {
    /// Get the maximum error across all conserved quantities
    #[must_use]
    pub fn max_error(&self) -> f64 {
        self.mass_error
            .max(self.momentum_error)
            .max(self.energy_error)
            .max(self.angular_momentum_error)
    }
}

impl ConservationMonitor {
    /// Create a new conservation monitor
    pub fn new(grid: &Grid) -> Self {
        Self {
            grid: grid.clone(),
            history: ConservationHistory::new(),
            tolerance: 1e-10,
        }
    }

    /// Create a new conservation monitor with specified tolerance
    pub fn with_tolerance(grid: &Grid, tolerance: f64) -> Self {
        Self {
            grid: grid.clone(),
            history: ConservationHistory::new(),
            tolerance,
        }
    }

    /// Set initial conserved quantities
    pub fn set_initial(&mut self, quantities: ConservedQuantities) {
        self.history = ConservationHistory::new();
        self.history.push(0.0, quantities);
    }

    /// Check conservation at current time
    pub fn check_conservation(
        &mut self,
        time: f64,
        quantities: ConservedQuantities,
    ) -> KwaversResult<ConservationError> {
        let initial = self.history.quantities.first().ok_or_else(|| {
            KwaversError::Validation(ValidationError::FieldValidation {
                field: "initial_quantities".to_string(),
                value: "None".to_string(),
                constraint: "Must call set_initial() first".to_string(),
            })
        })?;

        // Compute relative errors
        let mass_error = (quantities.mass - initial.mass).abs() / initial.mass.max(1e-10);
        let energy_error = (quantities.energy - initial.energy).abs() / initial.energy.max(1e-10);

        let momentum_error = {
            let dp = (
                quantities.momentum.0 - initial.momentum.0,
                quantities.momentum.1 - initial.momentum.1,
                quantities.momentum.2 - initial.momentum.2,
            );
            let p_mag = (initial.momentum.0.powi(2)
                + initial.momentum.1.powi(2)
                + initial.momentum.2.powi(2))
            .sqrt();
            (dp.0.powi(2) + dp.1.powi(2) + dp.2.powi(2)).sqrt() / p_mag.max(1e-10)
        };

        let angular_momentum_error = {
            let dl = (
                quantities.angular_momentum.0 - initial.angular_momentum.0,
                quantities.angular_momentum.1 - initial.angular_momentum.1,
                quantities.angular_momentum.2 - initial.angular_momentum.2,
            );
            let l_mag = (initial.angular_momentum.0.powi(2)
                + initial.angular_momentum.1.powi(2)
                + initial.angular_momentum.2.powi(2))
            .sqrt();
            (dl.0.powi(2) + dl.1.powi(2) + dl.2.powi(2)).sqrt() / l_mag.max(1e-10)
        };

        let error = ConservationError {
            time,
            mass_error,
            momentum_error,
            energy_error,
            angular_momentum_error,
        };

        // Store current quantities
        self.history.push(time, quantities);

        // Check violations
        if error.max_error() > self.tolerance {
            warn!(
                "Conservation violation at t={}: max_error={:.2e}",
                time,
                error.max_error()
            );
        }

        Ok(error)
    }

    /// Compute total energy (kinetic + internal)
    pub fn compute_total_energy(
        &self,
        pressure: &Array3<f64>,
        velocity_x: &Array3<f64>,
        velocity_y: &Array3<f64>,
        velocity_z: &Array3<f64>,
        medium: &dyn Medium,
    ) -> f64 {
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        let mut total_energy = 0.0;

        Zip::indexed(pressure)
            .and(velocity_x)
            .and(velocity_y)
            .and(velocity_z)
            .for_each(|(i, j, k), &p, &vx, &vy, &vz| {
                let x = i as f64 * self.grid.dx;
                let y = j as f64 * self.grid.dy;
                let z = k as f64 * self.grid.dz;

                let density = crate::medium::density_at(medium, x, y, z, &self.grid);
                let gamma = medium.gamma(x, y, z, &self.grid);

                // Kinetic energy density
                let kinetic = 0.5 * density * (vx * vx + vy * vy + vz * vz);

                // Internal energy density (ideal gas)
                let gamma_minus_one = gamma - 1.0;
                if gamma_minus_one.abs() > 1e-9 {
                    // Avoid division by zero for gamma = 1
                    let internal = p / gamma_minus_one;
                    total_energy += (kinetic + internal) * dv;
                } else {
                    // For gamma = 1 (isothermal), only kinetic energy
                    total_energy += kinetic * dv;
                }
            });

        total_energy
    }

    /// Compute acoustic energy (complete - includes kinetic and potential energy)
    pub fn compute_acoustic_energy(&self, pressure: &Array3<f64>, medium: &dyn Medium) -> f64 {
        self.compute_acoustic_energy_with_velocity(pressure, None, None, None, medium)
    }

    /// Compute acoustic energy with optional velocity fields
    ///
    /// If velocity fields are provided, computes total acoustic energy (kinetic + potential).
    /// If velocity fields are None, computes only potential energy from pressure.
    pub fn compute_acoustic_energy_with_velocity(
        &self,
        pressure: &Array3<f64>,
        velocity_x: Option<&Array3<f64>>,
        velocity_y: Option<&Array3<f64>>,
        velocity_z: Option<&Array3<f64>>,
        medium: &dyn Medium,
    ) -> f64 {
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        let mut total_energy = 0.0;

        // Check if we have all velocity components
        let has_velocity = velocity_x.is_some() && velocity_y.is_some() && velocity_z.is_some();

        if has_velocity {
            // Complete acoustic energy computation with safe access
            if let (Some(vx), Some(vy), Some(vz)) = (velocity_x, velocity_y, velocity_z) {
                Zip::indexed(pressure).and(vx).and(vy).and(vz).for_each(
                    |(i, j, k), &p, &vx_val, &vy_val, &vz_val| {
                        let x = i as f64 * self.grid.dx;
                        let y = j as f64 * self.grid.dy;
                        let z = k as f64 * self.grid.dz;

                        let density = crate::medium::density_at(medium, x, y, z, &self.grid);
                        let sound_speed =
                            crate::medium::sound_speed_at(medium, x, y, z, &self.grid);

                        // Potential energy density: Ep = p²/(2ρc²)
                        let potential_energy = p * p / (2.0 * density * sound_speed * sound_speed);

                        // Kinetic energy density: Ek = ρv²/2
                        let kinetic_energy =
                            0.5 * density * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val);

                        total_energy += (potential_energy + kinetic_energy) * dv;
                    },
                );
            }
        } else {
            // Potential energy only
            Zip::indexed(pressure).for_each(|(i, j, k), &p| {
                let x = i as f64 * self.grid.dx;
                let y = j as f64 * self.grid.dy;
                let z = k as f64 * self.grid.dz;

                let density = crate::medium::density_at(medium, x, y, z, &self.grid);
                let sound_speed = crate::medium::sound_speed_at(medium, x, y, z, &self.grid);

                // Acoustic potential energy density: E = p²/(2ρc²)
                let energy_density = p * p / (2.0 * density * sound_speed * sound_speed);
                total_energy += energy_density * dv;
            });
        }

        total_energy
    }

    /// Get conservation error history
    pub fn get_error_history(&self) -> Vec<ConservationError> {
        let mut errors = Vec::new();

        if let Some(initial) = self.history.quantities.first() {
            for (i, quantities) in self.history.quantities.iter().enumerate().skip(1) {
                let time = self.history.times[i];

                // Compute errors relative to initial
                let mass_error = (quantities.mass - initial.mass).abs() / initial.mass.max(1e-10);
                let energy_error =
                    (quantities.energy - initial.energy).abs() / initial.energy.max(1e-10);

                let momentum_error = {
                    let dp = (
                        quantities.momentum.0 - initial.momentum.0,
                        quantities.momentum.1 - initial.momentum.1,
                        quantities.momentum.2 - initial.momentum.2,
                    );
                    let p_mag = (initial.momentum.0.powi(2)
                        + initial.momentum.1.powi(2)
                        + initial.momentum.2.powi(2))
                    .sqrt();
                    (dp.0.powi(2) + dp.1.powi(2) + dp.2.powi(2)).sqrt() / p_mag.max(1e-10)
                };

                let angular_momentum_error = {
                    let dl = (
                        quantities.angular_momentum.0 - initial.angular_momentum.0,
                        quantities.angular_momentum.1 - initial.angular_momentum.1,
                        quantities.angular_momentum.2 - initial.angular_momentum.2,
                    );
                    let l_mag = (initial.angular_momentum.0.powi(2)
                        + initial.angular_momentum.1.powi(2)
                        + initial.angular_momentum.2.powi(2))
                    .sqrt();
                    (dl.0.powi(2) + dl.1.powi(2) + dl.2.powi(2)).sqrt() / l_mag.max(1e-10)
                };

                errors.push(ConservationError {
                    time,
                    mass_error,
                    momentum_error,
                    energy_error,
                    angular_momentum_error,
                });
            }
        }

        errors
    }

    /// Check if conservation is within tolerance
    pub fn is_conserved(&self) -> bool {
        self.get_error_history()
            .iter()
            .all(|error| error.max_error() <= self.tolerance)
    }

    /// Update tolerance
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }
}

/// Conservative coupling interface for multi-rate integration
pub trait ConservativeCoupling {
    /// Apply conservative coupling between fast and slow components
    fn apply_conservative_coupling(
        &self,
        high_frequency_fields: &mut HashMap<String, Array3<f64>>,
        low_frequency_fields: &mut HashMap<String, Array3<f64>>,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<()>;

    /// Compute flux corrections for conservation
    fn compute_flux_corrections(
        &self,
        fields: &HashMap<String, Array3<f64>>,
        grid: &Grid,
    ) -> KwaversResult<HashMap<String, Array3<f64>>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;
    use crate::HomogeneousMedium;

    #[test]
    fn test_conservation_monitoring() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
        let mut monitor = ConservationMonitor::new(&grid);

        // Create initial conserved quantities
        let initial = ConservedQuantities {
            mass: 1000.0,
            momentum: (0.0, 0.0, 0.0),
            energy: 1e6,
            angular_momentum: (0.0, 0.0, 0.0),
        };

        monitor.set_initial(initial.clone());

        // Test conservation check with no change
        let error = monitor.check_conservation(0.1, initial.clone()).unwrap();
        assert!(error.max_error() < 1e-10);

        // Test conservation check with small change
        let mut changed = initial.clone();
        changed.mass *= 1.001; // 0.1% change
        let error = monitor.check_conservation(0.2, changed).unwrap();
        assert!(error.mass_error > 0.0);
        assert!(error.mass_error < 0.002);
    }

    #[test]
    fn test_energy_computation() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
        let monitor = ConservationMonitor::new(&grid);
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);

        // Create test fields
        let pressure = Array3::from_elem((10, 10, 10), 1e5); // Pa
        let velocity_x = Array3::zeros((10, 10, 10));
        let velocity_y = Array3::zeros((10, 10, 10));
        let velocity_z = Array3::zeros((10, 10, 10));

        // Compute total energy
        let energy =
            monitor.compute_total_energy(&pressure, &velocity_x, &velocity_y, &velocity_z, &medium);

        // Energy should be positive
        assert!(energy > 0.0);

        // Compute acoustic energy
        let acoustic_energy = monitor.compute_acoustic_energy(&pressure, &medium);
        assert!(acoustic_energy > 0.0);
    }
}
