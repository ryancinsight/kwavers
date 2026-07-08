//! `ConservationMonitor` — check and accumulate conservation errors over time.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use log::warn;
use ndarray::Array3;

use super::types::{ConservationError, ConservationHistory, ConservedQuantities};

fn for_each_cell((nx, ny, nz): (usize, usize, usize), mut f: impl FnMut(usize, usize, usize)) {
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                f(i, j, k);
            }
        }
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn check_conservation(
        &mut self,
        time: f64,
        quantities: ConservedQuantities,
    ) -> KwaversResult<ConservationError> {
        let initial = self.history.quantities.first().ok_or_else(|| {
            KwaversError::Validation(ValidationError::FieldValidation {
                field: "initial_quantities".to_owned(),
                value: "None".to_owned(),
                constraint: "Must call set_initial() first".to_owned(),
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
            let p_mag = initial
                .momentum
                .2
                .mul_add(
                    initial.momentum.2,
                    initial
                        .momentum
                        .1
                        .mul_add(initial.momentum.1, initial.momentum.0.powi(2)),
                )
                .sqrt();
            dp.2.mul_add(dp.2, dp.1.mul_add(dp.1, dp.0.powi(2))).sqrt() / p_mag.max(1e-10)
        };

        let angular_momentum_error = {
            let dl = (
                quantities.angular_momentum.0 - initial.angular_momentum.0,
                quantities.angular_momentum.1 - initial.angular_momentum.1,
                quantities.angular_momentum.2 - initial.angular_momentum.2,
            );
            let l_mag = initial
                .angular_momentum
                .2
                .mul_add(
                    initial.angular_momentum.2,
                    initial.angular_momentum.1.mul_add(
                        initial.angular_momentum.1,
                        initial.angular_momentum.0.powi(2),
                    ),
                )
                .sqrt();
            dl.2.mul_add(dl.2, dl.1.mul_add(dl.1, dl.0.powi(2))).sqrt() / l_mag.max(1e-10)
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
        assert_eq!(
            pressure.dim(),
            velocity_x.dim(),
            "invariant: pressure and x-velocity shapes must match"
        );
        assert_eq!(
            pressure.dim(),
            velocity_y.dim(),
            "invariant: pressure and y-velocity shapes must match"
        );
        assert_eq!(
            pressure.dim(),
            velocity_z.dim(),
            "invariant: pressure and z-velocity shapes must match"
        );

        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        let mut total_energy = 0.0;

        for_each_cell(pressure.dim(), |i, j, k| {
            let x = i as f64 * self.grid.dx;
            let y = j as f64 * self.grid.dy;
            let z = k as f64 * self.grid.dz;
            let p = pressure[(i, j, k)];
            let vx = velocity_x[(i, j, k)];
            let vy = velocity_y[(i, j, k)];
            let vz = velocity_z[(i, j, k)];

            let density = kwavers_medium::density_at(medium, x, y, z, &self.grid);
            let gamma = medium.gamma(x, y, z, &self.grid);

            // Kinetic energy density
            let kinetic = 0.5 * density * vz.mul_add(vz, vx.mul_add(vx, vy * vy));

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
                assert_eq!(
                    pressure.dim(),
                    vx.dim(),
                    "invariant: pressure and x-velocity shapes must match"
                );
                assert_eq!(
                    pressure.dim(),
                    vy.dim(),
                    "invariant: pressure and y-velocity shapes must match"
                );
                assert_eq!(
                    pressure.dim(),
                    vz.dim(),
                    "invariant: pressure and z-velocity shapes must match"
                );
                for_each_cell(pressure.dim(), |i, j, k| {
                    let x = i as f64 * self.grid.dx;
                    let y = j as f64 * self.grid.dy;
                    let z = k as f64 * self.grid.dz;
                    let p = pressure[(i, j, k)];
                    let vx_val = vx[(i, j, k)];
                    let vy_val = vy[(i, j, k)];
                    let vz_val = vz[(i, j, k)];

                    let density = kwavers_medium::density_at(medium, x, y, z, &self.grid);
                    let sound_speed = kwavers_medium::sound_speed_at(medium, x, y, z, &self.grid);

                    // Potential energy density: Ep = p²/(2ρc²)
                    let potential_energy = p * p / (2.0 * density * sound_speed * sound_speed);

                    // Kinetic energy density: Ek = ρv²/2
                    let kinetic_energy = 0.5
                        * density
                        * vz_val.mul_add(vz_val, vx_val.mul_add(vx_val, vy_val * vy_val));

                    total_energy += (potential_energy + kinetic_energy) * dv;
                });
            }
        } else {
            // Potential energy only
            for_each_cell(pressure.dim(), |i, j, k| {
                let x = i as f64 * self.grid.dx;
                let y = j as f64 * self.grid.dy;
                let z = k as f64 * self.grid.dz;
                let p = pressure[(i, j, k)];

                let density = kwavers_medium::density_at(medium, x, y, z, &self.grid);
                let sound_speed = kwavers_medium::sound_speed_at(medium, x, y, z, &self.grid);

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
                    let p_mag = initial
                        .momentum
                        .2
                        .mul_add(
                            initial.momentum.2,
                            initial
                                .momentum
                                .1
                                .mul_add(initial.momentum.1, initial.momentum.0.powi(2)),
                        )
                        .sqrt();
                    dp.2.mul_add(dp.2, dp.1.mul_add(dp.1, dp.0.powi(2))).sqrt() / p_mag.max(1e-10)
                };

                let angular_momentum_error = {
                    let dl = (
                        quantities.angular_momentum.0 - initial.angular_momentum.0,
                        quantities.angular_momentum.1 - initial.angular_momentum.1,
                        quantities.angular_momentum.2 - initial.angular_momentum.2,
                    );
                    let l_mag = initial
                        .angular_momentum
                        .2
                        .mul_add(
                            initial.angular_momentum.2,
                            initial.angular_momentum.1.mul_add(
                                initial.angular_momentum.1,
                                initial.angular_momentum.0.powi(2),
                            ),
                        )
                        .sqrt();
                    dl.2.mul_add(dl.2, dl.1.mul_add(dl.1, dl.0.powi(2))).sqrt() / l_mag.max(1e-10)
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
