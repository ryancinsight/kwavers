//! Energy Conservation Validation for Wave Solvers
//!
//! This module provides energy conservation tests for validating that
//! numerical solvers preserve the Hamiltonian in conservative systems.
//!
//! # Mathematical Specification
//!
//! For elastic wave propagation without dissipation, the total energy
//! (Hamiltonian) must be conserved:
//!
//! ```text
//! H(t) = K(t) + U(t) = const
//! ```
//!
//! where:
//! - K(t) = (1/2)∫ρ|∂u/∂t|²dV  (kinetic energy)
//! - U(t) = (1/2)∫σ:ε dV       (strain energy)
//!
//! # Invariants
//!
//! 1. **Energy Conservation**: dH/dt = 0 for conservative systems
//! 2. **Positive Definiteness**: H(t) > 0 for non-trivial solutions
//! 3. **Bounded Variation**: H(t) ∈ [H_min, H_max] for finite domains
//! 4. **Equipartition**: Long-term average K̄ ≈ Ū for ergodic systems
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use kwavers::tests::validation::energy::EnergyValidator;
//!
//! let mut validator = EnergyValidator::new(1e-6); // 0.0001% tolerance
//!
//! for time_step in 0..n_steps {
//!     let kinetic = compute_kinetic_energy(&velocity, &density);
//!     let strain = compute_strain_energy(&stress, &strain);
//!
//!     validator.add_measurement(time_step as f64 * dt, kinetic, strain);
//! }
//!
//! let result = validator.validate();
//! assert!(result.is_conserved());
//! ```

use std::f64;

/// Energy validator for conservation testing
#[derive(Debug, Clone)]
pub struct EnergyValidator {
    /// Time points
    times: Vec<f64>,
    /// Kinetic energy K(t)
    kinetic: Vec<f64>,
    /// Strain/potential energy U(t)
    strain: Vec<f64>,
    /// Total energy H(t) = K(t) + U(t)
    total: Vec<f64>,
    /// Relative tolerance for conservation
    tolerance: f64,
}

impl EnergyValidator {
    /// Create new energy validator
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Relative tolerance for energy conservation
    ///   (typical values: 1e-6 to 1e-4)
    pub fn new(tolerance: f64) -> Self {
        Self {
            times: Vec::new(),
            kinetic: Vec::new(),
            strain: Vec::new(),
            total: Vec::new(),
            tolerance,
        }
    }

    /// Add energy measurement at a given time
    ///
    /// # Arguments
    ///
    /// * `time` - Current simulation time
    /// * `kinetic_energy` - Kinetic energy K = (1/2)∫ρ|v|²dV
    /// * `strain_energy` - Strain energy U = (1/2)∫σ:ε dV
    pub fn add_measurement(&mut self, time: f64, kinetic_energy: f64, strain_energy: f64) {
        let total_energy = kinetic_energy + strain_energy;

        self.times.push(time);
        self.kinetic.push(kinetic_energy);
        self.strain.push(strain_energy);
        self.total.push(total_energy);
    }

    /// Validate energy conservation
    ///
    /// # Returns
    ///
    /// Validation result with energy drift statistics
    pub fn validate(&self) -> EnergyValidationResult {
        if self.total.is_empty() {
            return EnergyValidationResult {
                is_conserved: false,
                initial_energy: 0.0,
                final_energy: 0.0,
                max_deviation: 0.0,
                relative_drift: f64::INFINITY,
                mean_kinetic: 0.0,
                mean_strain: 0.0,
                tolerance: self.tolerance,
                details: "No measurements recorded".to_string(),
            };
        }

        let initial_energy = self.total[0];
        let final_energy = *self.total.last().unwrap();

        // Compute maximum absolute deviation from initial energy
        let max_deviation = self
            .total
            .iter()
            .map(|&h| (h - initial_energy).abs())
            .fold(0.0, f64::max);

        // Relative energy drift: max|H(t) - H(0)| / H(0)
        let relative_drift = if initial_energy.abs() > 1e-15 {
            max_deviation / initial_energy.abs()
        } else {
            // If initial energy is near zero, use absolute deviation
            max_deviation
        };

        // Mean kinetic and strain energies
        let mean_kinetic = self.kinetic.iter().sum::<f64>() / self.kinetic.len() as f64;
        let mean_strain = self.strain.iter().sum::<f64>() / self.strain.len() as f64;

        let is_conserved = relative_drift <= self.tolerance;

        let details = if is_conserved {
            format!(
                "Energy conserved: H(0) = {:.3e}, drift = {:.3e} ({:.2}%)",
                initial_energy,
                max_deviation,
                relative_drift * 100.0
            )
        } else {
            format!(
                "Energy NOT conserved: H(0) = {:.3e}, drift = {:.3e} ({:.2}%), exceeds tolerance {:.2}%",
                initial_energy,
                max_deviation,
                relative_drift * 100.0,
                self.tolerance * 100.0
            )
        };

        EnergyValidationResult {
            is_conserved,
            initial_energy,
            final_energy,
            max_deviation,
            relative_drift,
            mean_kinetic,
            mean_strain,
            tolerance: self.tolerance,
            details,
        }
    }

    /// Compute energy drift rate: dH/dt
    ///
    /// # Returns
    ///
    /// Vector of energy drift rates at each time step
    pub fn compute_drift_rate(&self) -> Vec<f64> {
        if self.times.len() < 2 {
            return Vec::new();
        }

        let mut drift_rates = Vec::with_capacity(self.times.len() - 1);

        for i in 0..self.times.len() - 1 {
            let dt = self.times[i + 1] - self.times[i];
            let dh = self.total[i + 1] - self.total[i];

            if dt > 0.0 {
                drift_rates.push(dh / dt);
            } else {
                drift_rates.push(0.0);
            }
        }

        drift_rates
    }

    /// Check equipartition: K̄ ≈ Ū (long-term average)
    ///
    /// # Returns
    ///
    /// Equipartition ratio K̄/Ū (should be ≈1 for ergodic systems)
    pub fn equipartition_ratio(&self) -> f64 {
        if self.kinetic.is_empty() || self.strain.is_empty() {
            return f64::NAN;
        }

        let mean_kinetic = self.kinetic.iter().sum::<f64>() / self.kinetic.len() as f64;
        let mean_strain = self.strain.iter().sum::<f64>() / self.strain.len() as f64;

        if mean_strain.abs() > 1e-15 {
            mean_kinetic / mean_strain
        } else {
            f64::INFINITY
        }
    }

    /// Get energy time series for plotting/analysis
    pub fn time_series(&self) -> (&[f64], &[f64], &[f64], &[f64]) {
        (&self.times, &self.kinetic, &self.strain, &self.total)
    }
}

/// Result of energy conservation validation
#[derive(Debug, Clone)]
pub struct EnergyValidationResult {
    /// Energy is conserved within tolerance
    pub is_conserved: bool,
    /// Initial total energy H(0)
    pub initial_energy: f64,
    /// Final total energy H(T)
    pub final_energy: f64,
    /// Maximum absolute deviation: max|H(t) - H(0)|
    pub max_deviation: f64,
    /// Relative energy drift: max_deviation / |H(0)|
    pub relative_drift: f64,
    /// Time-averaged kinetic energy
    pub mean_kinetic: f64,
    /// Time-averaged strain energy
    pub mean_strain: f64,
    /// Acceptance tolerance
    pub tolerance: f64,
    /// Human-readable details
    pub details: String,
}

impl EnergyValidationResult {
    /// Check if result passes validation
    pub fn passed(&self) -> bool {
        self.is_conserved
    }
}

/// Compute kinetic energy from velocity field
///
/// # Mathematical Specification
///
/// K = (1/2)∫ρ|v|²dV ≈ (1/2)Σᵢ ρᵢ|vᵢ|² Δ Vᵢ
///
/// # Arguments
///
/// * `velocity` - Velocity field as flat array [v0_x, v0_y, ..., vN_x, vN_y]
/// * `density` - Density at each grid point
/// * `cell_volume` - Volume element ΔV (e.g., Δx·Δy·Δz)
/// * `n_components` - Number of velocity components (2 for 2D, 3 for 3D)
///
/// # Returns
///
/// Total kinetic energy in Joules
pub fn compute_kinetic_energy(
    velocity: &[f64],
    density: &[f64],
    cell_volume: f64,
    n_components: usize,
) -> f64 {
    let n_points = velocity.len() / n_components;
    assert_eq!(
        density.len(),
        n_points,
        "Density array must match number of spatial points"
    );

    let mut kinetic = 0.0;

    for i in 0..n_points {
        let rho = density[i];

        // Compute |v|² = v_x² + v_y² + v_z²
        let mut v_squared = 0.0;
        for c in 0..n_components {
            let v_component = velocity[i * n_components + c];
            v_squared += v_component * v_component;
        }

        kinetic += 0.5 * rho * v_squared * cell_volume;
    }

    kinetic
}

/// Compute strain energy from stress and strain tensors
///
/// # Mathematical Specification
///
/// U = (1/2)∫σ:ε dV = (1/2)∫σᵢⱼεᵢⱼ dV
///
/// For linear elasticity:
/// U = (1/2)Σᵢ (σ:ε)ᵢ ΔVᵢ
///
/// # Arguments
///
/// * `stress` - Stress tensor in Voigt notation [σxx, σyy, σzz, σxy, σyz, σzx]
/// * `strain` - Strain tensor in Voigt notation [εxx, εyy, εzz, 2εxy, 2εyz, 2εzx]
/// * `cell_volume` - Volume element ΔV
/// * `n_components` - Number of stress/strain components (3 for 2D, 6 for 3D)
///
/// # Returns
///
/// Total strain energy in Joules
pub fn compute_strain_energy(
    stress: &[f64],
    strain: &[f64],
    cell_volume: f64,
    n_components: usize,
) -> f64 {
    assert_eq!(
        stress.len(),
        strain.len(),
        "Stress and strain arrays must have same length"
    );

    let n_points = stress.len() / n_components;
    let mut strain_energy = 0.0;

    for i in 0..n_points {
        let mut sigma_epsilon = 0.0;

        for c in 0..n_components {
            let sigma = stress[i * n_components + c];
            let epsilon = strain[i * n_components + c];

            // Voigt notation: shear components already doubled in strain
            if c < 3 {
                // Normal components: σxx*εxx, σyy*εyy, σzz*εzz
                sigma_epsilon += sigma * epsilon;
            } else {
                // Shear components: σxy*(2εxy), etc.
                // But in Voigt notation, strain already contains 2εxy
                sigma_epsilon += sigma * epsilon;
            }
        }

        strain_energy += 0.5 * sigma_epsilon * cell_volume;
    }

    strain_energy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_validator_perfect_conservation() {
        let mut validator = EnergyValidator::new(1e-6);

        // Perfect conservation: H = 1.0 for all time
        for i in 0..100 {
            let t = i as f64 * 0.01;
            validator.add_measurement(t, 0.5, 0.5);
        }

        let result = validator.validate();
        assert!(result.is_conserved);
        assert_eq!(result.max_deviation, 0.0);
        assert_eq!(result.relative_drift, 0.0);
    }

    #[test]
    fn test_energy_validator_within_tolerance() {
        let mut validator = EnergyValidator::new(0.01); // 1% tolerance

        // Small drift: H varies from 1.0 to 1.005
        for i in 0..100 {
            let t = i as f64 * 0.01;
            let kinetic = 0.5 + 0.0025 * (t * 2.0 * std::f64::consts::PI).sin();
            let strain = 0.5 - 0.0025 * (t * 2.0 * std::f64::consts::PI).sin();
            validator.add_measurement(t, kinetic, strain);
        }

        let result = validator.validate();
        assert!(result.is_conserved);
        assert!(result.relative_drift < 0.01);
    }

    #[test]
    fn test_energy_validator_exceeds_tolerance() {
        let mut validator = EnergyValidator::new(0.01); // 1% tolerance

        // Large drift: H increases by 5%
        for i in 0..100 {
            let t = i as f64 * 0.01;
            let total = 1.0 + 0.05 * t;
            validator.add_measurement(t, total / 2.0, total / 2.0);
        }

        let result = validator.validate();
        assert!(!result.is_conserved);
        assert!(result.relative_drift > 0.01);
    }

    #[test]
    fn test_kinetic_energy_2d() {
        // 2 points, 2 components each: [vx0, vy0, vx1, vy1]
        let velocity = vec![1.0, 0.0, 0.0, 1.0];
        let density = vec![1000.0, 1000.0];
        let cell_volume = 0.001; // 1mm³

        let kinetic = compute_kinetic_energy(&velocity, &density, cell_volume, 2);

        // K = (1/2) * 1000 * 1.0² * 0.001 + (1/2) * 1000 * 1.0² * 0.001
        //   = 0.5 + 0.5 = 1.0 J
        assert!((kinetic - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kinetic_energy_3d() {
        // 1 point, 3 components: [vx, vy, vz]
        let velocity = vec![3.0, 4.0, 0.0];
        let density = vec![1000.0];
        let cell_volume = 0.001;

        let kinetic = compute_kinetic_energy(&velocity, &density, cell_volume, 3);

        // |v|² = 9 + 16 + 0 = 25
        // K = (1/2) * 1000 * 25 * 0.001 = 12.5 J
        assert!((kinetic - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_strain_energy_2d() {
        // 1 point, 3 components: [εxx, εyy, 2εxy]
        let strain = vec![0.001, 0.001, 0.0];
        // E = 200 GPa, ν = 0.3
        // λ = Eν/((1+ν)(1-2ν)) ≈ 115.38 GPa
        // μ = E/(2(1+ν)) ≈ 76.92 GPa
        // σxx = (λ+2μ)εxx + λεyy = 269.23 * 0.001 + 115.38 * 0.001 ≈ 0.385 MPa
        let stress = vec![0.385e6, 0.385e6, 0.0];
        let cell_volume = 1e-6; // 1mm³

        let strain_energy = compute_strain_energy(&stress, &strain, cell_volume, 3);

        // U = (1/2) * (σxx*εxx + σyy*εyy) * ΔV
        //   = (1/2) * (0.385e6 * 0.001 + 0.385e6 * 0.001) * 1e-6
        //   = (1/2) * 770 * 1e-6 = 3.85e-4 J
        assert!((strain_energy - 3.85e-4).abs() / 3.85e-4 < 0.01);
    }

    #[test]
    fn test_equipartition_ratio() {
        let mut validator = EnergyValidator::new(1e-6);

        // Equal kinetic and strain: K = U = 0.5
        for i in 0..100 {
            let t = i as f64 * 0.01;
            validator.add_measurement(t, 0.5, 0.5);
        }

        let ratio = validator.equipartition_ratio();
        assert!((ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_drift_rate() {
        let mut validator = EnergyValidator::new(1e-6);

        // Linear drift: H(t) = 1.0 + 0.1*t => dH/dt = 0.1
        for i in 0..10 {
            let t = i as f64;
            let total = 1.0 + 0.1 * t;
            validator.add_measurement(t, total / 2.0, total / 2.0);
        }

        let drift_rates = validator.compute_drift_rate();

        for &rate in &drift_rates {
            assert!((rate - 0.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_energy_validator_empty() {
        let validator = EnergyValidator::new(1e-6);
        let result = validator.validate();

        assert!(!result.is_conserved);
        assert!(result.relative_drift.is_infinite());
    }
}
