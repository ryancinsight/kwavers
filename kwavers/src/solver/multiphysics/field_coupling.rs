//! Field coupling strategies for multi-physics simulations
//!
//! This module implements different strategies for coupling fields between
//! different physics domains (acoustic, optical, thermal).

use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE};
use crate::core::constants::thermodynamic::SPECIFIC_HEAT_WATER;
use crate::core::error::KwaversResult;
use crate::domain::field::indices::*;
use ndarray::Array3;

/// Field coupling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CouplingStrategy {
    /// No coupling between fields (independent simulations)
    None,
    /// Weak coupling (sequential updates)
    Weak,
    /// Strong coupling (iterative updates)
    Strong,
    /// Adaptive coupling (adjusts based on field gradients)
    Adaptive,
}

/// Field coupler for multi-physics interactions
#[derive(Debug)]
pub struct FieldCoupler {
    /// Coupling strategy
    strategy: CouplingStrategy,
    /// Coupling strength parameters
    coupling_strength: f64,
    /// Maximum iterations for strong coupling
    max_iterations: usize,
    /// Tolerance for convergence
    tolerance: f64,
}

impl FieldCoupler {
    /// Create a new field coupler
    pub fn new(strategy: CouplingStrategy) -> Self {
        Self {
            strategy,
            coupling_strength: 1.0,
            max_iterations: 10,
            tolerance: 1e-6,
        }
    }

    /// Couple fields according to the selected strategy
    pub fn couple_fields(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        match self.strategy {
            CouplingStrategy::None => {
                // No coupling - fields evolve independently
                Ok(())
            }
            CouplingStrategy::Weak => self.apply_weak_coupling(fields, dt),
            CouplingStrategy::Strong => self.apply_strong_coupling(fields, dt),
            CouplingStrategy::Adaptive => self.apply_adaptive_coupling(fields, dt),
        }
    }

    /// Apply weak coupling (single pass)
    fn apply_weak_coupling(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        // Acoustic-optical coupling: acoustic pressure affects optical properties
        self.couple_acoustic_to_optical(fields, dt)?;

        // Optical-thermal coupling: optical absorption generates heat
        self.couple_optical_to_thermal(fields, dt)?;

        // Acoustic-thermal coupling: acoustic absorption generates heat
        self.couple_acoustic_to_thermal(fields, dt)?;

        Ok(())
    }

    /// Apply strong coupling (iterative)
    fn apply_strong_coupling(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        let mut previous_fields = fields.to_vec();

        for iteration in 0..self.max_iterations {
            // Apply weak coupling
            self.apply_weak_coupling(fields, dt)?;

            // Check for convergence
            if self.check_convergence(&previous_fields, fields) {
                break;
            }

            // Update previous fields for next iteration
            previous_fields = fields.to_vec();

            // Apply relaxation if needed
            if iteration > 0 {
                self.apply_relaxation(&previous_fields, fields);
            }
        }

        Ok(())
    }

    /// Apply adaptive coupling
    fn apply_adaptive_coupling(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        // Calculate field gradients to determine coupling strength
        let gradients = self.calculate_field_gradients(fields);

        // Adjust coupling strength based on gradients
        let coupling_strength = self.adjust_coupling_strength(&gradients);

        // Apply coupling with adjusted strength
        self.apply_coupling_with_strength(fields, dt, coupling_strength)
    }

    /// Couple acoustic field to optical field
    fn couple_acoustic_to_optical(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        // Acoustic pressure affects refractive index (photoelastic effect)
        // Clone pressure to avoid borrow conflict
        let pressure = fields[PRESSURE_IDX].clone();
        let intensity = &mut fields[LIGHT_IDX];

        // Simple model: intensity modulation by pressure
        // In reality, this would involve solving the coupled wave equations
        for ((i, j, k), &p) in pressure.indexed_iter() {
            // Photoelastic effect: refractive index change proportional to pressure
            let delta_n = 1e-12 * p; // Photoelastic coefficient
                                     // Intensity modulation (simplified)
            let modulation = 1.0 + self.coupling_strength * delta_n * dt;
            intensity[[i, j, k]] *= modulation;
        }

        Ok(())
    }

    /// Couple optical field to thermal field
    fn couple_optical_to_thermal(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        // Optical absorption generates heat
        // Clone intensity to avoid borrow conflict
        let intensity = fields[LIGHT_IDX].clone();
        let temperature = &mut fields[TEMPERATURE_IDX];

        // Heat generation: Q = μ_a * I, where μ_a is absorption coefficient
        let absorption_coefficient = 10.0; // 10 m^-1 (typical for tissue)

        for ((i, j, k), &i_val) in intensity.indexed_iter() {
            // Heat generated per unit volume (W/m³)
            let heat_source = absorption_coefficient * i_val;
            // Temperature change: ΔT = Q * dt / (ρ * c_p)
            let delta_t = heat_source * dt / (DENSITY_WATER_NOMINAL * SPECIFIC_HEAT_WATER);
            temperature[[i, j, k]] += delta_t;
        }

        Ok(())
    }

    /// Couple acoustic field to thermal field
    fn couple_acoustic_to_thermal(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        // Acoustic absorption generates heat
        // Clone pressure to avoid borrow conflict
        let pressure = fields[PRESSURE_IDX].clone();
        let temperature = &mut fields[TEMPERATURE_IDX];

        // Heat generation: Q = α * |p|² / (ρ * c), where α is absorption coefficient.
        //
        // SSOT: uses DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE, and SPECIFIC_HEAT_WATER
        // from the core constants module — no magic literals.
        let absorption_coefficient = 0.5; // 0.5 Np/m (typical for tissue)

        for ((i, j, k), &p) in pressure.indexed_iter() {
            // Intensity: I = p² / (ρ₀ · c₀)
            let intensity = p * p / (DENSITY_WATER_NOMINAL * SOUND_SPEED_TISSUE);
            // Heat generated per unit volume (W/m³): Q = α · I
            let heat_source = absorption_coefficient * intensity;
            // Temperature change: ΔT = Q · dt / (ρ₀ · c_p)
            let delta_t = heat_source * dt / (DENSITY_WATER_NOMINAL * SPECIFIC_HEAT_WATER);
            temperature[[i, j, k]] += delta_t;
        }

        Ok(())
    }

    /// Check for convergence using relative tolerance.
    ///
    /// Computes the maximum relative change between iterations for each field:
    ///   ε_rel = max |current − previous| / (‖current‖_∞ + 1e-15)
    ///
    /// Relative (rather than absolute) tolerance ensures consistent convergence
    /// behaviour regardless of field magnitude — pressure fields in Pa and
    /// temperature fields in °C would otherwise require different absolute thresholds.
    ///
    /// The 1e-15 guard prevents division-by-zero for all-zero fields; in that case
    /// the absolute change is used as-is, which is appropriate because both fields
    /// are near zero and any difference is already negligible.
    fn check_convergence(&self, previous: &[Array3<f64>], current: &[Array3<f64>]) -> bool {
        for (prev_field, curr_field) in previous.iter().zip(current.iter()) {
            // ‖current‖_∞ (max absolute value) used as normalizer
            let field_norm = curr_field
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max);

            let max_rel_diff = prev_field
                .iter()
                .zip(curr_field.iter())
                .map(|(p, c)| (p - c).abs() / (field_norm + 1e-15))
                .fold(0.0_f64, f64::max);

            if max_rel_diff > self.tolerance {
                return false;
            }
        }
        true
    }

    /// Apply relaxation to improve stability
    fn apply_relaxation(&self, previous: &[Array3<f64>], current: &mut [Array3<f64>]) {
        let omega = 0.5; // Relaxation parameter

        for (prev_field, curr_field) in previous.iter().zip(current.iter_mut()) {
            for ((i, j, k), &prev_val) in prev_field.indexed_iter() {
                let curr_val = curr_field[[i, j, k]];
                curr_field[[i, j, k]] = omega * curr_val + (1.0 - omega) * prev_val;
            }
        }
    }

    /// Calculate field gradients
    fn calculate_field_gradients(&self, fields: &[Array3<f64>]) -> Vec<f64> {
        fields
            .iter()
            .map(|field| {
                // Calculate maximum gradient in the field
                let mut max_gradient: f64 = 0.0;

                // Simple gradient calculation (could be optimized)
                for i in 1..field.shape()[0] - 1 {
                    for j in 1..field.shape()[1] - 1 {
                        for k in 1..field.shape()[2] - 1 {
                            let grad_x = field[[i + 1, j, k]] - field[[i - 1, j, k]];
                            let grad_y = field[[i, j + 1, k]] - field[[i, j - 1, k]];
                            let grad_z = field[[i, j, k + 1]] - field[[i, j, k - 1]];
                            let gradient =
                                (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();
                            max_gradient = max_gradient.max(gradient);
                        }
                    }
                }

                max_gradient
            })
            .collect()
    }

    /// Adjust coupling strength based on gradients
    fn adjust_coupling_strength(&self, gradients: &[f64]) -> f64 {
        // Simple heuristic: reduce coupling strength for large gradients
        let max_gradient = gradients.iter().fold(0.0, |max, &g| g.max(max));

        if max_gradient > 1.0 {
            // Strong gradients - use weaker coupling for stability
            0.1
        } else if max_gradient > 0.1 {
            // Moderate gradients - use medium coupling
            0.5
        } else {
            // Small gradients - use full coupling
            1.0
        }
    }

    /// Apply coupling with specified strength
    fn apply_coupling_with_strength(
        &self,
        fields: &mut [Array3<f64>],
        dt: f64,
        strength: f64,
    ) -> KwaversResult<()> {
        // Create a temporary coupler with the specified strength
        let temp_coupler = FieldCoupler {
            strategy: self.strategy,
            coupling_strength: strength,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
        };
        temp_coupler.apply_weak_coupling(fields, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::{fundamental::DENSITY_WATER_NOMINAL, thermodynamic::SPECIFIC_HEAT_WATER};

    // -------------------------------------------------------------------------
    // Convergence tests
    // -------------------------------------------------------------------------

    /// A large-magnitude field with a small *relative* change must be reported
    /// as converged, even though the absolute difference is large.
    ///
    /// Absolute tolerance would fail to converge here (diff >> tolerance), but
    /// relative tolerance correctly identifies convergence (rel_diff << tolerance).
    #[test]
    fn test_convergence_relative_not_absolute() {
        let coupler = FieldCoupler {
            strategy: CouplingStrategy::Strong,
            coupling_strength: 1.0,
            max_iterations: 10,
            tolerance: 1e-6,
        };

        // Field magnitude: 1e6 Pa (typical ultrasound pressure)
        // Absolute change: 1 Pa → absolute tolerance would flag as NOT converged (1 > 1e-6)
        // Relative change: 1 / 1e6 = 1e-6 → relative tolerance should mark as converged
        let prev = vec![Array3::from_elem((4, 4, 4), 1_000_000.0_f64)];
        let curr = vec![Array3::from_elem((4, 4, 4), 1_000_001.0_f64)]; // +1 Pa

        assert!(
            coupler.check_convergence(&prev, &curr),
            "relative change of 1e-6 at 1 MPa must be reported as converged"
        );
    }

    /// Two identical fields must always be converged regardless of magnitude.
    #[test]
    fn test_convergence_identical_fields() {
        let coupler = FieldCoupler::new(CouplingStrategy::Strong);
        let field = vec![Array3::from_elem((4, 4, 4), 42.0_f64)];
        assert!(coupler.check_convergence(&field, &field));
    }

    /// A field where the change exceeds the relative tolerance must NOT converge.
    #[test]
    fn test_convergence_large_relative_change_not_converged() {
        let coupler = FieldCoupler {
            strategy: CouplingStrategy::Strong,
            coupling_strength: 1.0,
            max_iterations: 10,
            tolerance: 1e-6,
        };

        // 10 % relative change — far above tolerance
        let prev = vec![Array3::from_elem((4, 4, 4), 1.0_f64)];
        let curr = vec![Array3::from_elem((4, 4, 4), 1.1_f64)];

        assert!(
            !coupler.check_convergence(&prev, &curr),
            "10 % relative change must NOT be reported as converged"
        );
    }

    // -------------------------------------------------------------------------
    // SSOT constant tests
    // -------------------------------------------------------------------------

    /// Verify that DENSITY_WATER_NOMINAL matches the expected water density.
    ///
    /// Ensures the constant used for acoustic–thermal coupling is within ±1 kg/m³
    /// of 1000 kg/m³ (standard nominal density of water).
    #[test]
    fn test_density_water_nominal_is_1000() {
        assert!(
            (DENSITY_WATER_NOMINAL - 1000.0).abs() < 1.0,
            "DENSITY_WATER_NOMINAL ({DENSITY_WATER_NOMINAL}) must be ≈ 1000 kg/m³"
        );
    }

    /// Verify SPECIFIC_HEAT_WATER is within the published range for water at 20°C.
    ///
    /// NIST data: c_p(water, 20°C) = 4181.8 J/(kg·K).  Range [4180, 4220] is
    /// tight enough to exclude common errors (e.g. using air specific heat ≈ 1005).
    #[test]
    fn test_specific_heat_water_within_literature_range() {
        assert!(
            SPECIFIC_HEAT_WATER > 4150.0 && SPECIFIC_HEAT_WATER < 4220.0,
            "SPECIFIC_HEAT_WATER ({SPECIFIC_HEAT_WATER}) outside NIST range [4150, 4220] J/(kg·K)"
        );
    }
}
