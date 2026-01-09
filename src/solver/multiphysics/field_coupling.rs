//! Field coupling strategies for multi-physics simulations
//!
//! This module implements different strategies for coupling fields between
//! different physics domains (acoustic, optical, thermal).

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
            // Temperature change: ΔT = Q * dt / (ρ * c), where ρ is density, c is specific heat
            let rho = 1000.0; // kg/m³ (water density)
            let c = 4186.0; // J/(kg·K) (water specific heat)
            let delta_t = heat_source * dt / (rho * c);
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

        // Heat generation: Q = α * |p|² / (ρ * c), where α is absorption coefficient
        let absorption_coefficient = 0.5; // 0.5 Np/m (typical for tissue)
        let rho = 1000.0; // kg/m³
        let c = 1540.0; // m/s (sound speed)

        for ((i, j, k), &p) in pressure.indexed_iter() {
            // Intensity: I = p² / (ρ * c)
            let intensity = p * p / (rho * c);
            // Heat generated per unit volume
            let heat_source = absorption_coefficient * intensity;
            // Temperature change
            let specific_heat = 4186.0; // J/(kg·K)
            let delta_t = heat_source * dt / (rho * specific_heat);
            temperature[[i, j, k]] += delta_t;
        }

        Ok(())
    }

    /// Check for convergence
    fn check_convergence(&self, previous: &[Array3<f64>], current: &[Array3<f64>]) -> bool {
        for (prev_field, curr_field) in previous.iter().zip(current.iter()) {
            let max_diff = prev_field
                .iter()
                .zip(curr_field.iter())
                .map(|(p, c)| (p - c).abs())
                .fold(0.0, f64::max);

            if max_diff > self.tolerance {
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
