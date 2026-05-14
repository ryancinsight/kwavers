//! Main phase randomizer implementation

use super::constants::{DEFAULT_SEED, MAX_PHASE_SHIFT};
use super::distribution::PhaseDistribution;
use super::scheme::RandomizationScheme;
use ndarray::{Array1, Array2, ArrayView1};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

/// Phase randomizer for standing wave suppression
#[derive(Debug)]
pub struct PhaseRandomizer {
    scheme: RandomizationScheme,
    distribution: PhaseDistribution,
    rng: ChaCha8Rng,
    phase_states: Vec<f64>,
    current_phases: Array1<f64>,
    time_since_switch: f64,
}

impl PhaseRandomizer {
    /// Create a new phase randomizer
    #[must_use]
    pub fn new(
        scheme: RandomizationScheme,
        distribution: PhaseDistribution,
        num_elements: usize,
    ) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(DEFAULT_SEED);
        let phase_states = Self::generate_phase_states(&distribution);
        let current_phases = Array1::zeros(num_elements);

        Self {
            scheme,
            distribution,
            rng,
            phase_states,
            current_phases,
            time_since_switch: 0.0,
        }
    }

    /// Generate phase states based on distribution
    fn generate_phase_states(distribution: &PhaseDistribution) -> Vec<f64> {
        match distribution {
            PhaseDistribution::Binary => vec![0.0, PI],
            PhaseDistribution::Ternary => vec![-2.0 * PI / 3.0, 0.0, 2.0 * PI / 3.0],
            PhaseDistribution::Quadrature => vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0],
            PhaseDistribution::Discrete { states } => (0..*states)
                .map(|i| 2.0 * PI * i as f64 / *states as f64)
                .collect(),
            _ => vec![], // Continuous distributions don't use discrete states
        }
    }

    /// Update phases based on time step
    pub fn update(&mut self, dt: f64) {
        match self.scheme {
            RandomizationScheme::Temporal { period } => {
                self.time_since_switch += dt;
                if self.time_since_switch >= period {
                    self.randomize_phases();
                    self.time_since_switch = 0.0;
                }
            }
            RandomizationScheme::SpatioTemporal { period, .. } => {
                self.time_since_switch += dt;
                if self.time_since_switch >= period {
                    self.randomize_phases();
                    self.time_since_switch = 0.0;
                }
            }
            _ => {} // Other schemes don't depend on time
        }
    }

    /// Randomize phases according to distribution
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    fn randomize_phases(&mut self) {
        let n = self.current_phases.len();

        match self.distribution {
            PhaseDistribution::Uniform => {
                for i in 0..n {
                    self.current_phases[i] = self.rng.gen::<f64>() * MAX_PHASE_SHIFT;
                }
            }
            PhaseDistribution::Gaussian { std_dev } => {
                use rand_distr::{Distribution, Normal};
                let normal = Normal::new(0.0, std_dev).unwrap();
                for i in 0..n {
                    self.current_phases[i] = normal.sample(&mut self.rng);
                }
            }
            PhaseDistribution::Binary
            | PhaseDistribution::Ternary
            | PhaseDistribution::Quadrature
            | PhaseDistribution::Discrete { .. } => {
                let num_states = self.phase_states.len();
                for i in 0..n {
                    let idx = self.rng.gen_range(0..num_states);
                    self.current_phases[i] = self.phase_states[idx];
                }
            }
        }
    }

    /// Apply phase randomization to complex field
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn apply_to_field(&self, field: &mut Array2<f64>) {
        let (n_elements, n_time) = field.dim();
        assert_eq!(n_elements, self.current_phases.len());

        for i in 0..n_elements {
            let phase = self.current_phases[i];
            let (_sin_phase, cos_phase) = phase.sin_cos();

            for t in 0..n_time {
                let real = field[[i, t]];
                // Apply phase rotation (assuming real signal)
                field[[i, t]] = real * cos_phase;
            }
        }
    }

    /// Get current phase values
    #[must_use]
    pub fn phases(&self) -> ArrayView1<'_, f64> {
        self.current_phases.view()
    }

    /// Reset randomizer to initial state
    pub fn reset(&mut self) {
        self.rng = ChaCha8Rng::seed_from_u64(DEFAULT_SEED);
        self.current_phases.fill(0.0);
        self.time_since_switch = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::super::distribution::PhaseDistribution;
    use super::super::scheme::RandomizationScheme;
    use super::*;

    /// Initial phases are all zero before any update.
    #[test]
    fn initial_phases_are_zero() {
        let r = PhaseRandomizer::new(
            RandomizationScheme::Temporal { period: 1e-3 },
            PhaseDistribution::Uniform,
            8,
        );
        assert!(
            r.phases().iter().all(|&p| p == 0.0),
            "all phases must be zero at construction"
        );
    }

    /// After reset, phases return to zero.
    #[test]
    fn reset_clears_phases_to_zero() {
        let mut r = PhaseRandomizer::new(
            RandomizationScheme::Temporal { period: 1e-4 },
            PhaseDistribution::Uniform,
            4,
        );
        // Exceed the period to trigger randomization
        r.update(2e-4);
        r.reset();
        assert!(
            r.phases().iter().all(|&p| p == 0.0),
            "phases must be zero after reset"
        );
    }

    /// Temporal update below period → phases remain zero.
    #[test]
    fn update_below_period_does_not_change_phases() {
        let period = 1e-3;
        let mut r = PhaseRandomizer::new(
            RandomizationScheme::Temporal { period },
            PhaseDistribution::Binary,
            6,
        );
        r.update(period * 0.5); // below threshold
        assert!(
            r.phases().iter().all(|&p| p == 0.0),
            "phases must remain zero when update < period"
        );
    }

    /// Temporal update at or above period triggers randomization.
    ///
    /// For Binary distribution, every phase must be in {0, π}.
    #[test]
    fn temporal_update_at_period_triggers_binary_randomization() {
        let period = 1e-3;
        let mut r = PhaseRandomizer::new(
            RandomizationScheme::Temporal { period },
            PhaseDistribution::Binary,
            16,
        );
        r.update(period); // exactly at period → should trigger
        let phases: Vec<f64> = r.phases().to_vec();
        assert!(
            phases
                .iter()
                .all(|&p| (p - 0.0).abs() < 1e-14 || (p - PI).abs() < 1e-14),
            "Binary distribution must produce phases in {{0, π}}: {phases:?}"
        );
    }

    /// After apply_to_field with zero phases, field is unchanged.
    #[test]
    fn apply_to_field_with_zero_phases_leaves_field_unchanged() {
        use ndarray::arr2;
        let r = PhaseRandomizer::new(
            RandomizationScheme::Temporal { period: 1e-3 },
            PhaseDistribution::Uniform,
            3,
        );
        // phases are all zero → cos(0)=1 → no change
        let mut field = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let original = field.clone();
        r.apply_to_field(&mut field);
        for (a, b) in field.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-14,
                "field must be unchanged with zero phases"
            );
        }
    }
}
