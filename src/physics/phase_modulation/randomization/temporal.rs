//! Temporal phase randomization

use super::constants::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Time-based phase randomization
#[derive(Debug)]
pub struct TimeRandomization {
    switching_period: f64,
    time_since_switch: f64,
    rng: ChaCha8Rng,
    phase_history: Vec<f64>,
}

impl TimeRandomization {
    /// Create new temporal randomizer
    pub fn new(switching_period: f64) -> Self {
        assert!(
            switching_period >= MIN_SWITCHING_PERIOD,
            "Switching period must be >= {} seconds",
            MIN_SWITCHING_PERIOD
        );

        Self {
            switching_period,
            time_since_switch: 0.0,
            rng: ChaCha8Rng::seed_from_u64(DEFAULT_SEED),
            phase_history: Vec::new(),
        }
    }

    /// Update time and check if switching needed
    pub fn update(&mut self, dt: f64) -> bool {
        self.time_since_switch += dt;

        if self.time_since_switch >= self.switching_period {
            self.time_since_switch = 0.0;
            true
        } else {
            false
        }
    }

    /// Generate random phase
    pub fn generate_phase(&mut self) -> f64 {
        let phase = self.rng.gen::<f64>() * MAX_PHASE_SHIFT;
        self.phase_history.push(phase);
        phase
    }

    /// Get phase history
    pub fn history(&self) -> &[f64] {
        &self.phase_history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.phase_history.clear();
    }
}
