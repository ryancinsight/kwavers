//! Temporal phase randomization

use super::constants::{DEFAULT_SEED, MAX_PHASE_SHIFT, MIN_SWITCHING_PERIOD};
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
    /// # Panics
    /// - Panics if assertion fails: `Switching period must be >= {MIN_SWITCHING_PERIOD} seconds`.
    ///
    #[must_use]
    pub fn new(switching_period: f64) -> Self {
        assert!(
            switching_period >= MIN_SWITCHING_PERIOD,
            "Switching period must be >= {MIN_SWITCHING_PERIOD} seconds"
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
    #[must_use]
    pub fn history(&self) -> &[f64] {
        &self.phase_history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.phase_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::constants::{MAX_PHASE_SHIFT, MIN_SWITCHING_PERIOD};
    use super::*;

    /// update returns false when time has not crossed the period.
    #[test]
    fn update_returns_false_below_period() {
        let period = 1e-3;
        let mut tr = TimeRandomization::new(period);
        let switched = tr.update(period * 0.5);
        assert!(!switched, "must not switch when dt < period");
    }

    /// update returns true when accumulated time reaches the period.
    #[test]
    fn update_returns_true_at_period() {
        let period = 1e-3;
        let mut tr = TimeRandomization::new(period);
        let switched = tr.update(period);
        assert!(switched, "must switch when accumulated time >= period");
    }

    /// generate_phase produces values in [0, MAX_PHASE_SHIFT).
    #[test]
    fn generate_phase_in_valid_range() {
        let mut tr = TimeRandomization::new(MIN_SWITCHING_PERIOD);
        for _ in 0..16 {
            let p = tr.generate_phase();
            assert!(
                p >= 0.0 && p < MAX_PHASE_SHIFT,
                "phase {p} out of [0, MAX_PHASE_SHIFT)"
            );
        }
    }

    /// history grows by one per generate_phase call; clear_history empties it.
    #[test]
    fn history_grows_and_clears() {
        let mut tr = TimeRandomization::new(MIN_SWITCHING_PERIOD);
        tr.generate_phase();
        tr.generate_phase();
        assert_eq!(tr.history().len(), 2, "history must contain 2 entries");
        tr.clear_history();
        assert!(tr.history().is_empty(), "history must be empty after clear");
    }
}
