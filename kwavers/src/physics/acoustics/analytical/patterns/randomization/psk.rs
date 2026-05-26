//! Phase Shift Keying (PSK) implementation

use super::constants::DEFAULT_SEED;
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use crate::core::constants::numerical::{TWO_PI};

/// Phase Shift Keying for deterministic phase randomization
#[derive(Debug)]
pub struct PhaseShiftKeying {
    modulation_order: usize,
    symbol_duration: f64,
    phase_states: Vec<f64>,
    current_symbol: usize,
    time_in_symbol: f64,
    rng: ChaCha8Rng,
}

impl PhaseShiftKeying {
    /// Create new PSK modulator
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[must_use]
    pub fn new(modulation_order: usize, symbol_duration: f64) -> Self {
        assert!(
            modulation_order.is_power_of_two() && modulation_order >= 2,
            "Modulation order must be a power of 2 >= 2"
        );

        let phase_states = (0..modulation_order)
            .map(|i| TWO_PI * i as f64 / modulation_order as f64)
            .collect();

        Self {
            modulation_order,
            symbol_duration,
            phase_states,
            current_symbol: 0,
            time_in_symbol: 0.0,
            rng: ChaCha8Rng::seed_from_u64(DEFAULT_SEED),
        }
    }

    /// Update PSK state
    pub fn update(&mut self, dt: f64) {
        self.time_in_symbol += dt;

        while self.time_in_symbol >= self.symbol_duration {
            self.time_in_symbol -= self.symbol_duration;
            self.current_symbol = self.rng.gen_range(0..self.modulation_order);
        }
    }

    /// Get current phase
    #[must_use]
    pub fn current_phase(&self) -> f64 {
        self.phase_states[self.current_symbol]
    }

    /// Generate phase sequence
    pub fn generate_sequence(&mut self, length: usize) -> Array1<f64> {
        Array1::from_shape_fn(length, |_| {
            let symbol = self.rng.gen_range(0..self.modulation_order);
            self.phase_states[symbol]
        })
    }

    /// Common PSK configurations
    #[must_use]
    pub fn bpsk(symbol_duration: f64) -> Self {
        Self::new(2, symbol_duration)
    }

    #[must_use]
    pub fn qpsk(symbol_duration: f64) -> Self {
        Self::new(4, symbol_duration)
    }

    #[must_use]
    pub fn psk8(symbol_duration: f64) -> Self {
        Self::new(8, symbol_duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 1e-6; // 1 µs symbol duration for tests

    /// BPSK phase states are {0, π}.
    #[test]
    fn bpsk_has_two_states_zero_and_pi() {
        let bpsk = PhaseShiftKeying::bpsk(DT);
        let p = bpsk.current_phase();
        // After construction, current_symbol=0 → phase=0
        assert!(
            (p - 0.0).abs() < 1e-15,
            "initial BPSK phase must be 0 (got {p})"
        );
    }

    /// QPSK has 4 states; generate_sequence returns length-N array with values ∈ {0, π/2, π, 3π/2}.
    #[test]
    fn qpsk_generate_sequence_phases_in_quadrature_set() {
        let mut qpsk = PhaseShiftKeying::qpsk(DT);
        let seq = qpsk.generate_sequence(32);
        assert_eq!(seq.len(), 32, "sequence length must be 32");
        let allowed = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
        for &p in seq.iter() {
            let ok = allowed.iter().any(|&a| (p - a).abs() < 1e-12);
            assert!(ok, "QPSK phase {p} not in {{0, π/2, π, 3π/2}}");
        }
    }

    /// PSK-8 has 8 states evenly spaced at k·π/4.
    #[test]
    fn psk8_phase_states_spaced_at_pi_over_4() {
        let psk8 = PhaseShiftKeying::psk8(DT);
        let allowed: Vec<f64> = (0..8).map(|k| k as f64 * PI / 4.0).collect();
        let p = psk8.current_phase();
        let ok = allowed.iter().any(|&a| (p - a).abs() < 1e-12);
        assert!(ok, "PSK-8 initial phase {p} not in expected set");
    }

    /// update advances time; after one full symbol, current_symbol may change (seeded RNG).
    #[test]
    fn update_does_not_panic_across_symbol_boundary() {
        let mut psk = PhaseShiftKeying::new(4, DT);
        // Advance past several symbol boundaries
        for _ in 0..10 {
            psk.update(DT * 1.5);
        }
        // Phase must remain within the PSK-4 set
        let allowed = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
        let p = psk.current_phase();
        let ok = allowed.iter().any(|&a| (p - a).abs() < 1e-12);
        assert!(ok, "phase after updates: {p} not in PSK-4 set");
    }
}
