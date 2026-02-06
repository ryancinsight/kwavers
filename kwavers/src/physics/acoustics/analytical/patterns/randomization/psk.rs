//! Phase Shift Keying (PSK) implementation

use super::constants::DEFAULT_SEED;
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

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
    #[must_use]
    pub fn new(modulation_order: usize, symbol_duration: f64) -> Self {
        assert!(
            modulation_order.is_power_of_two() && modulation_order >= 2,
            "Modulation order must be a power of 2 >= 2"
        );

        let phase_states = (0..modulation_order)
            .map(|i| 2.0 * PI * i as f64 / modulation_order as f64)
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
