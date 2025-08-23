//! Phase Randomization for Standing Wave Suppression
//!
//! Implements phase randomization techniques to reduce standing waves and
//! improve spatial uniformity of acoustic fields.
//!
//! References:
//! - Tang & Clement (2010): "Standing wave suppression for transcranial ultrasound"
//! - Liu et al. (2018): "Random phase modulation for reduction of peak to average power ratio"
//! - Guo et al. (2020): "Reduced cavitation threshold using phase shift keying"

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

// Phase randomization constants
/// Maximum phase shift for randomization (radians)
const MAX_PHASE_SHIFT: f64 = 2.0 * PI;

/// Default number of phase states for discrete randomization
const DEFAULT_PHASE_STATES: usize = 4;

/// Minimum switching period for temporal randomization (seconds)
const MIN_SWITCHING_PERIOD: f64 = 1e-6;

/// Default correlation length for spatial randomization
const DEFAULT_CORRELATION_LENGTH: f64 = 0.001; // 1mm

/// Phase distribution types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseDistribution {
    Uniform,    // Uniform random distribution
    Discrete,   // Discrete phase states (e.g., 0, π/2, π, 3π/2)
    Gaussian,   // Gaussian distribution
    Binary,     // Binary phase (0 or π)
    Ternary,    // Ternary phase (-2π/3, 0, 2π/3)
    Quadrature, // Quadrature phases (0, π/2, π, 3π/2)
}

/// Randomization schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RandomizationScheme {
    Temporal,       // Time-varying phase randomization
    Spatial,        // Spatially-varying phase randomization
    SpatioTemporal, // Combined spatial and temporal
    Frequency,      // Frequency-dependent randomization
    Amplitude,      // Amplitude-weighted randomization
}

/// Phase randomizer for standing wave suppression
pub struct PhaseRandomizer {
    scheme: RandomizationScheme,
    distribution: PhaseDistribution,
    rng: ChaCha8Rng,
    phase_states: Vec<f64>,
    current_phases: Array1<f64>,
    switching_period: f64,
    time_since_switch: f64,
    correlation_length: f64,
}

impl PhaseRandomizer {
    pub fn new(
        scheme: RandomizationScheme,
        distribution: PhaseDistribution,
        num_elements: usize,
    ) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(42); // Reproducible randomness
        let phase_states = Self::generate_phase_states(&distribution);
        let current_phases = Array1::zeros(num_elements);

        Self {
            scheme,
            distribution,
            rng,
            phase_states,
            current_phases,
            switching_period: 1e-3, // 1 ms default
            time_since_switch: 0.0,
            correlation_length: DEFAULT_CORRELATION_LENGTH,
        }
    }

    /// Generate phase states based on distribution
    fn generate_phase_states(distribution: &PhaseDistribution) -> Vec<f64> {
        match distribution {
            PhaseDistribution::Binary => vec![0.0, PI],
            PhaseDistribution::Ternary => vec![-2.0 * PI / 3.0, 0.0, 2.0 * PI / 3.0],
            PhaseDistribution::Quadrature => vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0],
            PhaseDistribution::Discrete => (0..DEFAULT_PHASE_STATES)
                .map(|i| 2.0 * PI * i as f64 / DEFAULT_PHASE_STATES as f64)
                .collect(),
            _ => vec![], // Continuous distributions don't use discrete states
        }
    }

    /// Set switching period for temporal randomization
    pub fn set_switching_period(&mut self, period: f64) {
        self.switching_period = period.max(MIN_SWITCHING_PERIOD);
    }

    /// Set correlation length for spatial randomization
    pub fn set_correlation_length(&mut self, length: f64) {
        self.correlation_length = length.max(0.0);
    }

    /// Update phase randomization
    pub fn update(&mut self, dt: f64) -> &Array1<f64> {
        match self.scheme {
            RandomizationScheme::Temporal => self.update_temporal(dt),
            RandomizationScheme::Spatial => self.update_spatial(),
            RandomizationScheme::SpatioTemporal => self.update_spatiotemporal(dt),
            RandomizationScheme::Frequency => self.update_frequency(),
            RandomizationScheme::Amplitude => self.update_amplitude(),
        }

        &self.current_phases
    }

    /// Update temporal randomization
    fn update_temporal(&mut self, dt: f64) {
        self.time_since_switch += dt;

        if self.time_since_switch >= self.switching_period {
            self.time_since_switch = 0.0;
            self.randomize_phases();
        }
    }

    /// Update spatial randomization
    fn update_spatial(&mut self) {
        // Spatial randomization is typically static or slowly varying
        // Update only if needed
        if self.current_phases.iter().all(|&p| p == 0.0) {
            self.randomize_phases();
        }
    }

    /// Update spatiotemporal randomization
    fn update_spatiotemporal(&mut self, dt: f64) {
        self.time_since_switch += dt;

        if self.time_since_switch >= self.switching_period {
            self.time_since_switch = 0.0;
            self.randomize_phases_with_correlation();
        }
    }

    /// Update frequency-dependent randomization
    fn update_frequency(&mut self) {
        // Frequency-dependent phase shifts
        for i in 0..self.current_phases.len() {
            let freq_factor = 1.0 + 0.1 * i as f64 / self.current_phases.len() as f64;
            self.current_phases[i] = self.generate_random_phase() * freq_factor;
        }
    }

    /// Update amplitude-weighted randomization
    fn update_amplitude(&mut self) {
        // Amplitude-weighted phase shifts (less randomization for higher amplitudes)
        for i in 0..self.current_phases.len() {
            let weight = 1.0 - (i as f64 / self.current_phases.len() as f64);
            self.current_phases[i] = self.generate_random_phase() * weight;
        }
    }

    /// Randomize all phases
    fn randomize_phases(&mut self) {
        for i in 0..self.current_phases.len() {
            self.current_phases[i] = self.generate_random_phase();
        }
    }

    /// Randomize phases with spatial correlation
    fn randomize_phases_with_correlation(&mut self) {
        let n = self.current_phases.len();
        if n == 0 {
            return;
        }

        // Generate correlated random phases
        let mut base_phase = self.generate_random_phase();
        self.current_phases[0] = base_phase;

        for i in 1..n {
            // Add small random variation based on correlation length
            let variation = self.rng.gen_range(-PI / 4.0..=PI / 4.0) / self.correlation_length;
            base_phase += variation;
            self.current_phases[i] = base_phase % (2.0 * PI);
        }
    }

    /// Generate a random phase based on distribution
    fn generate_random_phase(&mut self) -> f64 {
        match self.distribution {
            PhaseDistribution::Uniform => self.rng.gen_range(0.0..MAX_PHASE_SHIFT),

            PhaseDistribution::Discrete
            | PhaseDistribution::Binary
            | PhaseDistribution::Ternary
            | PhaseDistribution::Quadrature => {
                let idx = self.rng.gen_range(0..self.phase_states.len());
                self.phase_states[idx]
            }

            PhaseDistribution::Gaussian => {
                // Box-Muller transform for Gaussian distribution
                let u1: f64 = self.rng.gen();
                let u2: f64 = self.rng.gen();
                let gaussian = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                (gaussian * PI).rem_euclid(2.0 * PI)
            }
        }
    }

    /// Get current phase array
    pub fn get_phases(&self) -> &Array1<f64> {
        &self.current_phases
    }

    /// Apply phase randomization to a signal
    pub fn apply_to_signal(&self, signal: &ArrayView1<f64>, element_idx: usize) -> Array1<f64> {
        let phase = self.current_phases[element_idx % self.current_phases.len()];
        signal.mapv(|s| s * phase.cos()) // Apply phase shift
    }

    /// Calculate standing wave suppression ratio
    pub fn calculate_suppression_ratio(&self, field: &ArrayView2<f64>) -> f64 {
        // Calculate spatial variance reduction
        let mean = field.mean().unwrap_or(0.0);
        let variance = field.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
        let std_dev = variance.sqrt();

        // Suppression ratio: lower std/mean indicates better uniformity
        if mean.abs() > 1e-10 {
            std_dev / mean.abs()
        } else {
            1.0
        }
    }
}

/// Temporal phase randomization controller
pub struct TemporalRandomization {
    randomizer: PhaseRandomizer,
    update_rate: f64,
    phase_history: Vec<Array1<f64>>,
    max_history: usize,
}

impl TemporalRandomization {
    pub fn new(num_elements: usize, update_rate: f64, distribution: PhaseDistribution) -> Self {
        let randomizer =
            PhaseRandomizer::new(RandomizationScheme::Temporal, distribution, num_elements);

        Self {
            randomizer,
            update_rate,
            phase_history: Vec::new(),
            max_history: 100,
        }
    }

    /// Update temporal randomization
    pub fn update(&mut self, dt: f64) -> &Array1<f64> {
        let phases = self.randomizer.update(dt);

        // Store history
        if self.phase_history.len() >= self.max_history {
            self.phase_history.remove(0);
        }
        self.phase_history.push(phases.clone());

        phases
    }

    /// Get phase coherence over time
    pub fn get_temporal_coherence(&self) -> f64 {
        if self.phase_history.len() < 2 {
            return 1.0;
        }

        let mut coherence = 0.0;
        let n = self.phase_history.len();

        for i in 1..n {
            let prev = &self.phase_history[i - 1];
            let curr = &self.phase_history[i];

            // Calculate phase correlation
            let correlation: f64 = prev
                .iter()
                .zip(curr.iter())
                .map(|(&p1, &p2)| (p2 - p1).cos())
                .sum::<f64>()
                / prev.len() as f64;

            coherence += correlation;
        }

        coherence / (n - 1) as f64
    }
}

/// Spatial phase randomization for multi-element arrays
pub struct SpatialRandomization {
    randomizer: PhaseRandomizer,
    element_positions: Array2<f64>,
    correlation_matrix: Array2<f64>,
}

impl SpatialRandomization {
    pub fn new(element_positions: Array2<f64>, distribution: PhaseDistribution) -> Self {
        let num_elements = element_positions.nrows();
        let randomizer =
            PhaseRandomizer::new(RandomizationScheme::Spatial, distribution, num_elements);

        let correlation_matrix = Self::compute_correlation_matrix(&element_positions);

        Self {
            randomizer,
            element_positions,
            correlation_matrix,
        }
    }

    /// Compute spatial correlation matrix
    fn compute_correlation_matrix(positions: &Array2<f64>) -> Array2<f64> {
        let n = positions.nrows();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let dist = Self::calculate_distance(positions.row(i), positions.row(j));
                // Exponential correlation function
                matrix[(i, j)] = (-dist / DEFAULT_CORRELATION_LENGTH).exp();
            }
        }

        matrix
    }

    /// Calculate Euclidean distance between two positions
    fn calculate_distance(pos1: ArrayView1<f64>, pos2: ArrayView1<f64>) -> f64 {
        pos1.iter()
            .zip(pos2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Generate spatially correlated random phases
    pub fn generate_correlated_phases(&mut self) -> Array1<f64> {
        let n = self.element_positions.nrows();
        let mut phases = Array1::zeros(n);

        // Generate uncorrelated random phases
        for i in 0..n {
            phases[i] = self.randomizer.generate_random_phase();
        }

        // Apply spatial correlation
        let correlated = self.correlation_matrix.dot(&phases);

        // Normalize to [0, 2π]
        correlated.mapv(|p| p.rem_euclid(2.0 * PI))
    }

    /// Update spatial randomization
    pub fn update(&mut self) -> &Array1<f64> {
        let phases = self.generate_correlated_phases();
        self.randomizer.current_phases = phases;
        &self.randomizer.current_phases
    }
}

/// Phase shift keying for cavitation control
pub struct PhaseShiftKeying {
    num_states: usize,
    current_state: usize,
    phase_values: Vec<f64>,
    switching_sequence: Vec<usize>,
    sequence_index: usize,
}

impl PhaseShiftKeying {
    pub fn new(num_states: usize) -> Self {
        assert!(
            (2..=8).contains(&num_states),
            "Number of states must be between 2 and 8"
        );

        let phase_values: Vec<f64> = (0..num_states)
            .map(|i| 2.0 * PI * i as f64 / num_states as f64)
            .collect();

        // Default Gray code sequence for minimal phase transitions
        let switching_sequence = Self::generate_gray_code_sequence(num_states);

        Self {
            num_states,
            current_state: 0,
            phase_values,
            switching_sequence,
            sequence_index: 0,
        }
    }

    /// Generate Gray code sequence for smooth transitions
    fn generate_gray_code_sequence(n: usize) -> Vec<usize> {
        let mut sequence = Vec::with_capacity(n);
        for i in 0..n {
            sequence.push(i ^ (i >> 1)); // Gray code formula
        }
        sequence
    }

    /// Get current phase value
    pub fn get_phase(&self) -> f64 {
        self.phase_values[self.current_state]
    }

    /// Switch to next state in sequence
    pub fn next_state(&mut self) {
        self.sequence_index = (self.sequence_index + 1) % self.switching_sequence.len();
        self.current_state = self.switching_sequence[self.sequence_index];
    }

    /// Set custom switching sequence
    pub fn set_sequence(&mut self, sequence: Vec<usize>) {
        assert!(!sequence.is_empty(), "Sequence cannot be empty");
        assert!(
            sequence.iter().all(|&s| s < self.num_states),
            "Invalid state in sequence"
        );
        self.switching_sequence = sequence;
        self.sequence_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_randomizer() {
        let mut randomizer = PhaseRandomizer::new(
            RandomizationScheme::Temporal,
            PhaseDistribution::Quadrature,
            4,
        );

        randomizer.set_switching_period(0.001);
        let phases = randomizer.update(0.002); // Should trigger switch

        // Check that phases are from the quadrature set
        for &phase in phases.iter() {
            let valid = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0]
                .iter()
                .any(|&p| (phase - p).abs() < 1e-6);
            assert!(valid);
        }
    }

    #[test]
    fn test_temporal_coherence() {
        let mut temporal = TemporalRandomization::new(4, 100.0, PhaseDistribution::Binary);

        // Update multiple times
        for _ in 0..10 {
            temporal.update(0.01);
        }

        let coherence = temporal.get_temporal_coherence();
        assert!(coherence >= -1.0 && coherence <= 1.0);
    }

    #[test]
    fn test_phase_shift_keying() {
        let mut psk = PhaseShiftKeying::new(4);

        let initial_phase = psk.get_phase();
        assert!((initial_phase - 0.0).abs() < 1e-6);

        psk.next_state();
        let next_phase = psk.get_phase();
        assert!(next_phase != initial_phase);
    }
}
