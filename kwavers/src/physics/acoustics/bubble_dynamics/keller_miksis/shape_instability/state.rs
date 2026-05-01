use super::{BREAKUP_FRACTION, N_MODES};

/// State of surface shape modes for a single bubble.
///
/// Tracks amplitude `a_n` and velocity `a_dot_n` for modes n = 2 through
/// `N_MODES + 1`. Index 0 maps to mode n = 2.
#[derive(Debug, Clone)]
pub struct ShapeModeState {
    /// Mode amplitudes a_n [m].
    pub amplitude: [f64; N_MODES],
    /// Mode amplitude rates a_dot_n [m/s].
    pub rate: [f64; N_MODES],
}

impl Default for ShapeModeState {
    fn default() -> Self {
        Self {
            amplitude: [0.0; N_MODES],
            rate: [0.0; N_MODES],
        }
    }
}

impl ShapeModeState {
    /// Seed mode `n` with an initial perturbation amplitude [m].
    pub fn seed(&mut self, n: usize, amplitude_0: f64) {
        if (2..N_MODES + 2).contains(&n) {
            self.amplitude[n - 2] = amplitude_0;
        }
    }

    /// Return the maximum absolute mode amplitude normalized by radius.
    #[must_use]
    pub fn max_normalised_amplitude(&self, r: f64) -> f64 {
        if r < 1e-15 {
            return f64::INFINITY;
        }

        self.amplitude
            .iter()
            .map(|amplitude| amplitude.abs() / r)
            .fold(0.0_f64, f64::max)
    }

    /// Return `true` when any mode crosses the breakup threshold.
    #[must_use]
    pub fn is_unstable(&self, r: f64) -> bool {
        self.max_normalised_amplitude(r) > BREAKUP_FRACTION
    }
}
