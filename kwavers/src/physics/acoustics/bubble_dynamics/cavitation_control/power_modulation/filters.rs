//! Filtering components for power modulation

/// Exponential filter for smoothing amplitude transitions
#[derive(Debug, Clone)]
pub struct ExponentialFilter {
    alpha: f64,
    state: f64,
}

impl ExponentialFilter {
    /// Create new exponential filter with specified time constant
    #[must_use]
    pub fn new(time_constant: f64) -> Self {
        Self {
            alpha: 1.0 - (-1.0 / time_constant).exp(),
            state: 0.0,
        }
    }

    /// Apply filter to input value
    pub fn filter(&mut self, input: f64) -> f64 {
        self.state = self.alpha.mul_add(input, (1.0 - self.alpha) * self.state);
        self.state
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Get current filter state
    #[must_use]
    pub fn get_state(&self) -> f64 {
        self.state
    }

    /// Set filter time constant
    pub fn set_time_constant(&mut self, time_constant: f64) {
        self.alpha = 1.0 - (-1.0 / time_constant).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// filter(x) on a fresh filter equals alpha*x where alpha = 1 - exp(-1/tc).
    ///
    /// For tc=1.0: alpha = 1 - exp(-1). After one step from zero: state = alpha * 1.0.
    #[test]
    fn exponential_filter_first_step_matches_alpha() {
        let tc = 1.0_f64;
        let mut f = ExponentialFilter::new(tc);
        let expected_alpha = 1.0 - (-1.0_f64 / tc).exp();
        let out = f.filter(1.0);
        assert!(
            (out - expected_alpha).abs() < 1e-14,
            "first filter step must equal alpha={expected_alpha:.6}, got {out:.6}"
        );
    }

    /// reset zeroes the filter state.
    #[test]
    fn exponential_filter_reset_clears_state() {
        let mut f = ExponentialFilter::new(1.0);
        f.filter(1.0); // state is now non-zero
        f.reset();
        assert!(
            f.get_state().abs() < 1e-30,
            "state must be zero after reset"
        );
    }

    /// filter converges toward a constant input from zero initial state.
    ///
    /// After many steps with input=1.0, output must approach 1.0.
    #[test]
    fn exponential_filter_converges_to_constant_input() {
        let mut f = ExponentialFilter::new(0.01); // large alpha ≈ 1 - exp(-100) ≈ 1
                                                  // After 100 steps from zero, must be within 1e-6 of 1.0
        for _ in 0..200 {
            f.filter(1.0);
        }
        assert!(
            (f.get_state() - 1.0).abs() < 1e-6,
            "filter must converge to 1.0; got {}",
            f.get_state()
        );
    }
}
