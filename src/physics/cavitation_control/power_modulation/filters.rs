//! Filtering components for power modulation

/// Exponential filter for smoothing amplitude transitions
#[derive(Debug, Clone)]
pub struct ExponentialFilter {
    alpha: f64,
    state: f64,
}

impl ExponentialFilter {
    /// Create new exponential filter with specified time constant
    pub fn new(time_constant: f64) -> Self {
        Self {
            alpha: 1.0 - (-1.0 / time_constant).exp(),
            state: 0.0,
        }
    }

    /// Apply filter to input value
    pub fn filter(&mut self, input: f64) -> f64 {
        self.state = self.alpha * input + (1.0 - self.alpha) * self.state;
        self.state
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Get current filter state
    pub fn get_state(&self) -> f64 {
        self.state
    }

    /// Set filter time constant
    pub fn set_time_constant(&mut self, time_constant: f64) {
        self.alpha = 1.0 - (-1.0 / time_constant).exp();
    }
}