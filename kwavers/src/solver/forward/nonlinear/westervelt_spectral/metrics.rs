//! Performance metrics tracking for Westervelt solver

use std::time::Duration;

/// Performance metrics for tracking computational costs
#[derive(Debug, Default, Clone)]
pub struct PerformanceMetrics {
    pub call_count: u64,
    pub nonlinear_time: f64,
    pub fft_time: f64,
    pub source_time: f64,
    pub combination_time: f64,
    pub kspace_ops_time: f64,
}

impl PerformanceMetrics {
    /// Create new metrics instance
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a nonlinear computation time
    pub fn record_nonlinear(&mut self, duration: Duration) {
        self.nonlinear_time += duration.as_secs_f64();
    }

    /// Record an FFT operation time
    pub fn record_fft(&mut self, duration: Duration) {
        self.fft_time += duration.as_secs_f64();
    }

    /// Record source injection time
    pub fn record_source(&mut self, duration: Duration) {
        self.source_time += duration.as_secs_f64();
    }

    /// Record field combination time
    pub fn record_combination(&mut self, duration: Duration) {
        self.combination_time += duration.as_secs_f64();
    }

    /// Record k-space operation time
    pub fn record_kspace(&mut self, duration: Duration) {
        self.kspace_ops_time += duration.as_secs_f64();
    }

    /// Increment call counter
    pub fn increment_calls(&mut self) {
        self.call_count += 1;
    }

    /// Get total computation time
    #[must_use]
    pub fn total_time(&self) -> f64 {
        self.nonlinear_time
            + self.fft_time
            + self.source_time
            + self.combination_time
            + self.kspace_ops_time
    }

    /// Print performance summary
    pub fn summary(&self) {
        if self.call_count > 0 {
            log::debug!("Westervelt Performance Summary:");
            log::debug!("  Total calls: {}", self.call_count);
            log::debug!("  Nonlinear time: {:.3}s", self.nonlinear_time);
            log::debug!("  FFT time: {:.3}s", self.fft_time);
            log::debug!("  Source time: {:.3}s", self.source_time);
            log::debug!("  Combination time: {:.3}s", self.combination_time);
            log::debug!("  K-space ops time: {:.3}s", self.kspace_ops_time);
            log::debug!("  Total time: {:.3}s", self.total_time());
        }
    }
}
