//! Source term definitions

use ndarray::ArrayD;

/// Source term trait
///
/// Defines time-varying source distributions f(x,t)
pub trait SourceTerm: Send + Sync {
    /// Evaluate source at given time [appropriate units]
    fn evaluate(&self, time: f64) -> ArrayD<f64>;

    /// Get temporal support [t_start, t_end]
    fn time_window(&self) -> (f64, f64);

    /// Check if source is active at given time
    fn is_active(&self, time: f64) -> bool {
        let (t_start, t_end) = self.time_window();
        time >= t_start && time <= t_end
    }
}
