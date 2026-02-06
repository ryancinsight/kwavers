//! Signal domain primitive
//!
//! Defines the core Signal trait used for time-varying amplitudes in sources and boundary conditions.

use std::fmt::Debug;

/// Core Signal trait representing time-varying amplitude
pub trait Signal: Debug + Send + Sync {
    /// Get the signal amplitude at time t
    fn amplitude(&self, t: f64) -> f64;

    /// Get the signal duration (if finite)
    fn duration(&self) -> Option<f64> {
        None
    }

    /// Get the instantaneous frequency at time t
    fn frequency(&self, t: f64) -> f64;

    /// Get the instantaneous phase at time t
    fn phase(&self, t: f64) -> f64;

    /// Clone the signal into a boxed trait object
    fn clone_box(&self) -> Box<dyn Signal>;
}

impl Clone for Box<dyn Signal> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
