//! Signal domain primitive
//!
//! Defines the core Signal trait used for time-varying amplitudes in sources and boundary conditions.

use std::fmt::Debug;

/// Core Signal trait representing time-varying amplitude.
///
// dyn: used as `dyn Signal`/`Arc<dyn Signal>` (open, cross-crate-extensible
// implementor set — 22 impls across 6 crates: kwavers-signal, -source,
// -transducer, -solver, -python, facade). Per ADR 012 this is a sanctioned
// dynamic-dispatch boundary; a closed `SignalKind` enum is infeasible without
// circular crate dependencies. Dispatch is O(num_sources)/step (sampled once per
// source per step), never per-cell, so it is not throughput-critical.
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
