//! Misfit functions for Full Waveform Inversion.
//!
//! Literature-validated misfit functions and adjoint sources:
//! - L2/L1 norms (Tarantola 1984)
//! - Envelope misfit with Hilbert transform adjoint (Bozdağ et al. 2011)
//! - Phase misfit with instantaneous phase adjoint (Fichtner et al. 2008)
//! - Cross-correlation and Wasserstein metrics

mod envelope_phase;
mod norm_metrics;
mod pwls;
mod types;
mod wasserstein;

#[cfg(test)]
mod ot_correlation_tests;

pub use pwls::{trace_weights, weighted_l2_objective, weighted_l2_residual, DataWeighting};
pub use types::{MisfitFunction, MisfitType};
