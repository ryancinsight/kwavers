//! Misfit functions for Full Waveform Inversion.
//!
//! Literature-validated misfit functions and adjoint sources:
//! - L2/L1 norms (Tarantola 1984)
//! - Envelope misfit with Hilbert transform adjoint (Bozdağ et al. 2011)
//! - Phase misfit with instantaneous phase adjoint (Fichtner et al. 2008)
//! - Cross-correlation and Wasserstein metrics

mod envelope_phase;
mod norm_metrics;
mod types;
mod wasserstein;

pub use types::{MisfitFunction, MisfitType};
