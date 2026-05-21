//! Frequency-domain full-waveform inversion.
//!
//! Solver-owned numerical inversion of the heterogeneous-medium Helmholtz
//! equation in slowness. Anatomy-, transducer-, and acquisition-neutral by
//! construction: callers supply source and receiver coordinates plus
//! frequency-domain observations. Clinical adapters (e.g. breast UST under
//! `clinical::imaging::reconstruction::breast_ust_fwi`) wrap this solver with
//! their use-case identifiers. Physics identities and reusable transducer
//! geometries live in
//! `physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi`.
//!
//! # Forward operator selection
//!
//! The forward Helmholtz solver is selected via the
//! [`HelmholtzForwardOperator`] trait carried on [`Config::forward_operator`].
//! Three impls ship today; adding more (BiCGSTAB-preconditioned Helmholtz,
//! sparse-direct, FEM Helmholtz) is `impl HelmholtzForwardOperator` with no
//! inversion-loop changes.
//!
//! 1. [`SingleScatterBornOperator`] (default).
//!    `p ≈ p0 + ∑_x ω² (s(x)² − s0²) u_inc(x) G(x, receiver) dV`,
//!    where `u_inc` is the homogeneous-medium incident field and `G` is the
//!    outgoing free-space Green's function. Validity is bounded by
//!    `‖V‖ · k₀² · L ≲ 1`; for breast/abdominal slowness contrasts of 5–10 %
//!    multiple-scattering corrections are required.
//! 2. [`DenseConvergentBornOperator`] ([`cbs`]). Fixed-point iteration with
//!    shifted potential `V_s = k² − k₀² − iε` and preconditioner
//!    `γ = iε/V_s`, using a finite-volume free-space Green operator for
//!    reduced-grid verification.
//! 3. [`SpectralConvergentBornOperator`]. Same CBS algebra with the
//!    pseudospectral symbol `(k₀² + iε - |k|²)⁻¹` and optional polynomial
//!    absorbing weights `W G W`, giving the FFT-accelerated operator boundary
//!    required for Ali-scale 3-D replication.
//!
//! The gradient is the exact discrete adjoint of the selected forward
//! operator; finite-difference checks against any [`HelmholtzForwardOperator`]
//! impl match the implemented gradient to first order.

pub mod cbs;
mod forward;
mod gradient;
mod inversion;
pub mod operator;
#[cfg(test)]
mod tests;
mod types;

pub use operator::{
    DenseConvergentBornOperator, HelmholtzForwardOperator, SingleScatterBornOperator,
    SpectralConvergentBornOperator,
};

pub use cbs::AbsorbingBoundary;
pub use forward::simulate_frequency_observation;
pub use inversion::invert;
pub use types::{
    Config, FrequencyObservation, InversionResult, FREQUENCY_DOMAIN_FWI_SOLVER_MODEL,
};
