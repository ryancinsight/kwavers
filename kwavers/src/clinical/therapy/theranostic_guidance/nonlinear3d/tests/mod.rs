//! Nonlinear 3-D Westervelt + Rayleigh-Plesset test suite.
//!
//! Split from the historical flat `tests.rs` (1391 lines) into SRP-aligned
//! children, one responsibility per file under the 500-line leaf rule:
//!
//! - [`fixtures`] — shared synthetic CT volumes and label maps.
//! - [`bessel`]   — `J_1` / `J_2` analytical helpers for Fubini validation.
//! - [`pipeline`] — end-to-end abdominal + brain focused-bowl integration tests.
//! - [`sign_correction`]   — Westervelt nonlinear-term sign regression.
//! - [`beta_scaling`]      — β = 0 linear baseline + β-scaling regression.
//! - [`harmonic_presence`] — point-source harmonic-generation check.
//! - [`fubini_1d`]         — Aanonsen-1984 1-D Fubini-absolute literature test.
//! - [`absorption`]        — Treeby-Cox 2010 fractional-Laplacian power-law decay.

mod absorption;
mod bessel;
mod beta_scaling;
mod fixtures;
mod fubini_1d;
mod harmonic_presence;
mod pipeline;
mod sign_correction;

pub(super) use super::super::Point3;
