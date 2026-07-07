mod analytic;
mod first_deriv;
mod helpers;
mod property;
mod second_deriv;

pub(super) type TestBackend = coeus_core::MoiraiBackend;

/// Finite difference step size for first derivatives
pub(super) const FD_H_FIRST: f64 = 1e-5;
/// Finite difference step size for second derivatives
pub(super) const FD_H_SECOND: f64 = 1e-4;
/// Relative tolerance for first derivative comparison
pub(super) const REL_TOL_FIRST: f64 = 1e-3;
/// Relative tolerance for second derivative comparison
pub(super) const REL_TOL_SECOND: f64 = 1e-2;
