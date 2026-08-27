mod analytic;
mod first_deriv;
mod helpers;
mod property;
mod second_deriv;

pub(super) type TestBackend = coeus_core::MoiraiBackend;

// Finite-difference step sizes, derived from the precision the network
// actually evaluates in.
//
// The model computes in `f32`, so eps = 1.19e-7. A central difference carries
// two competing errors: truncation, which falls with `h`, and round-off, which
// rises as eps/h because the subtraction cancels leading digits. Their sum is
// minimised at
//
//     first derivative:   h* = cbrt(eps)      = 4.9e-3, residual eps^(2/3) = 2.4e-5
//     second derivative:  h* = eps^(1/4)      = 1.9e-2, residual eps^(1/2) = 3.5e-4
//
// These were 1e-5 and 1e-4, which are the corresponding optima for `f64`
// (cbrt(2.2e-16) = 6e-6). Against `f32` arithmetic, h = 1e-5 puts round-off at
// eps/h = 1.2e-2 -- a 1% floor on a check whose tolerance is 1e-3. It passed
// only because `Linear::new` set every weight to 1.0, which made the network
// degenerate and its derivatives trivial; against a network with distinct
// units it reports a 1.7% disagreement that is entirely the step size.
//
// The tolerances are unchanged and remain well above the residuals above.

/// Finite difference step size for first derivatives: cbrt(f32::EPSILON).
pub(super) const FD_H_FIRST: f64 = 5e-3;
/// Finite difference step size for second derivatives: f32::EPSILON^(1/4).
pub(super) const FD_H_SECOND: f64 = 2e-2;
/// Relative tolerance for first derivative comparison
pub(super) const REL_TOL_FIRST: f64 = 1e-3;
/// Relative tolerance for second derivative comparison
pub(super) const REL_TOL_SECOND: f64 = 1e-2;
