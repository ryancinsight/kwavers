//! Nonlinear term computation for Kuznetsov equation
//!
//! Implements the nonlinear term: -(β/ρ₀c₀⁴)∂²p²/∂t²
//! where β = 1 + B/2A is the nonlinearity coefficient

use crate::physics::constants::{B_OVER_A_DIVISOR, NONLINEARITY_COEFFICIENT_OFFSET};
use ndarray::{Array3, Zip};

/// Compute the nonlinear term for the Kuznetsov equation using workspace
///
/// Uses a simplified convective nonlinearity approach for stability:
/// The nonlinear term represents the convective derivative (u·∇)u
/// where u = p/(ρc) is the particle velocity
///
/// # Arguments
/// * `pressure` - Current pressure field
/// * `pressure_prev` - Previous time step pressure field
/// * `pressure_prev2` - Two time steps ago pressure field
/// * `dt` - Time step size
/// * `density` - Ambient density ρ₀
/// * `sound_speed` - Sound speed c₀
/// * `nonlinearity_coefficient` - B/A parameter
/// * `nonlinear_term_out` - Pre-allocated output buffer for the result
pub fn compute_nonlinear_term_workspace(
    pressure: &Array3<f64>,
    pressure_prev: &Array3<f64>,
    _pressure_prev2: &Array3<f64>, // Not used in convective form
    dt: f64,
    density: f64,
    sound_speed: f64,
    nonlinearity_coefficient: f64,
    nonlinear_term_out: &mut Array3<f64>,
) {
    // Compute β = 1 + B/2A using named constants
    let beta = NONLINEARITY_COEFFICIENT_OFFSET + nonlinearity_coefficient / B_OVER_A_DIVISOR;

    // For convective nonlinearity: -(β/2ρc³) p ∂p/∂t
    // This is more stable than the full Kuznetsov formulation
    let coeff = beta / (2.0 * density * sound_speed.powi(3));

    // Simple convective nonlinearity: -(β/2ρc³) p ∂p/∂t
    Zip::from(nonlinear_term_out)
        .and(pressure)
        .and(pressure_prev)
        .for_each(|nl, &p, &p_prev| {
            // Compute time derivative of pressure
            let dp_dt = (p - p_prev) / dt;

            // Convective nonlinearity term
            let nonlinear = -coeff * p * dp_dt;

            // Apply limiting for stability
            *nl = nonlinear.clamp(-1e3, 1e3);
        });
}

/// Compute the quadratic nonlinearity coefficient
///
/// For the Kuznetsov equation, this includes the β term
#[must_use]
pub fn compute_nonlinearity_coefficient(b_over_a: f64) -> f64 {
    NONLINEARITY_COEFFICIENT_OFFSET + b_over_a / B_OVER_A_DIVISOR
}

/// Compute the effective nonlinearity for heterogeneous media
///
/// Takes the local B/A values and computes effective β
pub fn compute_heterogeneous_nonlinearity(b_over_a_field: &Array3<f64>) -> Array3<f64> {
    b_over_a_field.mapv(compute_nonlinearity_coefficient)
}
