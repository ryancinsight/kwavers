//! Nonlinear term computation for Kuznetsov equation
//!
//! Implements the nonlinear term: -(β/ρ₀c₀⁴)∂²p²/∂t²
//! where β = 1 + B/2A is the nonlinearity coefficient

use crate::constants::numerical::SECOND_ORDER_DIFF_COEFF;
use crate::constants::physics::{B_OVER_A_DIVISOR, NONLINEARITY_COEFFICIENT_OFFSET};
use ndarray::{Array3, Zip};

/// Compute the nonlinear term for the Kuznetsov equation using workspace
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
///
/// # Returns
/// The nonlinear term: -(β/ρ₀c₀⁴)∂²p²/∂t² is written to nonlinear_term_out
pub fn compute_nonlinear_term_workspace(
    pressure: &Array3<f64>,
    pressure_prev: &Array3<f64>,
    pressure_prev2: &Array3<f64>,
    dt: f64,
    density: f64,
    sound_speed: f64,
    nonlinearity_coefficient: f64,
    nonlinear_term_out: &mut Array3<f64>,
) {
    // Compute β = 1 + B/2A using named constants
    let beta = NONLINEARITY_COEFFICIENT_OFFSET + nonlinearity_coefficient / B_OVER_A_DIVISOR;

    // For harmonic generation, we need the convective derivative form
    // The Kuznetsov equation nonlinear term: β/(ρ₀c₀⁴) ∂²(p²)/∂t²
    // This generates harmonics through the p² term
    // Note: positive coefficient for physical shock steepening
    let coeff = beta / (density * sound_speed.powi(4));

    // Use centered difference for better accuracy in harmonic generation
    // This preserves the phase relationships needed for second harmonic
    let dt_squared = dt * dt;

    // For initial steps, use forward difference
    if pressure_prev2.iter().all(|&x| x.abs() < 1e-15) {
        // First time step: use forward difference approximation
        let p_squared = pressure * pressure;
        let p_squared_prev = pressure_prev * pressure_prev;

        Zip::from(nonlinear_term_out)
            .and(&p_squared)
            .and(&p_squared_prev)
            .for_each(|nl, &p2, &p2_prev| {
                // Simple first derivative squared as approximation
                let dp_dt = (p2 - p2_prev) / dt;
                *nl = coeff * dp_dt / dt;
            });
    } else {
        // Use centered difference for established propagation
        let p_squared = pressure * pressure;
        let p_squared_prev = pressure_prev * pressure_prev;
        let p_squared_prev2 = pressure_prev2 * pressure_prev2;

        Zip::from(nonlinear_term_out)
            .and(&p_squared)
            .and(&p_squared_prev)
            .and(&p_squared_prev2)
            .for_each(|nl, &p2, &p2_prev, &p2_prev2| {
                // Centered second derivative for better harmonic generation
                let d2p2_dt2 = (p2 - SECOND_ORDER_DIFF_COEFF * p2_prev + p2_prev2) / dt_squared;
                // Apply limiter to prevent numerical instability
                let limited = d2p2_dt2.clamp(-1e20, 1e20);
                *nl = coeff * limited;
            });
    }
}

/// Compute the quadratic nonlinearity coefficient
///
/// For the Kuznetsov equation, this includes the β term
pub fn compute_nonlinearity_coefficient(b_over_a: f64) -> f64 {
    NONLINEARITY_COEFFICIENT_OFFSET + b_over_a / B_OVER_A_DIVISOR
}

/// Compute the effective nonlinearity for heterogeneous media
///
/// Takes the local B/A values and computes effective β
pub fn compute_heterogeneous_nonlinearity(b_over_a_field: &Array3<f64>) -> Array3<f64> {
    b_over_a_field.mapv(compute_nonlinearity_coefficient)
}
