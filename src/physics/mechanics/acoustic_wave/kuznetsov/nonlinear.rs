//! Nonlinear term computation for Kuznetsov equation
//!
//! Implements the nonlinear term: -(β/ρ₀c₀⁴)∂²p²/∂t²
//! where β = 1 + B/2A is the nonlinearity coefficient

use ndarray::{Array3, Zip};
use crate::constants::nonlinear::B_OVER_A_WATER;

/// Compute the nonlinear term for the Kuznetsov equation
///
/// # Arguments
/// * `pressure` - Current pressure field
/// * `pressure_prev` - Previous time step pressure field
/// * `pressure_prev2` - Two time steps ago pressure field
/// * `dt` - Time step size
/// * `density` - Ambient density ρ₀
/// * `sound_speed` - Sound speed c₀
/// * `nonlinearity_coefficient` - B/A parameter
///
/// # Returns
/// The nonlinear term: -(β/ρ₀c₀⁴)∂²p²/∂t²
pub fn compute_nonlinear_term(
    pressure: &Array3<f64>,
    pressure_prev: &Array3<f64>,
    pressure_prev2: &Array3<f64>,
    dt: f64,
    density: f64,
    sound_speed: f64,
    nonlinearity_coefficient: f64,
) -> Array3<f64> {
    // Compute β = 1 + B/2A
    let beta = 1.0 + nonlinearity_coefficient / 2.0;
    
    // Compute coefficient: -β/(ρ₀c₀⁴)
    let coeff = -beta / (density * sound_speed.powi(4));
    
    // Compute p²
    let p_squared = pressure * pressure;
    let p_squared_prev = pressure_prev * pressure_prev;
    let p_squared_prev2 = pressure_prev2 * pressure_prev2;
    
    // Compute second time derivative using finite differences
    // ∂²p²/∂t² ≈ (p²[n+1] - 2*p²[n] + p²[n-1]) / dt²
    let dt_squared = dt * dt;
    let mut nonlinear_term = Array3::zeros(pressure.dim());
    
    Zip::from(&mut nonlinear_term)
        .and(&p_squared)
        .and(&p_squared_prev)
        .and(&p_squared_prev2)
        .for_each(|nl, &p2, &p2_prev, &p2_prev2| {
            *nl = coeff * (p2 - 2.0 * p2_prev + p2_prev2) / dt_squared;
        });
    
    nonlinear_term
}

/// Compute the quadratic nonlinearity coefficient
///
/// For the Kuznetsov equation, this includes the β term
pub fn compute_nonlinearity_coefficient(b_over_a: f64) -> f64 {
    1.0 + b_over_a / 2.0
}

/// Compute the effective nonlinearity for heterogeneous media
///
/// Takes the local B/A values and computes effective β
pub fn compute_heterogeneous_nonlinearity(
    b_over_a_field: &Array3<f64>,
) -> Array3<f64> {
    b_over_a_field.mapv(|b_a| compute_nonlinearity_coefficient(b_a))
}