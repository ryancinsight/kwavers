//! Acoustic diffusivity and absorption for Kuznetsov equation
//!
//! Implements the diffusive term: -(δ/c₀⁴)∂³p/∂t³
//! where δ is the acoustic diffusivity

use ndarray::{Array3, Zip};
use crate::constants::nonlinear::DIFFUSIVITY_WATER;

/// Compute the diffusive term for the Kuznetsov equation
///
/// # Arguments
/// * `pressure` - Current pressure field
/// * `pressure_prev` - Previous time step pressure field
/// * `pressure_prev2` - Two time steps ago pressure field  
/// * `pressure_prev3` - Three time steps ago pressure field
/// * `dt` - Time step size
/// * `sound_speed` - Sound speed c₀
/// * `acoustic_diffusivity` - Diffusivity parameter δ [m²/s]
///
/// # Returns
/// The diffusive term: -(δ/c₀⁴)∂³p/∂t³
pub fn compute_diffusive_term(
    pressure: &Array3<f64>,
    pressure_prev: &Array3<f64>,
    pressure_prev2: &Array3<f64>,
    pressure_prev3: &Array3<f64>,
    dt: f64,
    sound_speed: f64,
    acoustic_diffusivity: f64,
) -> Array3<f64> {
    // Compute coefficient: -δ/c₀⁴
    let coeff = -acoustic_diffusivity / sound_speed.powi(4);
    
    // Compute third time derivative using finite differences
    // ∂³p/∂t³ ≈ (p[n+1] - 3*p[n] + 3*p[n-1] - p[n-2]) / dt³
    let dt_cubed = dt * dt * dt;
    let mut diffusive_term = Array3::zeros(pressure.dim());
    
    Zip::from(&mut diffusive_term)
        .and(pressure)
        .and(pressure_prev)
        .and(pressure_prev2)
        .and(pressure_prev3)
        .for_each(|diff, &p, &p_prev, &p_prev2, &p_prev3| {
            let d3p_dt3 = (p - 3.0 * p_prev + 3.0 * p_prev2 - p_prev3) / dt_cubed;
            *diff = coeff * d3p_dt3;
        });
    
    diffusive_term
}

/// Compute frequency-dependent absorption coefficient
///
/// Uses power-law absorption: α = α₀ * f^n
/// where α₀ is the absorption coefficient at 1 MHz
/// and n is the power (typically 1-2 for biological tissues)
pub fn compute_absorption_coefficient(
    frequency: f64,
    alpha_0: f64,
    power: f64,
) -> f64 {
    alpha_0 * (frequency / 1e6).powf(power)
}

/// Apply thermoviscous absorption using the diffusivity model
///
/// This models absorption through the acoustic diffusivity parameter
/// which captures both thermal conduction and viscous losses
pub fn apply_thermoviscous_absorption(
    pressure: &mut Array3<f64>,
    diffusivity: f64,
    dt: f64,
    sound_speed: f64,
) {
    // Simple exponential decay model for absorption
    // This is a first-order approximation
    let absorption_factor = (-diffusivity * dt / sound_speed.powi(2)).exp();
    
    pressure.mapv_inplace(|p| p * absorption_factor);
}