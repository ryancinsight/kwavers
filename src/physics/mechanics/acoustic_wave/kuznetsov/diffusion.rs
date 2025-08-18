//! Acoustic diffusivity and absorption for Kuznetsov equation
//!
//! Implements the diffusive term: -(δ/c₀⁴)∂³p/∂t³
//! where δ is the acoustic diffusivity

use ndarray::{Array3, Zip};
use crate::constants::physics::REFERENCE_FREQUENCY_FOR_ABSORPTION_HZ;
use crate::constants::numerical::THIRD_ORDER_DIFF_COEFF;

/// Compute the diffusive term for the Kuznetsov equation using workspace
///
/// # Arguments
/// * `pressure` - Current pressure field
/// * `pressure_prev` - Previous time step pressure field
/// * `pressure_prev2` - Two time steps ago pressure field  
/// * `pressure_prev3` - Three time steps ago pressure field
/// * `dt` - Time step size
/// * `sound_speed` - Sound speed c₀
/// * `acoustic_diffusivity` - Diffusivity parameter δ [m²/s]
/// * `diffusive_term_out` - Pre-allocated output buffer for the result
///
/// # Returns
/// The diffusive term: -(δ/c₀⁴)∂³p/∂t³ is written to diffusive_term_out
pub fn compute_diffusive_term_workspace(
    pressure: &Array3<f64>,
    pressure_prev: &Array3<f64>,
    pressure_prev2: &Array3<f64>,
    pressure_prev3: &Array3<f64>,
    dt: f64,
    sound_speed: f64,
    acoustic_diffusivity: f64,
    diffusive_term_out: &mut Array3<f64>,
) {
    // Compute coefficient: -δ/c₀⁴
    let coeff = -acoustic_diffusivity / sound_speed.powi(4);
    
    // Compute third time derivative using four-point backward finite difference
    // ∂³p/∂t³ ≈ (p[n] - 3*p[n-1] + 3*p[n-2] - p[n-3]) / dt³
    let dt_cubed = dt.powi(3);
    
    Zip::from(diffusive_term_out)
        .and(pressure)
        .and(pressure_prev)
        .and(pressure_prev2)
        .and(pressure_prev3)
        .for_each(|diff, &p, &p_prev, &p_prev2, &p_prev3| {
            let d3p_dt3 = (p - THIRD_ORDER_DIFF_COEFF * p_prev + THIRD_ORDER_DIFF_COEFF * p_prev2 - p_prev3) / dt_cubed;
            *diff = coeff * d3p_dt3;
        });
}



/// Compute frequency-dependent absorption coefficient
///
/// Uses power-law absorption: α = α₀ * (f/f_ref)^n
/// where α₀ is the absorption coefficient at reference frequency
/// and n is the power (typically 1-2 for biological tissues)
pub fn compute_absorption_coefficient(
    frequency: f64,
    alpha_0: f64,
    power: f64,
) -> f64 {
    alpha_0 * (frequency / REFERENCE_FREQUENCY_FOR_ABSORPTION_HZ).powf(power)
}

// Note: The thermoviscous absorption is properly handled through compute_diffusive_term
// which implements the correct -(δ/c₀⁴)∂³p/∂t³ formulation from the Kuznetsov equation.