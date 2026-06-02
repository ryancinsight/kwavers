//! Nonlinear term computation for Kuznetsov equation
//!
//! ## Theorem (Kuznetsov nonlinear term derivation)
//!
//! **Statement** (Kuznetsov 1971, eq. 1; Hamilton & Blackstock 1998, §2.3.2):
//! Starting from the compressible Euler equations with a Taylor-expanded equation
//! of state to second order in density perturbation ρ' = ρ − ρ₀:
//!
//! ```text
//! p = c₀²ρ' + (B/A)c₀²(ρ')²/(2ρ₀) + O(ρ'³)
//! ```
//!
//! the second-order wave equation for acoustic pressure includes the nonlinear
//! source term:
//!
//! ```text
//! −(β/ρ₀c₀⁴) ∂²(p²)/∂t²
//! ```
//!
//! where the **coefficient of nonlinearity** is:
//! ```text
//! β = 1 + B/(2A)
//! ```
//!
//! The `B/A` parameter is the ratio of the second- to first-order Taylor
//! coefficients of the equation of state (Beyer 1960). Typical values:
//! water ≈ 5.0, soft tissue ≈ 6.0–7.5.
//!
//! **Discrete approximation of ∂²(p²)/∂t²**: Using the three-point backward
//! finite difference (LeVeque 2007, §2.2). The backward stencil requires no
//! future values, keeping the scheme fully explicit. Taylor expansion shows
//! the truncation error is O(Δt), not O(Δt²) — the centered stencil
//! `(p²[n+1] − 2p²[n] + p²[n-1])/Δt²` achieves O(Δt²) but is implicit:
//!
//! ```text
//! ∂²(p²)/∂t² ≈ (p²[n] − 2p²[n-1] + p²[n-2]) / Δt²   + O(Δt)
//! ```
//!
//! For weakly nonlinear acoustics (Ma ≪ 1) this is acceptable: the nonlinear
//! term is subdominant and the dominant O(Δt²) error comes from the linear
//! leapfrog propagation step.
//!
//! ## References
//!
//! - Kuznetsov VP (1971). Sov. Phys. Acoust. 16(4), 467–470.
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press.
//!   §2.3.2, eq. (2.3.10).
//! - Beyer RT (1960). J. Acoust. Soc. Am. 32(6), 719–721.
//!   DOI: 10.1121/1.1908195
//! - LeVeque RJ (2007). Finite Difference Methods for ODEs and PDEs.
//!   SIAM. §2.2.

use kwavers_core::constants::numerical::{B_OVER_A_DIVISOR, NONLINEARITY_COEFFICIENT_OFFSET};
use ndarray::{Array3, Zip};

/// Compute the nonlinear term for the Kuznetsov equation using workspace.
///
/// ## Theorem — explicit-form nonlinear contribution to ∂²p/∂t²
///
/// Kuznetsov equation operator form (Kuznetsov 1971, eq. 1):
/// ```text
/// ∇²p − (1/c₀²)∂²p/∂t² = −(β/ρ₀c₀⁴)∂²(p²)/∂t² − (δ/c₀⁴)∂³p/∂t³
/// ```
///
/// Rearranging for the leapfrog explicit form `∂²p/∂t²`:
/// ```text
/// ∂²p/∂t² = c₀²∇²p + (β/ρ₀c₀²)∂²(p²)/∂t² + (δ/c₀²)∂³p/∂t³ + S
/// ```
///
/// This function returns the nonlinear contribution `+(β/ρ₀c₀²)∂²(p²)/∂t²`
/// (positive, c² not c⁴).
///
/// Discrete ∂²(p²)/∂t² — backward stencil (LeVeque 2007, §2.2), O(Δt):
/// ```text
/// ∂²(p²)/∂t² ≈ (p²[n] − 2p²[n−1] + p²[n−2]) / Δt²   + O(Δt)
/// ```
///
/// # Arguments
/// * `pressure` - Current pressure field p[n]
/// * `pressure_prev` - Previous pressure field p[n−1]
/// * `pressure_prev2` - Two steps back p[n−2]
/// * `dt` - Time step size Δt
/// * `density` - Ambient density ρ₀
/// * `sound_speed` - Sound speed c₀
/// * `nonlinearity_coefficient` - B/A parameter
/// * `nonlinear_term_out` - Pre-allocated output for `+(β/ρ₀c₀²)∂²(p²)/∂t²`
#[allow(clippy::too_many_arguments)]
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
    // β = 1 + B/(2A)
    let beta = NONLINEARITY_COEFFICIENT_OFFSET + nonlinearity_coefficient / B_OVER_A_DIVISOR;

    // Explicit-form coefficient: +(β/ρ₀c₀²)   [positive; c² not c⁴]
    // Derived from: ∂²p/∂t² = c₀²∇²p + (β/ρ₀c₀²)∂²(p²)/∂t² + …
    let coeff = beta / (density * sound_speed.powi(2));

    Zip::from(nonlinear_term_out)
        .and(pressure)
        .and(pressure_prev)
        .and(pressure_prev2)
        .par_for_each(|nl, &p, &p_prev, &p_prev2| {
            let p2 = p * p;
            let p2_prev = p_prev * p_prev;
            let p2_prev2 = p_prev2 * p_prev2;
            let d2p2_dt2 = (2.0f64.mul_add(-p2_prev, p2) + p2_prev2) / (dt * dt);
            *nl = coeff * d2p2_dt2; // positive; added to leapfrog RHS
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
#[must_use]
pub fn compute_heterogeneous_nonlinearity(b_over_a_field: &Array3<f64>) -> Array3<f64> {
    b_over_a_field.mapv(compute_nonlinearity_coefficient)
}
