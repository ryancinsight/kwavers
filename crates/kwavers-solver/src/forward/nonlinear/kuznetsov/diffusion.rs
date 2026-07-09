//! Acoustic diffusivity and absorption for Kuznetsov equation
//!
//! ## Theorem (Stokes-Kirchhoff thermoviscous absorption term)
//!
//! **Statement** (Lighthill 1978, §3.4): The diffusive term in the Kuznetsov
//! equation arises from viscous and thermal losses in the fluid:
//!
//! ```text
//! −(δ/c₀⁴) ∂³p/∂t³
//! ```
//!
//! where the acoustic diffusivity is:
//! ```text
//! δ = (4μ/3 + μ_B)/ρ₀ + κ(1/cᵥ − 1/cₚ)/ρ₀   [m²/s]
//! ```
//! with `μ` = shear viscosity, `μ_B` = bulk viscosity, `κ` = thermal
//! conductivity, `cᵥ`, `cₚ` = specific heats at constant volume/pressure.
//!
//! **Frequency-domain interpretation**: In the frequency domain the diffusive
//! term produces a power-law absorption coefficient:
//! ```text
//! α(ω) = δω²/(2c₀³)   [Np/m]
//! ```
//! consistent with the classical Stokes-Kirchhoff formula. At low megahertz
//! frequencies in water, this gives α ∝ f² in agreement with measurements.
//!
//! **Third-order finite-difference approximation**: The term `∂³p/∂t³` is
//! approximated by the four-point backward difference (LeVeque 2007, §2.14):
//! ```text
//! ∂³p/∂t³ ≈ (p[n] − 3p[n-1] + 3p[n-2] − p[n-3]) / Δt³   + O(Δt)
//! ```
//! This is the forward-most stable 4-point stencil; the truncation error is
//! O(Δt), sufficient when Δt ≪ T_period (resolved time scale).
//!
//! ## References
//!
//! - Lighthill MJ (1978). Waves in Fluids. Cambridge UP. §3.4.
//! - Kuznetsov VP (1971). Sov. Phys. Acoust. 16(4), 467–470.
//! - LeVeque RJ (2007). Finite Difference Methods for ODEs and PDEs.
//!   SIAM. §2.14.

use kwavers_core::constants::acoustic_parameters::REFERENCE_FREQUENCY_FOR_ABSORPTION_HZ;
use kwavers_core::constants::numerical::THIRD_ORDER_DIFF_COEFF;
use moirai_parallel::ParallelSliceMut;
use leto::Array3;

/// Compute the diffusive term for the Kuznetsov equation using workspace.
///
/// ## Theorem — explicit-form diffusive contribution to ∂²p/∂t²
///
/// Kuznetsov operator form: `∇²p − (1/c₀²)∂²p/∂t² = … − (δ/c₀⁴)∂³p/∂t³`
///
/// Rearranging for the leapfrog explicit form:
/// ```text
/// ∂²p/∂t² = c₀²∇²p + … + (δ/c₀²)∂³p/∂t³
/// ```
///
/// This function returns the diffusive contribution `+(δ/c₀²)∂³p/∂t³`
/// (positive, c² not c⁴).
///
/// # Arguments
/// * `pressure` - Current pressure p[n]
/// * `pressure_prev` - p[n−1]
/// * `pressure_prev2` - p[n−2]
/// * `pressure_prev3` - p[n−3]
/// * `dt` - Time step Δt
/// * `sound_speed` - Sound speed c₀
/// * `acoustic_diffusivity` - Diffusivity δ [m²/s]
/// * `diffusive_term_out` - Pre-allocated output for `+(δ/c₀²)∂³p/∂t³`
#[allow(clippy::too_many_arguments)]
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
    // Explicit-form coefficient: +(δ/c₀²)   [positive; c² not c⁴]
    // Derived from: ∂²p/∂t² = c₀²∇²p + … + (δ/c₀²)∂³p/∂t³
    let coeff = acoustic_diffusivity / sound_speed.powi(2);

    // Compute third time derivative using four-point backward finite difference
    // ∂³p/∂t³ ≈ (p[n] - 3*p[n-1] + 3*p[n-2] - p[n-3]) / dt³
    let dt_cubed = dt.powi(3);

    // Standard-layout asserts: original Zip iteration is layout-agnostic; the
    // migrated par_mut().enumerate() requires C-contiguous storage so the
    // flat-slice index space matches Zip's C-order iteration. Failing here
    // produces a discoverable error before any silent OOB reads.
    assert!(
        diffusive_term_out,
        "diffusive_term_out must be C-contiguous (default Array3 layout) for the migration"
    );
    assert!(
        pressure,
        "pressure must be C-contiguous (default Array3 layout) for the migration"
    );
    assert!(
        pressure_prev,
        "pressure_prev must be C-contiguous (default Array3 layout) for the migration"
    );
    assert!(
        pressure_prev2,
        "pressure_prev2 must be C-contiguous (default Array3 layout) for the migration"
    );
    assert!(
        pressure_prev3,
        "pressure_prev3 must be C-contiguous (default Array3 layout) for the migration"
    );
    {
        let diff_slice = diffusive_term_out
            .as_slice_mut()
            .expect("diffusive_term_out: standard-layout asserted just above; layout matched");
        let p_slice = pressure
            .as_slice()
            .expect("pressure: standard-layout asserted just above; layout matched");
        let prev_slice = pressure_prev
            .as_slice()
            .expect("pressure_prev: standard-layout asserted just above; layout matched");
        let prev2_slice = pressure_prev2
            .as_slice()
            .expect("pressure_prev2: standard-layout asserted just above; layout matched");
        let prev3_slice = pressure_prev3
            .as_slice()
            .expect("pressure_prev3: standard-layout asserted just above; layout matched");
        diff_slice.iter_mut().enumerate(|idx, diff: &mut f64| {
            let p_val = p_slice[idx];
            let prev_val = prev_slice[idx];
            let prev2_val = prev2_slice[idx];
            let prev3_val = prev3_slice[idx];
            // ∂³p/∂t³ ≈ (p[n] − 3p[n−1] + 3p[n−2] − p[n−3]) / Δt³  + O(Δt)
            // (four-point backward, exact for cubic polynomials)
            let d3p_dt3 = (THIRD_ORDER_DIFF_COEFF
                .mul_add(prev2_val, THIRD_ORDER_DIFF_COEFF.mul_add(-prev_val, p_val))
                - prev3_val)
                / dt_cubed;
            // coeff = +(δ/c₀²); positive contribution to explicit ∂²p/∂t²
            *diff = coeff * d3p_dt3;
        });
    }
}

/// Compute frequency-dependent absorption coefficient
///
/// Uses power-law absorption: α = α₀ * (`f/f_ref)^n`
/// where α₀ is the absorption coefficient at reference frequency
/// and n is the power (typically 1-2 for biological tissues)
#[must_use]
pub fn compute_absorption_coefficient(frequency: f64, alpha_0: f64, power: f64) -> f64 {
    alpha_0 * (frequency / REFERENCE_FREQUENCY_FOR_ABSORPTION_HZ).powf(power)
}

// Note: The thermoviscous absorption is properly handled through compute_diffusive_term
// which implements the correct -(δ/c₀⁴)∂³p/∂t³ formulation from the Kuznetsov equation.
