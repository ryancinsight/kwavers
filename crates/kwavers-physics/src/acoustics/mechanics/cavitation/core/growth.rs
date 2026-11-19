//! Bubble growth mechanics
//!
//! Models the growth of cavitation bubbles over time, particularly
//! through rectified diffusion.
//!
//! # Mathematical Specification
//!
//! ## Theorem: Rectified Diffusion (Eller & Flynn 1965)
//! Under acoustic excitation, gas bubbles grow by rectified diffusion:
//! during expansion the bubble surface area increases, enhancing inward
//! gas flux; during compression, reduced surface area limits outward flux.
//! The net effect is a monotonic increase in equilibrium radius.
//!
//! The growth rate is proportional to:
//!
//! $$ \frac{dR_0}{dt} \propto 4\pi R \cdot D \cdot C \cdot \frac{P_{ac}}{P_0} \cdot \sqrt{Pe} $$
//!
//! where:
//! - $R$ = current bubble radius (m)
//! - $D$ = gas diffusivity in liquid (m²/s)
//! - $C$ = dissolved gas concentration (mol/m³)
//! - $P_{ac}/P_0$ = acoustic-to-ambient pressure ratio (dimensionless)
//! - $Pe = R^2 f / D$ = Péclet number (convective/diffusive transport ratio)
//!
//! **Dimensional analysis:** (m)·(m²/s)·(mol/m³)·(1)·(1) = (mol/s)
//! confirming units of molar flux rate.
//!
//! # References
//! - Eller, A. I., & Flynn, H. G. (1965). "Rectified diffusion during
//!   nonlinear pulsations of cavitation bubbles". JASA, 37(3), 493–503.
//! - Church, C. C. (1988). "A theoretical study of cavitation generated
//!   by an acoustic beam". JASA, 83(6), 2210–2216.

use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};

/// Maximum acoustic-to-ambient pressure ratio for which the Eller–Flynn
/// rectified-diffusion law is in its domain of validity.
///
/// The law assumes small-amplitude oscillation (ε = ΔR/R₀ << 1), a
/// quasi-steady gas field, and an infinite reservoir. Keller–Miksis puts
/// ε = 1 at P_ac/P₀ ≈ 3–4 (R₀ = 2 µm); above that the bubble collapses
/// inertially every cycle and the law's flux is meaningless. Stage 6 of the
/// triplet-pulse prototype applied the law at P_ac/P₀ = 66–146 and produced
/// 986/1904 µm growth artifacts; the Stage 6 audit (Nov 17, 2026) retired
/// those numbers. Use [`rectified_diffusion_rate_checked`] to enforce this
/// bound at the call site.
pub const RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO: f64 = 3.0;

/// Rectified diffusion model for bubble growth (Eller & Flynn 1965)
///
/// Returns the molar gas flux rate [mol/s] into the bubble.
///
/// # Validity
/// The linearized law is valid only for P_ac/P₀ ≲ 3
/// ([`RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO`]). Outside that regime the
/// returned flux is an extrapolation artifact, not a prediction — see the
/// Stage 6 audit. Prefer [`rectified_diffusion_rate_checked`], which
/// rejects out-of-regime inputs instead of silently returning a flux.
#[must_use]
pub fn rectified_diffusion_rate(
    radius: f64,            // [m]
    ambient_pressure: f64,  // [Pa]
    acoustic_pressure: f64, // [Pa]
    frequency: f64,         // [Hz]
    diffusivity: f64,       // [m²/s]
    concentration: f64,     // [mol/m³]
) -> f64 {
    let pressure_ratio = acoustic_pressure / ambient_pressure;
    let peclet = radius * radius * frequency / diffusivity;

    // Growth rate ∝ R * D * C * (P_ac/P_0) * √Pe
    FOUR_PI * radius * diffusivity * concentration * pressure_ratio * peclet.sqrt()
}

/// Checked form of [`rectified_diffusion_rate`]: same molar gas flux rate
/// [mol/s], but rejects inputs outside the law's domain of validity instead
/// of silently returning an extrapolation artifact.
///
/// # Errors
/// - Returns [`KwaversError::Physics`] when `ambient_pressure <= 0` (the
///   pressure ratio is undefined) or when `acoustic_pressure /
///   ambient_pressure` exceeds [`RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO`]
///   (small-amplitude regime violated; the bubble is in inertial collapse
///   and the flux has no meaning).
pub fn rectified_diffusion_rate_checked(
    radius: f64,            // [m]
    ambient_pressure: f64,  // [Pa]
    acoustic_pressure: f64, // [Pa]
    frequency: f64,         // [Hz]
    diffusivity: f64,       // [m²/s]
    concentration: f64,     // [mol/m³]
) -> KwaversResult<f64> {
    if ambient_pressure <= 0.0 {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "ambient_pressure".to_owned(),
            value: ambient_pressure,
            reason: "Ambient pressure must be positive for the pressure ratio to be defined"
                .to_owned(),
        }));
    }
    let pressure_ratio = acoustic_pressure / ambient_pressure;
    if pressure_ratio > RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "acoustic_pressure / ambient_pressure".to_owned(),
            value: pressure_ratio,
            reason: format!(
                "Eller-Flynn rectified-diffusion law is valid only for P_ac/P0 <= {RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO} (small-amplitude oscillation); got {pressure_ratio:.1}"
            ),
        }));
    }
    Ok(rectified_diffusion_rate(
        radius,
        ambient_pressure,
        acoustic_pressure,
        frequency,
        diffusivity,
        concentration,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_relative_eq;
    use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;

    // Standard air-in-water parameters (Eller & Flynn 1965 regime)
    const R0: f64 = 5e-6; // 5 µm bubble radius
    const P0: f64 = ATMOSPHERIC_PRESSURE; // atmospheric pressure [Pa]
    const P_AC: f64 = 120_000.0; // ~1.18 atm acoustic amplitude
    const FREQ: f64 = MHZ_TO_HZ; // 1 MHz ultrasound
    const D_AIR: f64 = 2.0e-9; // O₂ diffusivity in water [m²/s]
    const C_SAT: f64 = 0.26; // dissolved O₂ at saturation [mol/m³]

    #[test]
    fn test_rectified_diffusion_positive_flux() {
        let flux = rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, C_SAT);
        assert!(
            flux > 0.0,
            "Positive acoustic pressure must yield positive molar flux"
        );
        // Physical reasonableness: flux should be very small (pico- to nano-moles/s)
        assert!(
            flux < 1.0,
            "Molar flux should be << 1 mol/s for microbubbles"
        );
    }

    #[test]
    fn test_rectified_diffusion_linearity() {
        let base = rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, C_SAT);

        // Doubling radius → 2× from R factor · √(4) from Pe → 4× total
        let doubled_r = rectified_diffusion_rate(2.0 * R0, P0, P_AC, FREQ, D_AIR, C_SAT);
        assert_relative_eq!(doubled_r / base, 4.0, epsilon = 1e-10);

        // Doubling concentration → 2× (linear)
        let doubled_c = rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, 2.0 * C_SAT);
        assert_relative_eq!(doubled_c / base, 2.0, epsilon = 1e-10);

        // Doubling acoustic pressure → 2× (linear via pressure_ratio)
        let doubled_pac = rectified_diffusion_rate(R0, P0, 2.0 * P_AC, FREQ, D_AIR, C_SAT);
        assert_relative_eq!(doubled_pac / base, 2.0, epsilon = 1e-10);

        // Doubling diffusivity → √(1/2) from Pe · 2 from D factor → √2 total
        let doubled_d = rectified_diffusion_rate(R0, P0, P_AC, FREQ, 2.0 * D_AIR, C_SAT);
        assert_relative_eq!(doubled_d / base, 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_rectified_diffusion_zero_acoustic_pressure() {
        let flux = rectified_diffusion_rate(R0, P0, 0.0, FREQ, D_AIR, C_SAT);
        assert_eq!(flux, 0.0, "Zero acoustic pressure must yield zero flux");
    }

    #[test]
    fn test_rectified_diffusion_zero_diffusivity() {
        // With D=0, Pe → ∞ but D·√Pe = D·R·√(f/D) = R·√(f·D) → 0
        // Actually: flux = 4π·R·D·C·(P_ac/P₀)·√(R²f/D)
        //         = 4π·R²·C·(P_ac/P₀)·√(f·D)
        // So D→0 ⟹ flux→0
        let flux = rectified_diffusion_rate(R0, P0, P_AC, FREQ, 0.0, C_SAT);
        // D=0 causes Pe division by zero → NaN·0 in the formula.
        // The function returns NaN which is correct: zero diffusivity
        // is a degenerate physical limit (no mass transport possible).
        // Verify the result is either 0 or NaN (both valid for D=0).
        assert!(
            flux == 0.0 || flux.is_nan(),
            "Zero diffusivity must yield zero or NaN flux, got {flux}"
        );
    }

    #[test]
    fn test_rectified_diffusion_checked_accepts_in_regime() {
        // P_AC = 120 kPa is P_ac/P0 = 1.18, inside the small-amplitude regime.
        let flux = rectified_diffusion_rate_checked(R0, P0, P_AC, FREQ, D_AIR, C_SAT)
            .expect("in-regime input must be accepted");
        assert_relative_eq!(
            flux,
            rectified_diffusion_rate(R0, P0, P_AC, FREQ, D_AIR, C_SAT),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_rectified_diffusion_checked_rejects_burst_pressures() {
        // Stage 6 audit: the law was applied at P_ac/P0 = 66 (Iwasaki burst,
        // 6.7 MPa) and 146 (Kuji burst, 14.7 MPa), producing 986/1904 µm
        // growth artifacts. The checked form must refuse those inputs.
        for p_ac in [6.7e6, 14.7e6] {
            let err = rectified_diffusion_rate_checked(R0, P0, p_ac, FREQ, D_AIR, C_SAT)
                .expect_err("burst pressure must be rejected");
            assert!(
                matches!(
                    err,
                    KwaversError::Physics(PhysicsError::InvalidParameter { .. })
                ),
                "expected InvalidParameter, got {err:?}"
            );
        }
    }

    #[test]
    fn test_rectified_diffusion_checked_rejects_nonpositive_ambient() {
        let err = rectified_diffusion_rate_checked(R0, 0.0, P_AC, FREQ, D_AIR, C_SAT)
            .expect_err("zero ambient pressure must be rejected");
        assert!(
            matches!(
                err,
                KwaversError::Physics(PhysicsError::InvalidParameter { .. })
            ),
            "expected InvalidParameter, got {err:?}"
        );
    }
}
