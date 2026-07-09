//! Rytov linearization of the scattered field.
//!
//! The Born and Rytov approximations are the two classical linearizations of the
//! scattering problem. **Born** perturbs the field *additively*,
//! `u = u_inc + u_B`, and is accurate for weak, compact scatterers. **Rytov**
//! perturbs the *complex phase* multiplicatively,
//!
//! ```text
//! u = u_inc · exp(ψ),     ψ ≈ u_B / u_inc        (Rytov phase),
//! ```
//!
//! and is accurate for large, smooth perturbations because it accumulates phase
//! correctly rather than amplitude. Both agree to first order in the scattering
//! strength (`exp(ψ) ≈ 1 + ψ`), but for an accumulated phase delay Rytov
//! preserves `|u| = |u_inc|` whereas Born does not.
//!
//! For inversion, the **Rytov data** extracted from measurements is the complex
//! phase `ψ_obs = ln(u_total / u_inc)`; it is linear in the scattering potential
//! through the same Born sensitivity kernel, so an existing Born forward operator
//! becomes a Rytov operator simply by predicting `ψ` instead of `u_B`.
//!
//! # References
//! - Devaney, A. J. (1981). "Inverse-scattering theory within the Rytov
//!   approximation." *Optics Letters*, 6(8), 374–376.
//! - Kak, A. C., & Slaney, M. (1988). *Principles of Computerized Tomographic
//!   Imaging*, Ch. 6. IEEE Press.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array2;
use kwavers_math::fft::Complex64;

/// Minimum incident-field magnitude below which the Rytov phase (a logarithm /
/// division by `u_inc`) is undefined.
const MIN_INCIDENT: f64 = 1e-12;

/// Born-approximation total field: `u = u_inc + u_B`.
#[must_use]
pub fn born_total_field(
    incident: &Array2<Complex64>,
    born_scattered: &Array2<Complex64>,
) -> Array2<Complex64> {
    incident + born_scattered
}

/// Rytov-approximation total field: `u = u_inc · exp(u_B / u_inc)`.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] on a shape mismatch or where the
/// incident field magnitude falls below [`MIN_INCIDENT`].
pub fn rytov_total_field(
    incident: &Array2<Complex64>,
    born_scattered: &Array2<Complex64>,
) -> KwaversResult<Array2<Complex64>> {
    if incident.dim() != born_scattered.dim() {
        return Err(KwaversError::InvalidInput(
            "incident and scattered fields have mismatched shapes".to_owned(),
        ));
    }
    let mut out = Array2::zeros(incident.dim());
    for ((idx, &u0), &ub) in incident.indexed_iter().zip(born_scattered.iter()) {
        if u0.norm() < MIN_INCIDENT {
            return Err(KwaversError::InvalidInput(
                "incident field magnitude too small for Rytov phase".to_owned(),
            ));
        }
        out[idx] = u0 * (ub / u0).exp();
    }
    Ok(out)
}

/// Rytov phase (complex log) of a measured total field: `ψ = ln(u_total/u_inc)`.
///
/// This is the data the scattering potential maps to linearly under the Rytov
/// approximation; feeding it to a Born forward operator yields Rytov inversion.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] on a shape mismatch, or where the
/// incident or total field magnitude falls below [`MIN_INCIDENT`].
pub fn rytov_phase(
    incident: &Array2<Complex64>,
    total: &Array2<Complex64>,
) -> KwaversResult<Array2<Complex64>> {
    if incident.dim() != total.dim() {
        return Err(KwaversError::InvalidInput(
            "incident and total fields have mismatched shapes".to_owned(),
        ));
    }
    let mut psi = Array2::zeros(incident.dim());
    for ((idx, &u0), &u) in incident.indexed_iter().zip(total.iter()) {
        if u0.norm() < MIN_INCIDENT || u.norm() < MIN_INCIDENT {
            return Err(KwaversError::InvalidInput(
                "field magnitude too small for Rytov phase".to_owned(),
            ));
        }
        psi[idx] = (u / u0).ln();
    }
    Ok(psi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(v: Complex64) -> Array2<Complex64> {
        Array2::from_elem((2, 3), v)
    }

    #[test]
    fn rytov_phase_inverts_rytov_total_field() {
        let inc = field(Complex64::new(1.0, 0.5));
        let born = field(Complex64::new(0.2, -0.1));
        let total = rytov_total_field(&inc, &born).unwrap();
        let psi = rytov_phase(&inc, &total).unwrap();
        // ψ must equal the seed u_B/u_inc.
        let expected = born[[0, 0]] / inc[[0, 0]];
        assert!(
            (psi[[0, 0]] - expected).norm() < 1e-12,
            "psi {}",
            psi[[0, 0]]
        );
    }

    #[test]
    fn born_and_rytov_differ_at_exactly_second_order() {
        let inc = field(Complex64::new(1.0, 0.0));
        let born = field(Complex64::new(1e-3, 5e-4)); // weak; ψ = u_B/u_inc
        let u0 = inc[[0, 0]];
        let psi = born[[0, 0]] / u0;
        let u_born = born_total_field(&inc, &born)[[0, 0]];
        let u_rytov = rytov_total_field(&inc, &born).unwrap()[[0, 0]];
        // exp(ψ) − 1 − ψ = ψ²/2 + O(ψ³): the leading Born/Rytov difference is
        // u_inc·ψ²/2, confirming first-order agreement and a second-order gap.
        let leading = u0 * psi * psi / 2.0;
        let residual = (u_rytov - u_born - leading).norm();
        assert!(
            residual < psi.norm().powi(3),
            "Born/Rytov gap must match ψ²/2 to third order: residual {residual}"
        );
        // And the gap itself is genuinely second-order (≫ third, ≪ first).
        assert!((u_rytov - u_born).norm() > 0.4 * psi.norm().powi(2));
    }

    #[test]
    fn rytov_preserves_amplitude_for_pure_phase_delay() {
        // A pure imaginary Born/u_inc ⇒ ψ imaginary ⇒ |u_total| = |u_inc|.
        // Born would instead change the amplitude.
        let inc = field(Complex64::new(1.0, 0.0));
        let born = field(Complex64::new(0.0, 0.8)); // ψ = i·0.8
        let u_rytov = rytov_total_field(&inc, &born).unwrap()[[0, 0]];
        assert!(
            (u_rytov.norm() - 1.0).abs() < 1e-12,
            "|u_rytov| = {}",
            u_rytov.norm()
        );
        let u_born = born_total_field(&inc, &born)[[0, 0]];
        assert!(
            (u_born.norm() - 1.0).abs() > 0.2,
            "Born must distort amplitude"
        );
    }

    #[test]
    fn rytov_rejects_singular_incident_field() {
        let inc = field(Complex64::new(0.0, 0.0));
        let born = field(Complex64::new(0.1, 0.0));
        assert!(rytov_total_field(&inc, &born).is_err());
    }
}
