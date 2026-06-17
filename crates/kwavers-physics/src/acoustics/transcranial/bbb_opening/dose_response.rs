//! Analytical BBB dose–response and closure-kinetics models (book Ch23 §23.4,
//! §23.6).
//!
//! Closed-form pharmacodynamic relations for LIFU-mediated blood–brain-barrier
//! opening: the cumulative acoustic dose, the Hill permeability dose-response,
//! and the bi-exponential post-sonication closure. These are the literature
//! dose/recovery curves (McDannold 2008; Deffieux & Konofagou 2010). They are
//! distinct from — and complementary to — the per-voxel real-time enhancement
//! model used by the field `BBBOpening` simulator: this module answers
//! "how much opening for a given cumulative dose, and how does it recover with
//! time", not "what is the local enhancement during one sonication".

/// Cumulative BBB acoustic dose `D = MI²·t_on·PRF` [MI²·s] (Definition 23.1):
/// proportional to the time-averaged acoustic energy delivered at the focus per
/// unit skull-window area.
#[must_use]
#[inline]
pub fn bbb_acoustic_dose(mechanical_index: f64, on_time_s: f64, prf_hz: f64) -> f64 {
    mechanical_index * mechanical_index * on_time_s * prf_hz
}

/// Hill dose-response of BBB permeability enhancement (§23.4):
///
/// ```text
/// P(D) = P_max · D^n / (D_50^n + D^n).
/// ```
///
/// At `D = D_50` the response is half-maximal; it is monotone increasing in `D`
/// and bounded above by `P_max`. For stable cavitation, `D_50 ≈ 1.2 MI²·s` and
/// `n ≈ 2.5` (fit to Evans-blue extravasation, McDannold 2008). Returns `0` for
/// non-positive dose or `D_50`.
#[must_use]
pub fn bbb_permeability_hill(dose: f64, p_max: f64, d50: f64, hill_n: f64) -> f64 {
    if dose <= 0.0 || d50 <= 0.0 {
        return 0.0;
    }
    let dn = dose.powf(hill_n);
    p_max * dn / (d50.powf(hill_n) + dn)
}

/// Bi-exponential post-sonication BBB closure (§23.6):
///
/// ```text
/// P(t) = P_peak·[0.6·e^(−t/τ_fast) + 0.4·e^(−t/τ_slow)].
/// ```
///
/// The fast component (`τ_fast ≈ 0.5 h`) is tight-junction re-assembly; the slow
/// component (`τ_slow ≈ 6 h`) is vesicular-transport clearance
/// (Deffieux & Konofagou 2010). `t`, `τ_fast`, and `τ_slow` share one time unit.
/// Returns `0` for non-positive time constants.
#[must_use]
pub fn bbb_closure_permeability(t: f64, p_peak: f64, tau_fast: f64, tau_slow: f64) -> f64 {
    if tau_fast <= 0.0 || tau_slow <= 0.0 {
        return 0.0;
    }
    p_peak * (0.6 * (-t / tau_fast).exp() + 0.4 * (-t / tau_slow).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acoustic_dose_scales_with_mi_squared_and_exposure() {
        // D = MI²·t_on·PRF: MI=0.4, 0.1 s on, 1 Hz ⇒ 0.016 MI²·s.
        assert!((bbb_acoustic_dose(0.4, 0.1, 1.0) - 0.016).abs() < 1e-12);
        // Doubling MI quadruples the dose.
        assert!((bbb_acoustic_dose(0.8, 0.1, 1.0) / bbb_acoustic_dose(0.4, 0.1, 1.0) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn hill_dose_response_is_half_max_at_d50_and_saturates() {
        let (p_max, d50, n) = (1.0, 1.2, 2.5);
        assert!((bbb_permeability_hill(d50, p_max, d50, n) - 0.5 * p_max).abs() < 1e-12);
        assert_eq!(bbb_permeability_hill(0.0, p_max, d50, n), 0.0);
        // Saturating and monotone increasing.
        assert!(bbb_permeability_hill(100.0 * d50, p_max, d50, n) > 0.99 * p_max);
        assert!(bbb_permeability_hill(2.0, p_max, d50, n) > bbb_permeability_hill(1.0, p_max, d50, n));
    }

    #[test]
    fn closure_starts_at_peak_and_recovers_through_the_50pct_crossing() {
        let (p_peak, tf, ts) = (1.0, 0.5, 6.0); // hours
        // t = 0 ⇒ full peak (0.6 + 0.4 = 1).
        assert!((bbb_closure_permeability(0.0, p_peak, tf, ts) - p_peak).abs() < 1e-12);
        // The fast-dominated bi-exponential crosses 50% of peak near t ≈ 0.7 h:
        // above 50% at 0.5 h, below at 1 h.
        assert!(bbb_closure_permeability(0.5, p_peak, tf, ts) > 0.5 * p_peak);
        assert!(bbb_closure_permeability(1.0, p_peak, tf, ts) < 0.5 * p_peak);
        // Monotone decreasing.
        assert!(bbb_closure_permeability(2.0, p_peak, tf, ts) > bbb_closure_permeability(8.0, p_peak, tf, ts));
    }
}
