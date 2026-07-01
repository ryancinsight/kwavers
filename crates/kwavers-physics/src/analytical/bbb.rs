//! Blood-brain barrier (BBB) permeability and closure kinetic models for ch24.
//!
//! Covers:
//! * Hill-function dose-response model for BBB permeability enhancement
//!   (McDannold 2008, Evans-blue extravasation fit).
//! * Logistic inertial-cavitation damage probability versus acoustic dose.
//! * Bi-exponential closure kinetics post-sonication
//!   (Deffieux & Konofagou 2010).
//! * CEUS backscatter signal vs microbubble concentration
//!   (de Jong 1991 single-scatter + attenuation model).
//!
//! All functions are pure analytical models: no wave simulation, no ODE
//! integration.  Used by Chapter 24 (LIFU-mediated BBB opening).

// ─── Hill dose-response model ─────────────────────────────────────────────────

/// BBB permeability enhancement as a function of acoustic dose (Hill model).
///
/// ```text
/// P(D) = D^n / (D₅₀^n + D^n)
/// ```
///
/// D is the cumulative acoustic dose (e.g., MI² × t_on [s]), D₅₀ is the dose
/// at half-maximum opening, and n is the Hill coefficient controlling steepness.
/// Output is normalised to [0, 1] where 1.0 represents maximum permeability
/// enhancement (~10× baseline Evans-blue extravasation).
///
/// # Arguments
/// * `dose` – cumulative acoustic dose [arbitrary units, e.g. MI²·s]
/// * `d50`  – dose at half-maximum permeability
/// * `hill_n` – Hill coefficient (dimensionless; n > 0)
///
/// # Reference
/// McDannold et al. (2008) *Ultrasound Med. Biol.* 34(6), 930–937, Fig. 6.
#[must_use]
pub fn bbb_permeability_hill(dose: &[f64], d50: f64, hill_n: f64) -> Vec<f64> {
    let d50n = d50.powf(hill_n);
    dose.iter()
        .map(|&d| {
            let dn = d.powf(hill_n);
            dn / (d50n + dn)
        })
        .collect()
}

// ─── Inertial-damage probability model ───────────────────────────────────────

/// Inertial-cavitation damage probability as a logistic function of dose.
///
/// ```text
/// P_damage(D) = 1 / (1 + exp[-s · (D - D_thr)])
/// ```
///
/// `D_thr` is the dose at 50% damage probability and `s` controls the
/// transition steepness. The model is phenomenological: it is used by the
/// Chapter 24 BBB figure to separate stable cavitation doses from an inertial
/// damage-risk regime, not as a standalone tissue-failure proof.
///
/// # Arguments
/// * `dose` – cumulative acoustic dose [arbitrary units, e.g. MI²·s]
/// * `damage_threshold` – dose at 50% damage probability
/// * `slope` – logistic slope with reciprocal dose units; positive values make
///   probability increase with dose
#[must_use]
pub fn bbb_inertial_damage_probability(
    dose: &[f64],
    damage_threshold: f64,
    slope: f64,
) -> Vec<f64> {
    dose.iter()
        .map(|&d| 1.0 / (1.0 + (-(slope * (d - damage_threshold))).exp()))
        .collect()
}

// ─── Bi-exponential closure kinetics ─────────────────────────────────────────

/// BBB closure kinetics post-sonication: bi-exponential permeability decay.
///
/// After sonication ends, BBB permeability decays as:
/// ```text
/// P(t) = P_peak · [0.6 · exp(−t / τ_fast) + 0.4 · exp(−t / τ_slow)]
/// τ_fast = 0.5 · τ_close   (tight-junction re-assembly)
/// τ_slow = 3.0 · τ_close   (vesicular transport clearance)
/// ```
///
/// # Arguments
/// * `t_h` – time post-sonication [hours]
/// * `tau_close` – characteristic closing time constant [hours]
/// * `perm_peak` – peak permeability at t = 0 (normalised, ≤ 1.0)
///
/// # Reference
/// Deffieux & Konofagou (2010) *Ultrasound Med. Biol.* 36(7), 1117–1126, §IV.
#[must_use]
pub fn bbb_closure_kinetics(t_h: &[f64], tau_close: f64, perm_peak: f64) -> Vec<f64> {
    let tau_fast = 0.5 * tau_close;
    let tau_slow = 3.0 * tau_close;
    t_h.iter()
        .map(|&t| perm_peak * (0.6 * (-t / tau_fast).exp() + 0.4 * (-t / tau_slow).exp()))
        .collect()
}

// ─── CEUS backscatter signal ──────────────────────────────────────────────────

/// Contrast-enhanced ultrasound (CEUS) backscatter signal vs MB concentration.
///
/// Single-scattering model with MB-layer self-attenuation:
/// ```text
/// N_V       = c_mb × 10⁹   [m⁻³]   (µL gas/mL → approximate number density)
/// σ_ext     = 2 · σ_bs      [m²]    (extinction ≈ 2× backscatter, resonant MBs)
/// I_bs(c)   = σ_bs · N_V · exp(−2 · σ_ext · N_V · thickness)
/// ```
///
/// At low concentrations I_bs is linear in c_mb; self-attenuation causes a
/// peak followed by roll-off at high concentrations.
///
/// # Arguments
/// * `c_mb_ul_ml`  – MB gas concentration [µL gas / mL tissue]
/// * `sigma_bs_m2` – backscatter cross-section per bubble [m²]
/// * `thickness_m` – tissue layer thickness [m]
///
/// # Reference
/// de Jong et al. (1991) *Ultrasound Med. Biol.* 17(2), 157–169.
#[must_use]
pub fn ceus_backscatter_signal(c_mb_ul_ml: &[f64], sigma_bs_m2: f64, thickness_m: f64) -> Vec<f64> {
    let sigma_ext = 2.0 * sigma_bs_m2;
    c_mb_ul_ml
        .iter()
        .map(|&c| {
            let n_v = c * 1.0e9; // µL/mL → approximate number density [m⁻³]
            let i_linear = sigma_bs_m2 * n_v;
            let attenuation = (-2.0 * sigma_ext * n_v * thickness_m).exp();
            i_linear * attenuation
        })
        .collect()
}

/// CEUS backscatter display payload for the Chapter 24 concentration sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct CeusBackscatterDisplay {
    /// Raw single-scatter backscatter signal at each concentration.
    pub signal: Vec<f64>,
    /// Peak-normalised signal in dB, floored by `db_floor`.
    pub signal_db: Vec<f64>,
    /// Concentration sample at the discrete peak signal.
    pub peak_concentration_ul_ml: f64,
    /// Maximum raw signal value over the supplied concentration sweep.
    pub peak_signal: f64,
}

/// Compute CEUS backscatter signal, peak-normalised dB values, and peak location.
///
/// The signal model is [`ceus_backscatter_signal`]. The dB display uses
/// `20 log10(max(signal / peak, 10^(db_floor / 20)))`, so the floor is a
/// declared display bound rather than a hidden epsilon.
///
/// Returns an error when inputs are empty, non-finite, negative where the
/// physical model requires non-negativity, or when the signal peak is not
/// positive and finite.
pub fn ceus_backscatter_display(
    c_mb_ul_ml: &[f64],
    sigma_bs_m2: f64,
    thickness_m: f64,
    db_floor: f64,
) -> Result<CeusBackscatterDisplay, String> {
    if c_mb_ul_ml.is_empty() {
        return Err("CEUS concentration sweep must not be empty".to_string());
    }
    if !sigma_bs_m2.is_finite() || sigma_bs_m2 <= 0.0 {
        return Err(format!(
            "CEUS backscatter cross-section must be finite and positive, got {sigma_bs_m2}"
        ));
    }
    if !thickness_m.is_finite() || thickness_m < 0.0 {
        return Err(format!(
            "CEUS layer thickness must be finite and non-negative, got {thickness_m}"
        ));
    }
    if !db_floor.is_finite() || db_floor >= 0.0 {
        return Err(format!(
            "CEUS display dB floor must be finite and negative, got {db_floor}"
        ));
    }
    for (idx, &c) in c_mb_ul_ml.iter().enumerate() {
        if !c.is_finite() || c < 0.0 {
            return Err(format!(
                "CEUS concentration at index {idx} must be finite and non-negative, got {c}"
            ));
        }
    }

    let signal = ceus_backscatter_signal(c_mb_ul_ml, sigma_bs_m2, thickness_m);
    let (peak_idx, peak_signal) = signal
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .ok_or_else(|| "CEUS signal sweep must not be empty".to_string())?;
    if !peak_signal.is_finite() || peak_signal <= 0.0 {
        return Err(format!(
            "CEUS signal peak must be finite and positive, got {peak_signal}"
        ));
    }

    let amplitude_floor = 10.0_f64.powf(db_floor / 20.0);
    let signal_db = signal
        .iter()
        .map(|&value| 20.0 * (value / peak_signal).max(amplitude_floor).log10())
        .collect();

    Ok(CeusBackscatterDisplay {
        signal,
        signal_db,
        peak_concentration_ul_ml: c_mb_ul_ml[peak_idx],
        peak_signal,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- Hill permeability ---

    #[test]
    fn bbb_permeability_hill_at_d50_is_half() {
        let d50 = 1.2_f64;
        let n = 2.5_f64;
        let p = bbb_permeability_hill(&[d50], d50, n);
        assert!((p[0] - 0.5).abs() < 1e-12, "P(D₅₀)={} ≠ 0.5", p[0]);
    }

    #[test]
    fn bbb_permeability_hill_zero_dose_is_zero() {
        let p = bbb_permeability_hill(&[0.0], 1.0, 2.0);
        assert!(p[0].abs() < 1e-15, "P(0)={} ≠ 0", p[0]);
    }

    #[test]
    fn bbb_permeability_hill_large_dose_approaches_one() {
        let p = bbb_permeability_hill(&[1.0e9], 1.0, 2.0);
        assert!((p[0] - 1.0).abs() < 1e-6, "P(∞)={} ≠ 1.0", p[0]);
    }

    #[test]
    fn bbb_permeability_hill_monotone_increasing() {
        let d: Vec<f64> = (0..=10).map(|i| i as f64 * 0.5).collect();
        let p = bbb_permeability_hill(&d, 1.2, 2.5);
        for k in 1..p.len() {
            assert!(
                p[k] >= p[k - 1],
                "P[{}]={} < P[{}]={}",
                k,
                p[k],
                k - 1,
                p[k - 1]
            );
        }
    }

    // --- Inertial-damage probability ---

    #[test]
    fn bbb_inertial_damage_probability_at_threshold_is_half() {
        let p = bbb_inertial_damage_probability(&[3.5], 3.5, 4.0);
        assert!(
            (p[0] - 0.5).abs() < 1e-12,
            "P_damage(D_thr)={} != 0.5",
            p[0]
        );
    }

    #[test]
    fn bbb_inertial_damage_probability_matches_logistic_formula() {
        let dose = [0.0, 1.0, 3.5, 5.0];
        let threshold = 3.5_f64;
        let slope = 4.0_f64;
        let p = bbb_inertial_damage_probability(&dose, threshold, slope);
        for (actual, d) in p.iter().zip(dose) {
            let expected = 1.0 / (1.0 + (-(slope * (d - threshold))).exp());
            assert!(
                (actual - expected).abs() < 1e-12,
                "P_damage({d})={actual} != {expected}"
            );
        }
    }

    #[test]
    fn bbb_inertial_damage_probability_is_bounded_and_monotone() {
        let dose: Vec<f64> = (0..=20).map(|i| i as f64 * 0.25).collect();
        let p = bbb_inertial_damage_probability(&dose, 3.5, 4.0);
        for (k, &value) in p.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&value),
                "P_damage[{k}]={value} is outside [0, 1]"
            );
        }
        for k in 1..p.len() {
            assert!(
                p[k] >= p[k - 1],
                "P_damage[{}]={} < P_damage[{}]={}",
                k,
                p[k],
                k - 1,
                p[k - 1]
            );
        }
    }

    // --- Closure kinetics ---

    #[test]
    fn bbb_closure_kinetics_at_zero_is_perm_peak() {
        let p = bbb_closure_kinetics(&[0.0], 2.0, 0.9);
        assert!((p[0] - 0.9).abs() < 1e-12, "P(0)={} ≠ 0.9", p[0]);
    }

    #[test]
    fn bbb_closure_kinetics_decays_monotonically() {
        let t: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let p = bbb_closure_kinetics(&t, 2.0, 1.0);
        for k in 1..p.len() {
            assert!(
                p[k] <= p[k - 1],
                "P[{}]={} > P[{}]={}",
                k,
                p[k],
                k - 1,
                p[k - 1]
            );
        }
    }

    #[test]
    fn bbb_closure_kinetics_approaches_zero() {
        let p = bbb_closure_kinetics(&[1.0e6], 1.0, 1.0);
        assert!(p[0].abs() < 1e-6, "P(∞)={} ≠ 0", p[0]);
    }

    // --- CEUS backscatter ---

    #[test]
    fn ceus_backscatter_zero_concentration_is_zero() {
        let s = ceus_backscatter_signal(&[0.0], 2.5e-8, 10e-3);
        assert!(s[0].abs() < 1e-30, "I(0)={} ≠ 0", s[0]);
    }

    #[test]
    fn ceus_backscatter_peaks_then_falls() {
        // Analytical peak: N_V* = 1/(2·σ_ext·thickness) = 1/(2·5e-8·10e-3) = 1e9 m⁻³
        // → c_mb* = N_V* / 1e9 = 1.0 µL/mL.
        // Values below and above the peak must bracket the maximum.
        let sigma_bs = 2.5e-8_f64;
        let thickness = 10e-3_f64;
        let low = ceus_backscatter_signal(&[1e-4], sigma_bs, thickness)[0]; // well below peak
        let peak = ceus_backscatter_signal(&[1.0], sigma_bs, thickness)[0]; // at analytical peak
        let high = ceus_backscatter_signal(&[100.0], sigma_bs, thickness)[0]; // well above peak
        assert!(peak > low, "no rise:  peak={} low={}", peak, low);
        assert!(peak > high, "no fall:  peak={} high={}", peak, high);
    }

    #[test]
    fn ceus_backscatter_display_matches_signal_and_peak() {
        let concentrations = [0.0, 1.0, 100.0];
        let display = ceus_backscatter_display(&concentrations, 2.5e-8, 10e-3, -80.0).unwrap();
        let signal = ceus_backscatter_signal(&concentrations, 2.5e-8, 10e-3);

        assert_eq!(display.signal, signal);
        assert_eq!(display.peak_concentration_ul_ml, 1.0);
        assert_eq!(display.peak_signal, signal[1]);
        assert_eq!(display.signal_db[1], 0.0);
        assert_eq!(display.signal_db[0], -80.0);
        assert!(display.signal_db[2] < 0.0);
    }

    #[test]
    fn ceus_backscatter_display_rejects_invalid_inputs() {
        let err = ceus_backscatter_display(&[], 2.5e-8, 10e-3, -80.0).unwrap_err();
        assert!(err.contains("must not be empty"));

        let err = ceus_backscatter_display(&[-1.0], 2.5e-8, 10e-3, -80.0).unwrap_err();
        assert!(err.contains("non-negative"));

        let err = ceus_backscatter_display(&[0.0], 2.5e-8, 10e-3, -80.0).unwrap_err();
        assert!(err.contains("peak"));
    }
}
