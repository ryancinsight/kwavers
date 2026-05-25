//! Blood-brain barrier (BBB) permeability and closure kinetic models for ch24.
//!
//! Covers:
//! * Hill-function dose-response model for BBB permeability enhancement
//!   (McDannold 2008, Evans-blue extravasation fit).
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
pub fn ceus_backscatter_signal(
    c_mb_ul_ml: &[f64],
    sigma_bs_m2: f64,
    thickness_m: f64,
) -> Vec<f64> {
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
            assert!(p[k] >= p[k - 1], "P[{}]={} < P[{}]={}", k, p[k], k - 1, p[k - 1]);
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
            assert!(p[k] <= p[k - 1], "P[{}]={} > P[{}]={}", k, p[k], k - 1, p[k - 1]);
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
        let low  = ceus_backscatter_signal(&[1e-4], sigma_bs, thickness)[0]; // well below peak
        let peak = ceus_backscatter_signal(&[1.0],  sigma_bs, thickness)[0]; // at analytical peak
        let high = ceus_backscatter_signal(&[100.0], sigma_bs, thickness)[0]; // well above peak
        assert!(peak > low,  "no rise:  peak={} low={}",  peak, low);
        assert!(peak > high, "no fall:  peak={} high={}", peak, high);
    }
}
