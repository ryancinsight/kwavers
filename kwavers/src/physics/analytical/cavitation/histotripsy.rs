use crate::math::statistics::erf;

/// Single-pulse intrinsic-threshold cavitation probability (Gaussian erf-CDF model).
///
/// Maxwell et al. (2013) showed experimentally that the single-pulse probability
/// of histotripsy cavitation at pressure magnitude `|p⁻|` follows a Gaussian
/// cumulative distribution function with threshold `p_T` and width `σ`:
///
/// ```text
/// P_cav(|p⁻|) = ½ · (1 + erf((|p⁻| − p_T) / (σ · √2)))   [Theorem 21.1]
/// ```
///
/// At `|p⁻| = p_T` → `P_cav = 0.5` (50 % probability per pulse).
/// At `|p⁻| ≪ p_T` → `P_cav → 0` (sub-threshold).
/// At `|p⁻| ≫ p_T` → `P_cav → 1` (deterministic cavitation).
///
/// The erf is evaluated via the Abramowitz & Stegun 7.1.26 rational approximation
/// (|ε| ≤ 1.5×10⁻⁷).
///
/// # Arguments
/// * `p_arr` – array of |peak negative pressure| values [Pa]
/// * `p_threshold` – mean intrinsic threshold [Pa] (bovine liver, 1 MHz: 28.2 MPa)
/// * `sigma_pa` – standard deviation [Pa] (bovine liver, 1 MHz: 0.96 MPa)
///
/// # Reference
/// Maxwell et al. (2013) *Ultrasound Med. Biol.* 39, 449, Table II.
/// Macoskey et al. (2018) *Phys. Med. Biol.* 63, 175022.
#[must_use]
pub fn intrinsic_threshold_cavitation_probability(
    p_arr: &[f64],
    p_threshold: f64,
    sigma_pa: f64,
) -> Vec<f64> {
    use std::f64::consts::SQRT_2;
    let denom = (sigma_pa * SQRT_2).max(f64::MIN_POSITIVE);
    p_arr
        .iter()
        .map(|&p| 0.5 * (1.0 + erf((p - p_threshold) / denom)))
        .collect()
}

/// Frequency-dependent intrinsic cavitation threshold (Vlaisavljevich 2015 log-linear fit).
///
/// Water-rich soft tissue shows a log-linear dependence of the mean intrinsic
/// threshold peak negative pressure on frequency:
///
/// ```text
/// p_T(f) = p_T(1 MHz) + slope · log₁₀(f / 1 MHz)   [Pa]
/// ```
///
/// Canonical values for bovine liver (Vlaisavljevich et al. 2015 Table I):
/// * `p_T(1 MHz) = 28.2 MPa`
/// * `slope = 1.4 MPa per decade` (over 0.25–3 MHz)
///
/// # Arguments
/// * `f_hz` – frequencies [Hz]
/// * `p_t_1mhz_pa` – threshold at 1 MHz [Pa]
/// * `slope_pa_per_decade` – slope [Pa] per factor-of-10 increase in frequency
///
/// # Reference
/// Vlaisavljevich et al. (2015), *Ultrasound Med. Biol.* 41, 1251, Table I.
/// Maxwell et al. (2013), *Ultrasound Med. Biol.* 39, 449, Table II.
#[must_use]
pub fn frequency_dependent_intrinsic_threshold_pa(
    f_hz: &[f64],
    p_t_1mhz_pa: f64,
    slope_pa_per_decade: f64,
) -> Vec<f64> {
    const F_REF: f64 = 1.0e6; // 1 MHz reference
    f_hz.iter()
        .map(|&f| {
            let f_pos = f.max(f64::MIN_POSITIVE);
            p_t_1mhz_pa + slope_pa_per_decade * (f_pos / F_REF).log10()
        })
        .collect()
}

/// Cumulative cavitation probability over N independent single-pulse trials.
///
/// Each pulse produces cavitation with probability P_single (Maxwell 2013 erf-CDF
/// model).  Assuming statistical independence across pulses:
///
/// ```text
/// P_cum(N) = 1 − (1 − P_single)^N
/// ```
///
/// For non-integer N (e.g. continuous pulse-duration sweep), the binomial law is
/// analytically continued via:
/// ```text
/// (1 − P_single)^N = exp(N · ln(1 − P_single))
/// ```
///
/// N is clamped to ≥ 1.0 before evaluation; the function returns P_single at N = 1.
///
/// # Arguments
/// * `p_single` – single-pulse cavitation probability ∈ [0, 1]
/// * `n_pulses_arr` – pulse count array N (may be non-integer, ≥ 0)
///
/// # Reference
/// Maxwell et al. (2013), *Ultrasound Med. Biol.* 39, 449.
/// Vlaisavljevich et al. (2015), *Ultrasound Med. Biol.* 41, 1251.
#[must_use]
pub fn cumulative_cavitation_probability(p_single: f64, n_pulses_arr: &[f64]) -> Vec<f64> {
    let p_clamped = p_single.clamp(0.0, 1.0);
    let ln_q = (1.0 - p_clamped).ln(); // ln(1 - P_single); −∞ at P=1 → handled
    n_pulses_arr
        .iter()
        .map(|&n| {
            let n = n.max(1.0);
            if p_clamped >= 1.0 {
                1.0
            } else if p_clamped <= 0.0 {
                0.0
            } else {
                1.0 - (n * ln_q).exp()
            }
        })
        .collect()
}

/// PRF efficacy factor — residual-bubble shielding model (Macoskey 2018).
///
/// At high PRF, residual bubble clouds from previous pulses have not fully
/// dissolved before the next pulse arrives, causing acoustic shadowing.  The
/// per-pulse treatment efficacy decays exponentially once the pulse repetition
/// period falls below the bubble dissolution time τ_d:
///
/// ```text
/// E(PRF) = exp(−max(0, PRF · τ_d − 1) · g)
/// ```
///
/// where `g` is a dimensionless shielding gain coefficient (Macoskey 2018
/// fitted g ≈ 1.2 for porcine liver at 1 MHz).
///
/// * At PRF·τ_d ≤ 1 (period ≥ τ_d): E = 1 — full efficacy, complete dissolution.
/// * As PRF·τ_d → ∞: E → 0 — total shielding.
///
/// The normalised lesion-volume rate is then proportional to `PRF × E(PRF)`,
/// yielding an optimum near PRF ≈ 1/τ_d.
///
/// # Arguments
/// * `prf_hz` – pulse repetition frequencies [Hz]
/// * `bubble_dissolution_time_s` – residual-bubble dissolution time τ_d [s]
///   (liver: ~5 ms; Vlaisavljevich 2015)
/// * `shielding_coefficient` – exponential decay gain `g` (dimensionless)
///
/// # Reference
/// Macoskey et al. (2018), *Ultrasound Med. Biol.* 44, 2971.
/// Vlaisavljevich et al. (2015), *Ultrasound Med. Biol.* 41, 1251.
#[must_use]
pub fn prf_efficacy_factor(
    prf_hz: &[f64],
    bubble_dissolution_time_s: f64,
    shielding_coefficient: f64,
) -> Vec<f64> {
    prf_hz
        .iter()
        .map(|&prf| {
            let excess = (prf * bubble_dissolution_time_s - 1.0).max(0.0);
            (-excess * shielding_coefficient).exp()
        })
        .collect()
}

/// FDA mechanical index: peak negative pressure normalised by √(frequency).
///
/// ```text
/// MI = |P_neg| [MPa] / √(f [MHz])
/// ```
/// FDA safety guideline for diagnostic imaging: MI < 1.9.
/// Histotripsy (intrinsic threshold) requires MI > 3 for microsecond pulses.
///
/// Delegates to the canonical
/// [`crate::physics::acoustics::analysis::calculate_mechanical_index`] so the
/// book-chapter API and the production safety paths share a single contract.
///
/// # Reference
/// Apfel & Holland (1991), *Ultrasound Med. Biol.* 17, 179.
#[must_use]
#[inline]
pub fn mechanical_index(p_neg_pa: f64, freq_hz: f64) -> f64 {
    crate::physics::acoustics::analysis::calculate_mechanical_index(p_neg_pa, freq_hz)
}

/// Inertial cavitation dose (ICD) from a bubble radius time series.
///
/// Accumulates the normalised collapse strength over all detected inertial
/// collapse events.  A collapse event is a local minimum of `R` below `R₀`
/// coinciding with a sign change of `Ṙ` from negative to non-negative:
/// ```text
/// ICD = Σ_{collapse events i} (R_max_i / R₀)³   [dimensionless]
/// ```
/// `R_max_i` is the maximum bubble radius reached during the expansion phase
/// immediately preceding the ith collapse.  The cubic weighting is proportional
/// to the maximum volume ratio and therefore to the inertial energy.
///
/// # Reference
/// Duryea et al. (2015), *Ultrasound Med. Biol.* 41, 1937.
#[must_use]
pub fn inertial_cavitation_dose(r_arr: &[f64], rdot_arr: &[f64], r0_m: f64) -> f64 {
    let r0 = r0_m.max(1e-15);
    let n = r_arr.len().min(rdot_arr.len());
    if n < 2 {
        return 0.0;
    }
    let mut dose = 0.0_f64;
    let mut r_max = r0;
    for i in 0..n - 1 {
        let r = r_arr[i].max(1e-15);
        r_max = r_max.max(r);
        let is_min = rdot_arr[i] < 0.0 && rdot_arr[i + 1] >= 0.0 && r < r0;
        if is_min {
            dose += (r_max / r0).powi(3);
            r_max = r0;
        }
    }
    dose
}

/// Estimated histotripsy lesion radius from cavitation energy balance.
///
/// ## Derivation
/// The Rayleigh–Plesset energy released during inertial collapse equals the
/// PdV work against ambient pressure over the bubble volume excursion:
/// ```text
/// E_collapse ≈ (4π/3) · P₀ · R_max³    [per event]
/// ```
/// Summing over all ICD events: `E_total = (4π/3) · P₀ · R₀³ · ICD`.
/// Setting `E_total = σ_y · (4π/3) · R_L³` and solving for `R_L`:
/// ```text
/// R_L = R₀ · (P₀ · ICD / σ_y)^(1/3)   [m]
/// ```
///
/// # Arguments
/// * `icd` – dimensionless inertial cavitation dose (from `inertial_cavitation_dose`)
/// * `r0_m` – equilibrium bubble radius [m]
/// * `p0_pa` – ambient pressure [Pa]
/// * `tissue_yield_stress_pa` – tensile yield stress of tissue [Pa]
///   (brain white matter: 1–4 kPa; Vlaisavljevich et al. 2015)
///
/// # Reference
/// Maxwell et al. (2011), *J. Acoust. Soc. Am.* 130, 2012.
/// Vlaisavljevich et al. (2015), *Ultrasound Med. Biol.* 41, 2896.
#[must_use]
#[inline]
pub fn histotripsy_lesion_radius_m(
    icd: f64,
    r0_m: f64,
    p0_pa: f64,
    tissue_yield_stress_pa: f64,
) -> f64 {
    if !(icd.is_finite()
        && r0_m.is_finite()
        && p0_pa.is_finite()
        && tissue_yield_stress_pa.is_finite()
        && icd >= 0.0
        && r0_m > 0.0
        && p0_pa > 0.0
        && tissue_yield_stress_pa > 0.0)
    {
        return 0.0;
    }
    // tissue_yield_stress_pa is already validated > 0 above; no further clamp.
    // Prior to 2026-05-21 a .max(1.0) silently floored any sub-1-Pa yield
    // stress to 1 Pa, mis-reporting cavitation lesion radii for very soft
    // materials (e.g. gel phantoms with σ_y ≪ 1 Pa) by replacing the true
    // yield stress with an arbitrary minimum.
    r0_m * (p0_pa * icd / tissue_yield_stress_pa).cbrt()
}
