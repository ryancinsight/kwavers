/// FDA mechanical index: peak negative pressure normalised by √(frequency).
///
/// ```text
/// MI = |P_neg| [MPa] / √(f [MHz])
/// ```
/// FDA safety guideline for diagnostic imaging: MI < 1.9.
/// Histotripsy (intrinsic threshold) requires MI > 3 for microsecond pulses.
///
/// # Reference
/// Apfel & Holland (1991), *Ultrasound Med. Biol.* 17, 179.
#[inline]
pub fn mechanical_index(p_neg_pa: f64, freq_hz: f64) -> f64 {
    let f_mhz = freq_hz * 1e-6;
    if !(p_neg_pa.is_finite() && f_mhz.is_finite() && f_mhz > 0.0) {
        return 0.0;
    }
    let p_neg_mpa = p_neg_pa.abs() * 1e-6;
    p_neg_mpa / f_mhz.sqrt()
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
    let sigma_y = tissue_yield_stress_pa.max(1.0);
    r0_m * (p0_pa * icd / sigma_y).cbrt()
}
