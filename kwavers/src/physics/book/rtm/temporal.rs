/// Temporal modulation frequency schedule for standing-wave suppression.
///
/// ```text
/// f_m = f₀ + m · c / (2·M·d_back)   for m = 0..M-1
/// ```
/// Each frequency shifts the standing-wave pattern by one lobe-width, so
/// coherent averaging over M frequencies cancels the standing-wave modulation.
///
/// # Arguments
/// * `f0_hz` – base frequency [Hz]
/// * `m_steps` – number of frequencies M
/// * `c` – sound speed [m/s]
/// * `d_back_m` – distance to the back-reflecting wall [m]
///
/// # Reference
/// Dencks & Schmitz (2005), *Ultrasonics* 43, 183.
pub fn temporal_modulation_frequencies(
    f0_hz: f64,
    m_steps: usize,
    c: f64,
    d_back_m: f64,
) -> Vec<f64> {
    let df = c / (2.0 * m_steps as f64 * d_back_m);
    (0..m_steps).map(|m| f0_hz + m as f64 * df).collect()
}

/// Analytical gain factor for standing-wave suppression by RTM.
///
/// For a back-reflection coefficient R_back, the peak-to-trough ratio of the
/// standing-wave modulation is `(1 + R_back)²/(1 + R_back²)`. Perfect RTM
/// suppression leaves only the smooth background; the relative gain is:
/// ```text
/// G = (1 + R_back)² / (1 + R_back²)
/// ```
///
/// # Reference
/// Thomas et al. (2017), *Phys. Rev. Lett.* 119, 034301.
#[inline]
pub fn standing_wave_suppression_gain(r_back: f64) -> f64 {
    (1.0 + r_back).powi(2) / (1.0 + r_back * r_back)
}
