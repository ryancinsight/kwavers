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
#[must_use]
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
#[must_use]
#[inline]
pub fn standing_wave_suppression_gain(r_back: f64) -> f64 {
    (1.0 + r_back).powi(2) / (1.0 + r_back * r_back)
}

// ─── Standing-wave spatial frequency ────────────────────────────────────────

/// Axial spatial frequency of the standing-wave pattern [cycles/m].
///
/// For a plane wave at frequency `f` in a medium with sound speed `c`, the
/// standing-wave antinode-to-antinode spacing is λ/2 = c/(2f), so the
/// spatial frequency in cycles/m is:
///
/// ```text
/// k_sw = 2·f / c   [cycles/m]
/// ```
///
/// This is the location of the peak in an axial spatial-frequency spectrum of
/// the standing-wave intensity, and equals 1/(half-wavelength).
///
/// # Arguments
/// * `freq_hz` – frequency [Hz]
/// * `c` – sound speed [m/s]
///
/// # Reference
/// Brekhovskikh & Godin (1990) *Acoustics of Layered Media I*, §1.2.
#[must_use]
#[inline]
pub fn standing_wave_spatial_frequency_cycles_m(freq_hz: f64, c: f64) -> f64 {
    2.0 * freq_hz / c
}

// ─── Standing-wave period ────────────────────────────────────────────────────

/// Period of one full standing-wave cycle in frequency [Hz].
///
/// For a back-reflector at distance `d_back_m`, the standing-wave pattern
/// completes one full oscillation cycle when the frequency shifts by:
/// ```text
/// ΔF_period = c / (2 · d_back)
/// ```
/// This is the reciprocal of the round-trip travel time to the back wall.
///
/// # Arguments
/// * `c` – sound speed [m/s]
/// * `d_back_m` – distance from field point to back-reflecting wall [m]
///
/// # Reference
/// Dencks & Schmitz (2005), *Ultrasonics* 43, 183.
#[must_use]
#[inline]
pub fn standing_wave_modulation_period_hz(c: f64, d_back_m: f64) -> f64 {
    c / (2.0 * d_back_m)
}

// ─── Standing-wave field ─────────────────────────────────────────────────────

/// 1-D standing-wave intensity pattern: |1 + R·exp(2ik·x)|².
///
/// For a plane wave in a 1-D medium with a back-reflector of pressure
/// reflection coefficient `r_back` at `x = 0`, the standing-wave intensity
/// at distance `x` from the reflector is:
///
/// ```text
/// SW²(x) = |1 + R · exp(2i·k·x)|²
///         = 1 + R² + 2R·cos(2kx)
/// k = 2π·f / c
/// ```
///
/// This is the exact standing-wave pattern in a lossless 1-D medium.
/// The time-averaged value over a full spatial period (or over M ≥ 2 uniformly
/// spaced modulation steps) converges to `1 + R²` (see `standing_wave_intensity_statistics`).
///
/// # Arguments
/// * `x_arr` – distances from the back reflector [m]; may include negative or
///   zero values (evaluated as-is; masking is the caller's responsibility)
/// * `freq_hz` – frequency [Hz]
/// * `c` – sound speed [m/s]
/// * `r_back` – pressure reflection coefficient (|R| ≤ 1)
///
/// # Reference
/// Brekhovskikh & Godin (1990) *Acoustics of Layered Media I*, §1.2.
#[must_use]
pub fn standing_wave_field_1d(
    x_arr: &[f64],
    freq_hz: f64,
    c: f64,
    r_back: f64,
) -> Vec<f64> {
    use std::f64::consts::PI;
    let k = 2.0 * PI * freq_hz / c;
    let two_k = 2.0 * k;
    let r2 = r_back * r_back;
    x_arr
        .iter()
        .map(|&x| {
            // |1 + R·exp(2ikx)|² = 1 + R² + 2R·cos(2kx) — exact, no complex arithmetic
            1.0 + r2 + 2.0 * r_back * (two_k * x).cos()
        })
        .collect()
}

// ─── Standing-wave statistics ────────────────────────────────────────────────

/// Exact statistical moments of the standing-wave intensity pattern.
///
/// For a back-reflection coefficient `R`, the pattern `|1 + R·exp(2ikx)|²`
/// has the following extremes and time/ensemble average:
///
/// ```text
/// SW²_mean   = 1 + R²           (spatial average, exact)
/// SW²_peak   = (1 + R)²         (antinodal maximum)
/// SW²_trough = (1 − R)²         (nodal minimum)
/// ```
///
/// The mean is also the ensemble average achieved by RTM temporal modulation
/// with M ≥ 2 uniformly spaced steps spanning one period ΔF (exact result).
///
/// Returns `(sw2_mean, sw2_peak, sw2_trough)`.
///
/// # Reference
/// Thomas et al. (2017), *Phys. Rev. Lett.* 119, 034301, eq. (3).
#[must_use]
#[inline]
pub fn standing_wave_intensity_statistics(r_back: f64) -> (f64, f64, f64) {
    let r2 = r_back * r_back;
    let sw2_mean = 1.0 + r2;
    let sw2_peak = (1.0 + r_back).powi(2);
    let sw2_trough = (1.0 - r_back).powi(2);
    (sw2_mean, sw2_peak, sw2_trough)
}
