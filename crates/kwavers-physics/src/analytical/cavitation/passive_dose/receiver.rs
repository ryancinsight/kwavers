//! Integration of multi-element passive-cavitation-detector (PCD) data over a
//! sonication volume `V_s`.
//!
//! A clinical array (e.g. the Exablate hemisphere) listens on many receive
//! elements during sonication. Without beamforming, the cavitation dose for the
//! focal sonication volume is obtained by **incoherently summing the per-element
//! power spectra** — the array integral that gives sensitivity to the whole
//! focal region rather than a single point. When a passive acoustic map (PAM)
//! is available instead, the emission energy is integrated over the spatial
//! `V_s` mask.

/// Incoherent power sum of per-element PCD spectra (array integration over `V_s`).
///
/// Given `n_channels` receive-element power spectra each of length `n_bins`,
/// laid out row-major (`channel_psds[ch*n_bins + bin]`), returns the
/// array-integrated spectrum `S(f) = Σ_ch S_ch(f)` of length `n_bins`. Power is
/// summed (not amplitude) because passive emissions from independent collapse
/// events are mutually incoherent, so their powers add (Gyöngy & Coussios 2010).
///
/// Returns an empty vector if `n_channels·n_bins` does not match the input
/// length or either dimension is zero.
///
/// # Reference
/// Gyöngy M. & Coussios C.C. (2010) *IEEE Trans. Biomed. Eng.* 57, 48.
#[must_use]
pub fn integrate_receiver_array_psd(
    channel_psds: &[f64],
    n_channels: usize,
    n_bins: usize,
) -> Vec<f64> {
    if n_channels == 0 || n_bins == 0 || channel_psds.len() != n_channels * n_bins {
        return Vec::new();
    }
    let mut out = vec![0.0_f64; n_bins];
    for ch in 0..n_channels {
        let base = ch * n_bins;
        for bin in 0..n_bins {
            out[bin] += channel_psds[base + bin];
        }
    }
    out
}

/// Integrate a passive-acoustic-map emission-energy field over a sonication
/// volume mask `V_s`.
///
/// ```text
///   E(V_s) = Σ_{voxels v ∈ V_s} source_map[v] · dV       [emission-energy]
/// ```
/// `source_map` and `mask` are flattened 3-D fields of equal length; `mask[v]`
/// is treated as "inside `V_s`" when it is non-zero. `dv_m3` is the voxel volume
/// `Δx·Δy·Δz` [m³]. Negative source samples are clamped to 0.
///
/// Returns 0.0 if the lengths differ, are empty, or `dv_m3` is non-positive.
#[must_use]
pub fn emission_energy_in_volume(source_map: &[f64], mask: &[f64], dv_m3: f64) -> f64 {
    if source_map.is_empty()
        || source_map.len() != mask.len()
        || !(dv_m3.is_finite() && dv_m3 > 0.0)
    {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for (&s, &m) in source_map.iter().zip(mask.iter()) {
        if m != 0.0 {
            acc += s.max(0.0);
        }
    }
    acc * dv_m3
}
