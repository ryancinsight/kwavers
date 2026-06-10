//! Direct spatial-wavelength estimation for shear-wave elastography (§11.10).
//!
//! The shear wavelength `λ_S` sets the spatial resolution of an SWE map. When
//! the shear-wave **speed** is unknown, `λ_S` can be estimated *directly* from a
//! measured 1-D displacement profile by **spatial autocorrelation**: for a
//! quasi-monochromatic wave `u(x) ≈ A sin(2πx/λ + φ)`, the (biased)
//! autocorrelation `R(m)` is proportional to `cos(2π m·dx/λ)`, whose first
//! positive peak after the central lobe sits at lag `m·dx = λ`. Parabolic
//! interpolation around that peak gives a sub-sample estimate.
//!
//! This complements the analytical `λ_S = c_S/f` (which needs a known `c_S`)
//! and the local-frequency / phase-gradient inversions (which estimate `c_S`).
//!
//! # References
//! - Manduca, A., et al. (2001). "Magnetic resonance elastography: Non-invasive
//!   mapping of tissue elasticity." *Med. Image Anal.* 5(4), 237–254 (LFE / local
//!   spatial-frequency estimation).

/// Estimate the dominant spatial **wavelength** \[m] of a 1-D shear-wave
/// displacement profile `displacement` sampled at spacing `dx` \[m], by
/// autocorrelation (first post-zero-crossing peak, parabolically interpolated).
///
/// Returns `None` when the input is too short (`< 8` samples), `dx ≤ 0`, the
/// profile is essentially constant (no oscillation), or fewer than ~one full
/// wavelength is present (the autocorrelation never forms a peak after its
/// first zero crossing).
#[must_use]
pub fn estimate_shear_wavelength(displacement: &[f64], dx: f64) -> Option<f64> {
    let n = displacement.len();
    let dx_valid = dx.is_finite() && dx > 0.0;
    if n < 8 || !dx_valid || !displacement.iter().all(|v| v.is_finite()) {
        return None;
    }

    // Mean-detrend (removes any DC offset that would bias the autocorrelation).
    let mean = displacement.iter().sum::<f64>() / n as f64;
    let u: Vec<f64> = displacement.iter().map(|v| v - mean).collect();

    // Biased autocorrelation R[m] = Σ_n u[n]·u[n+m], m = 0 … n-2.
    let max_lag = n - 1;
    let mut r = vec![0.0_f64; max_lag + 1];
    for (m, rm) in r.iter_mut().enumerate() {
        let mut acc = 0.0;
        for k in 0..(n - m) {
            acc += u[k] * u[k + m];
        }
        *rm = acc;
    }
    if r[0] <= 0.0 {
        return None; // constant / zero signal
    }

    // First lag where R goes negative (≈ λ/4): the wave must oscillate.
    let m_zero = (1..=max_lag).find(|&m| r[m] < 0.0)?;

    // The first (and, for a biased estimate, largest) positive peak after the
    // zero crossing sits at lag ≈ λ. Take the global argmax over [m_zero, …],
    // keeping one neighbour on each side for interpolation.
    let search_hi = max_lag.saturating_sub(1);
    if m_zero + 1 > search_hi {
        return None;
    }
    let mut m_peak = m_zero;
    let mut best = f64::NEG_INFINITY;
    for (m, &rm) in r.iter().enumerate().take(search_hi + 1).skip(m_zero) {
        if rm > best {
            best = rm;
            m_peak = m;
        }
    }
    if m_peak == 0 || m_peak >= max_lag || best <= 0.0 {
        return None;
    }

    // Parabolic sub-sample refinement of the peak lag.
    let (ym, y0, yp) = (r[m_peak - 1], r[m_peak], r[m_peak + 1]);
    let denom = ym - 2.0 * y0 + yp;
    let delta = if denom.abs() > 0.0 {
        0.5 * (ym - yp) / denom
    } else {
        0.0
    };
    Some((m_peak as f64 + delta) * dx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    /// Sample `n` points of `A·sin(2πx/λ + φ)` at spacing `dx`.
    fn sinusoid(lambda: f64, dx: f64, n: usize, amp: f64, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (TAU * (i as f64 * dx) / lambda + phase).sin())
            .collect()
    }

    /// A clean sinusoid of known wavelength is recovered to sub-sample accuracy.
    #[test]
    fn recovers_known_wavelength() {
        let dx = 0.2e-3; // 0.2 mm
        for &lambda in &[2.0e-3, 4.0e-3, 8.0e-3] {
            // ~5 wavelengths of data.
            let n = (5.0 * lambda / dx) as usize;
            let u = sinusoid(lambda, dx, n, 1.0, 0.3);
            let est = estimate_shear_wavelength(&u, dx).expect("estimate");
            assert!(
                (est - lambda).abs() / lambda < 0.02,
                "λ={lambda} recovered {est} (>2% error)"
            );
        }
    }

    /// A DC offset does not bias the estimate (detrending).
    #[test]
    fn dc_offset_is_removed() {
        let dx = 0.25e-3;
        let lambda = 5.0e-3;
        let n = (6.0 * lambda / dx) as usize;
        let u: Vec<f64> = sinusoid(lambda, dx, n, 1.0, 0.0)
            .iter()
            .map(|v| v + 7.5) // large DC bias
            .collect();
        let est = estimate_shear_wavelength(&u, dx).expect("estimate");
        assert!((est - lambda).abs() / lambda < 0.03, "λ recovered {est}");
    }

    /// The estimate scales linearly with the true wavelength.
    #[test]
    fn estimate_scales_with_wavelength() {
        let dx = 0.2e-3;
        let n = 400;
        let e1 = estimate_shear_wavelength(&sinusoid(3.0e-3, dx, n, 1.0, 0.0), dx).unwrap();
        let e2 = estimate_shear_wavelength(&sinusoid(6.0e-3, dx, n, 1.0, 0.0), dx).unwrap();
        assert!((e2 / e1 - 2.0).abs() < 0.05, "ratio {}", e2 / e1);
    }

    /// Degenerate inputs return `None`.
    #[test]
    fn degenerate_inputs_return_none() {
        assert!(estimate_shear_wavelength(&[1.0; 4], 1e-3).is_none()); // too short
        assert!(estimate_shear_wavelength(&[2.5; 64], 1e-3).is_none()); // constant
        assert!(estimate_shear_wavelength(&sinusoid(4e-3, 2e-4, 64, 1.0, 0.0), 0.0).is_none()); // dx≤0
        // A monotone ramp never oscillates ⇒ no peak.
        let ramp: Vec<f64> = (0..64).map(|i| i as f64).collect();
        assert!(estimate_shear_wavelength(&ramp, 1e-3).is_none());
    }
}
