//! Diagnostic ultrasound imaging physics for book chapter ch05.
//!
//! Covers: lateral and axial PSF models, Doppler frequency shift,
//! plane-wave compounding PSF, and resolution limits.

use std::f64::consts::PI;

// ─── PSF models ───────────────────────────────────────────────────────────────

/// Lateral point spread function — sinc² approximation for a uniform aperture.
///
/// ```text
/// PSF_lat(x) = sinc²(x / (0.886·F#·λ))
/// ```
/// where `sinc(u) = sin(πu)/(πu)` and 0.886 accounts for the −6 dB width
/// of the sinc² function equalling the Rayleigh criterion.
///
/// Normalised to 1.0 at x = 0.
///
/// # Arguments
/// * `x_arr` – lateral offsets from beam axis [m]
/// * `f_number` – F-number = focal_length / aperture
/// * `wavelength_m` – acoustic wavelength λ [m]
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §8.3.
#[must_use]
#[inline]
pub fn lateral_psf_sinc2(x_arr: &[f64], f_number: f64, wavelength_m: f64) -> Vec<f64> {
    let width = 0.886 * f_number * wavelength_m;
    x_arr
        .iter()
        .map(|&x| {
            let u = x / width;
            sinc2(u)
        })
        .collect()
}

/// Axial point spread function — sinc² for a rectangular frequency spectrum.
///
/// ```text
/// PSF_ax(z) = sinc²(2·z·BW / c)
/// ```
///
/// Normalised to 1.0 at z = 0.
///
/// # Arguments
/// * `z_arr` – axial offsets from focal plane [m]
/// * `c` – sound speed [m/s]
/// * `bandwidth_hz` – receiver −6 dB bandwidth [Hz]
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §6.5.
#[must_use]
#[inline]
pub fn axial_psf_rect(z_arr: &[f64], c: f64, bandwidth_hz: f64) -> Vec<f64> {
    z_arr
        .iter()
        .map(|&z| {
            let u = 2.0 * z * bandwidth_hz / c;
            sinc2(u)
        })
        .collect()
}

// ─── Doppler ──────────────────────────────────────────────────────────────────

/// Doppler frequency shift for a moving reflector.
///
/// ```text
/// Δf = 2·f₀·v·cos(θ) / c   [Hz]
/// ```
///
/// Positive for motion towards the transducer (θ < π/2).
///
/// # Arguments
/// * `v_m_s` – reflector speed [m/s]
/// * `theta_rad` – angle between flow direction and beam axis [rad]
/// * `f0_hz` – transmit centre frequency [Hz]
/// * `c` – sound speed [m/s]
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §11.2.
#[must_use]
#[inline]
pub fn doppler_frequency_shift(v_m_s: f64, theta_rad: f64, f0_hz: f64, c: f64) -> f64 {
    2.0 * f0_hz * v_m_s * theta_rad.cos() / c
}

// ─── Plane-wave compounding ───────────────────────────────────────────────────

/// Effective lateral PSF for coherent plane-wave compounding.
///
/// Each compounding angle contributes a sinc² PSF shifted in angle; coherent
/// averaging reduces the effective FWHM by roughly 1/√N_angles. The
/// resulting PSF is approximated as:
/// ```text
/// PSF_comp(x) = sinc²(x / (0.886·F#·λ / √N_angles))
/// ```
///
/// # Arguments
/// * `x_arr` – lateral offsets [m]
/// * `n_angles` – number of compounding angles
/// * `f_number` – effective F-number for a single angle
/// * `wavelength_m` – acoustic wavelength [m]
///
/// # Reference
/// Montaldo et al. (2009), *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*
/// 56, 489.
#[must_use]
pub fn pw_compounding_lateral_psf(
    x_arr: &[f64],
    n_angles: usize,
    f_number: f64,
    wavelength_m: f64,
) -> Vec<f64> {
    let eff_width = 0.886 * f_number * wavelength_m / (n_angles as f64).sqrt();
    x_arr.iter().map(|&x| sinc2(x / eff_width)).collect()
}

// ─── Resolution limit ─────────────────────────────────────────────────────────

/// −6 dB lateral resolution (Rayleigh criterion).
///
/// ```text
/// δx = 0.886 · F# · λ   [m]
/// ```
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §8.3.
#[must_use]
#[inline]
pub fn lateral_resolution_m(f_number: f64, wavelength_m: f64) -> f64 {
    0.886 * f_number * wavelength_m
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Normalised sinc-squared: sinc²(u) = (sin(πu)/(πu))²
#[inline]
fn sinc2(u: f64) -> f64 {
    if u.abs() < 1e-12 {
        1.0
    } else {
        let s = (PI * u).sin() / (PI * u);
        s * s
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;

    #[test]
    fn lateral_psf_peak_at_zero() {
        let psf = lateral_psf_sinc2(&[0.0], 2.0, 1e-3);
        assert!((psf[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn axial_psf_peak_at_zero() {
        let psf = axial_psf_rect(&[0.0], SOUND_SPEED_WATER_SIM, 5.0 * MHZ_TO_HZ);
        assert!((psf[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lateral_psf_decreases_away_from_axis() {
        let lam = SOUND_SPEED_WATER_SIM / (2.0 * MHZ_TO_HZ);
        let psf = lateral_psf_sinc2(&[0.0, 0.5e-3, 1e-3], 2.0, lam);
        assert!(psf[0] > psf[1] && psf[1] > psf[2]);
    }

    #[test]
    fn doppler_towards_transducer() {
        // θ = 0 → maximum shift
        let df = doppler_frequency_shift(1.0, 0.0, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM);
        assert!((df - 2.0 * MHZ_TO_HZ / SOUND_SPEED_WATER_SIM).abs() < 1e-6);
    }

    #[test]
    fn doppler_perpendicular_is_zero() {
        let df = doppler_frequency_shift(1.0, PI / 2.0, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM);
        assert!(df.abs() < 1e-10);
    }

    #[test]
    fn compounding_narrower_than_single() {
        let lam = SOUND_SPEED_WATER_SIM / (2.0 * MHZ_TO_HZ);
        let psf1 = lateral_psf_sinc2(&[0.5e-3], 2.0, lam);
        let psf4 = pw_compounding_lateral_psf(&[0.5e-3], 4, 2.0, lam);
        // 4-angle compounding narrows the PSF (FWHM radius 1.33mm → 0.665mm).
        // At x=0.5mm the compound PSF is past its −6 dB point while the
        // single-angle PSF is still within its mainlobe → psf4 ≪ psf1.
        // sinc²: u1≈0.376 → 0.613, u4≈0.752 → 0.088.
        assert!(psf4[0] < psf1[0], "psf4={} psf1={}", psf4[0], psf1[0]);
    }

    #[test]
    fn lateral_resolution_positive() {
        let lam = SOUND_SPEED_WATER_SIM / (3.5 * MHZ_TO_HZ);
        let dx = lateral_resolution_m(2.0, lam);
        assert!(dx > 0.0 && dx < 1e-3);
    }
}
