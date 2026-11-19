use std::f64::consts::PI;

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
/// * `x_arr` – lateral offsets from beam axis `m`
/// * `f_number` – F-number = focal_length / aperture
/// * `wavelength_m` – acoustic wavelength λ `m`
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
/// * `z_arr` – axial offsets from focal plane `m`
/// * `c` – sound speed [m/s]
/// * `bandwidth_hz` – receiver −6 dB bandwidth `Hz`
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
/// * `x_arr` – lateral offsets `m`
/// * `n_angles` – number of compounding angles
/// * `f_number` – effective F-number for a single angle
/// * `wavelength_m` – acoustic wavelength `m`
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

/// −6 dB lateral resolution (Rayleigh criterion).
///
/// ```text
/// δx = 0.886 · F# · λ   `m`
/// ```
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §8.3.
#[must_use]
#[inline]
pub fn lateral_resolution_m(f_number: f64, wavelength_m: f64) -> f64 {
    0.886 * f_number * wavelength_m
}

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

#[cfg(test)]
mod tests;
