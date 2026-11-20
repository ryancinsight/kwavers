use super::axial_psf_rect;
use super::lateral_psf_sinc2;
use super::lateral_resolution_m;
use super::pw_compounding_lateral_psf;
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
