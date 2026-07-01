//! PyO3 wrappers for optoacoustic focused-ultrasound formulas.

use kwavers_physics::analytical::transducer;
use pyo3::prelude::*;

/// Numerical aperture of an optoacoustic focused emitter from its geometry.
///
/// NA = sin(arctan((D_t/2) / r)) for a spherical cap of curvature radius `r`
/// and transverse aperture diameter `D_t` (book Ch34, Li et al. 2022).
///
/// Args:
///     radius_m: Radius of curvature r [m].
///     transverse_diameter_m: Transverse aperture diameter D_t [m].
///
/// Returns:
///     Numerical aperture NA (dimensionless).
#[pyfunction]
#[pyo3(signature = (radius_m, transverse_diameter_m))]
pub fn numerical_aperture_from_geometry(radius_m: f64, transverse_diameter_m: f64) -> f64 {
    transducer::numerical_aperture_from_geometry(radius_m, transverse_diameter_m)
}

/// f-number `f_N = 1/(2·NA)` from the numerical aperture (book Ch34).
///
/// Args:
///     na: Numerical aperture (dimensionless).
///
/// Returns:
///     f-number f_N (dimensionless).
#[pyfunction]
#[pyo3(signature = (na,))]
pub fn f_number_from_na(na: f64) -> f64 {
    transducer::f_number_from_na(na)
}

/// SOAP focal pressure gain `G = (2πf/c0)·r·(1 − √(1 − 1/(4 f_N²)))` (book
/// Eq. 34.4, Li et al. 2022). Diffraction focal gain of a spherical-cap
/// source — source-mechanism-agnostic (O'Neil 1949).
///
/// Args:
///     freq_hz: Acoustic centre frequency f [Hz].
///     c0: Ambient sound speed c0 [m/s].
///     radius_m: Radius of curvature r [m].
///     f_number: f-number f_N (dimensionless).
///
/// Returns:
///     Focal pressure gain G (dimensionless).
#[pyfunction]
#[pyo3(signature = (freq_hz, c0, radius_m, f_number))]
pub fn soap_focal_gain(freq_hz: f64, c0: f64, radius_m: f64, f_number: f64) -> f64 {
    transducer::soap_focal_gain(freq_hz, c0, radius_m, f_number)
}

/// Acoustic-resolution lateral focal width `R_L = 0.71·ν/(NA·f)` (book
/// Eq. 34.5, Li et al. 2022), where `ν` is the sound speed.
///
/// Args:
///     sound_speed: Sound speed ν [m/s].
///     na: Numerical aperture (dimensionless).
///     freq_hz: Acoustic centre frequency f [Hz].
///
/// Returns:
///     Lateral resolution R_L [m].
#[pyfunction]
#[pyo3(signature = (sound_speed, na, freq_hz))]
pub fn acoustic_resolution_lateral(sound_speed: f64, na: f64, freq_hz: f64) -> f64 {
    transducer::acoustic_resolution_lateral(sound_speed, na, freq_hz)
}
