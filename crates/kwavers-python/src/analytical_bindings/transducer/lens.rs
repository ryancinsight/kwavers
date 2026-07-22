//! PyO3 wrappers for static acoustic-lens helpers.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Static acoustic-lens focusing delay profile τ(r) across the aperture.
///
/// A silicone refractive lens designed for `focal_length_m` imposes the same
/// focusing delay as the phased-array delay law: τ(r) = (√(F²+r²) − F)/c_medium.
///
/// Args:
///     radii_m: Aperture radii `m`.
///     focal_length_m: Design focal length F `m`.
///     aperture_m: Lens aperture (full width) `m`.
///     medium_sound_speed: Medium sound speed c_medium [m/s].
///
/// Returns:
///     Focusing delay τ(r) `s` at each radius (0 at centre, monotone increasing).
#[pyfunction]
#[pyo3(signature = (radii_m, focal_length_m, aperture_m, medium_sound_speed))]
pub fn acoustic_lens_delay_profile(
    py: Python<'_>,
    radii_m: PyReadonlyArray1<f64>,
    focal_length_m: f64,
    aperture_m: f64,
    medium_sound_speed: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    use kwavers_transducer::transducers::physics::materials::AcousticLens;
    let radii = radii_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let lens = AcousticLens::silicone(focal_length_m, aperture_m);
    let tau = lens.aperture_delay_profile(radii, medium_sound_speed);
    Ok(PyArray1::from_vec(py, tau).unbind())
}

/// Fresnel zone-plate boundary radii within the aperture.
///
/// r_n = √(n·λ·F + (n·λ/2)²) for n = 1, 2, … while r_n ≤ aperture_radius_m.
///
/// Args:
///     focal_length_m: Primary focal length F `m`.
///     wavelength_m: Design wavelength λ = c/f `m`.
///     aperture_radius_m: Outer aperture radius `m`.
///
/// Returns:
///     Zone boundary radii `m`, increasing.
#[pyfunction]
#[pyo3(signature = (focal_length_m, wavelength_m, aperture_radius_m))]
pub fn fresnel_zone_radii(
    py: Python<'_>,
    focal_length_m: f64,
    wavelength_m: f64,
    aperture_radius_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    use kwavers_transducer::transducers::physics::materials::FresnelZonePlate;
    let zp = FresnelZonePlate::new(focal_length_m, wavelength_m, aperture_radius_m);
    Ok(PyArray1::from_vec(py, zp.zone_radii()).unbind())
}

/// Isoplanatic mechanical-steering pose curve for a single-element corrective
/// lens (Maimbourg 2020, Eq. 2): θ_y = arcsin(x/F), T_z = F − √(F²−x²).
///
/// Args:
///     x_offsets_m: Transverse focus offsets `m`.
///     focal_length_m: Transducer focal length F `m`.
///
/// Returns:
///     (theta_y_rad, t_z_m) arrays; NaN where |x| > F (unphysical).
#[pyfunction]
#[pyo3(signature = (x_offsets_m, focal_length_m))]
pub fn isoplanatic_steering_curve(
    py: Python<'_>,
    x_offsets_m: PyReadonlyArray1<f64>,
    focal_length_m: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    use kwavers_transducer::transducers::physics::materials::isoplanatic_steering_pose;
    let xs = x_offsets_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let mut thetas = Vec::with_capacity(xs.len());
    let mut tzs = Vec::with_capacity(xs.len());
    for &x in xs {
        match isoplanatic_steering_pose(x, focal_length_m) {
            Some((th, tz)) => {
                thetas.push(th);
                tzs.push(tz);
            }
            None => {
                thetas.push(f64::NAN);
                tzs.push(f64::NAN);
            }
        }
    }
    Ok((
        PyArray1::from_vec(py, thetas).unbind(),
        PyArray1::from_vec(py, tzs).unbind(),
    ))
}

/// Corrective-lens thickness from a per-point aberration phase (Maimbourg 2020,
/// Eq. 1): p(M) = φ̃/(2πf₀)·1/(1/c_water − 1/c_lens) + K.
///
/// Args:
///     phase_rad: Unwrapped correction phase φ̃ at each surface point `rad`.
///     frequency_hz: Drive frequency f₀ `Hz`.
///     c_water: Coupling-medium sound speed [m/s].
///     c_lens: Lens sound speed [m/s].
///     min_thickness_m: Minimal castable lens thickness K `m`.
///
/// Returns:
///     Lens thickness p(M) `m` at each point (min equals min_thickness_m).
#[pyfunction]
#[pyo3(signature = (phase_rad, frequency_hz, c_water, c_lens, min_thickness_m))]
pub fn corrective_lens_thickness(
    py: Python<'_>,
    phase_rad: PyReadonlyArray1<f64>,
    frequency_hz: f64,
    c_water: f64,
    c_lens: f64,
    min_thickness_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    use kwavers_transducer::transducers::physics::materials::corrective_lens_thickness as clt;
    let phase = phase_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let p = clt(phase, frequency_hz, c_water, c_lens, min_thickness_m);
    Ok(PyArray1::from_vec(py, p).unbind())
}
