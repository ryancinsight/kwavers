//! PyO3 wrappers for electronic steering and sparse-aperture helpers.

use kwavers_physics::analytical::transducer;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Fresnel near-field transition distance — the natural focus of an aperture.
///
/// N = D² / (4λ), λ = c/f. Deepest range at which an unfocused aperture
/// naturally concentrates energy; electronic focusing is effective only for
/// z ≲ N.
///
/// Args:
///     aperture_m: Full aperture width D `m`.
///     freq_hz: Frequency `Hz`.
///     c: Sound speed [m/s].
///
/// Returns:
///     Natural-focus range N `m`.
#[pyfunction]
#[pyo3(signature = (aperture_m, freq_hz, c))]
pub fn near_field_distance(aperture_m: f64, freq_hz: f64, c: f64) -> f64 {
    transducer::near_field_distance(aperture_m, freq_hz, c)
}

/// Cartesian focal point on the natural-focus arc for a steering angle.
///
/// x_f = R·sin θ, z_f = R·cos θ. Steering at fixed R traces the focus along
/// the arc of constant focal range.
///
/// Args:
///     focal_range_m: Focal range R `m`.
///     steer_rad: Steering angle from the array normal `rad`.
///
/// Returns:
///     (x_f, z_f) — focal point `m`.
#[pyfunction]
#[pyo3(signature = (focal_range_m, steer_rad))]
pub fn steering_focus_point(focal_range_m: f64, steer_rad: f64) -> (f64, f64) {
    transducer::steering_focus_point(focal_range_m, steer_rad)
}

/// Steering+focusing delay law for a focus on the natural-focus arc.
///
/// Places the focus at range R and angle θ on the natural-focus arc, then
/// returns per-element delays. Set focal_range_m to the natural focus
/// (`near_field_distance`) to focus around the natural focus.
///
/// Args:
///     elem_x: Element x-positions `m`.
///     elem_z: Element z-positions `m`.
///     focal_range_m: Focal range R `m`.
///     steer_rad: Steering angle from the array normal `rad`.
///     c: Sound speed [m/s].
///
/// Returns:
///     Delay array `s`, same length as elem_x.
#[pyfunction]
#[pyo3(signature = (elem_x, elem_z, focal_range_m, steer_rad, c))]
pub fn delay_law_steer_2d(
    py: Python<'_>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    focal_range_m: f64,
    steer_rad: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::delay_law_steer_2d(ex, ez, focal_range_m, steer_rad, c);
    Ok(result.to_pyarray(py).unbind())
}

/// Aperiodic ("sparse") linear element layout — same aperture and element
/// count as a uniform array, but the periodic grid is broken by a
/// deterministic low-discrepancy (golden-ratio) dither so a single coherent
/// grating lobe is scattered into a low pedestal. Endpoints are anchored at
/// +/-aperture/2 (identical aperture); interior elements are dithered by
/// jitter_frac of the element pitch.
///
/// Args:
///     n: Number of elements (matched to the uniform array).
///     aperture_m: Full aperture width `m`.
///     jitter_frac: Dither amplitude as a fraction of the element pitch
///         (0 reproduces the uniform layout; ~1.0 suppresses grating lobes).
///
/// Returns:
///     Element x-positions `m`, length n, centred at the origin.
#[pyfunction]
#[pyo3(signature = (n, aperture_m, jitter_frac))]
pub fn linear_array_aperiodic_positions(
    py: Python<'_>,
    n: usize,
    aperture_m: f64,
    jitter_frac: f64,
) -> Py<PyArray1<f64>> {
    let x = transducer::linear_array_aperiodic_positions(n, aperture_m, jitter_frac);
    x.to_pyarray(py).unbind()
}

/// Steered far-field beam pattern of a linear array.
///
/// The N elements lie along x at positions elem_x and radiate broadside; the
/// array is phased to steer its main lobe to steer_theta (from broadside). The
/// far-field response at observation angle theta is
/// P(theta) = D(theta) * |(1/N) sum_i exp[i*k*x_i*(sin theta - sin theta_s)]|,
/// where D is the baffled circular-piston element factor with parameter
/// ka_elem. P peaks at theta_s; coherent secondary peaks are grating lobes.
///
/// Args:
///     elem_x: Element x-positions `m`.
///     obs_theta: Observation angles `rad`, from broadside.
///     k: Wavenumber 2*pi*f/c [rad/m].
///     steer_theta: Steering angle `rad`, from broadside.
///     ka_elem: Element directivity parameter k*a_elem.
///
/// Returns:
///     Beam-pattern magnitude at each obs_theta.
#[pyfunction]
#[pyo3(signature = (elem_x, obs_theta, k, steer_theta, ka_elem))]
pub fn steered_beam_pattern_1d(
    py: Python<'_>,
    elem_x: PyReadonlyArray1<f64>,
    obs_theta: PyReadonlyArray1<f64>,
    k: f64,
    steer_theta: f64,
    ka_elem: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let obs = obs_theta
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::steered_beam_pattern_1d(ex, obs, k, steer_theta, ka_elem);
    Ok(result.to_pyarray(py).unbind())
}

/// Grating-lobe ratio versus steering angle — the basis of the steering
/// envelope at a fixed frequency.
///
/// For each steering angle theta_s the array is phased to that angle and its
/// beam pattern is searched for the strongest lobe outside the main lobe (a
/// +/-mainlobe_halfwidth_rad window about theta_s). The returned value
/// G(theta_s) = max P(theta) / P(theta_s) over the secondary region; the safe
/// steering envelope is {theta_s : G <= 0.5}. At fixed frequency a uniform
/// aperture raises a coherent grating lobe when steered (G jumps up), while an
/// aperiodic aperture keeps G low over a much wider range.
///
/// Args:
///     elem_x: Element x-positions `m`.
///     steer_theta: Steering-angle grid `rad`.
///     obs_theta: Observation-angle grid `rad` for the lobe search.
///     k: Wavenumber 2*pi*f/c [rad/m].
///     ka_elem: Element directivity parameter k*a_elem.
///     mainlobe_halfwidth_rad: Half-width of the main-lobe exclusion window `rad`.
///
/// Returns:
///     Grating-lobe ratio at each steer_theta.
#[pyfunction]
#[pyo3(signature = (elem_x, steer_theta, obs_theta, k, ka_elem, mainlobe_halfwidth_rad))]
pub fn steering_grating_lobe_ratio_1d(
    py: Python<'_>,
    elem_x: PyReadonlyArray1<f64>,
    steer_theta: PyReadonlyArray1<f64>,
    obs_theta: PyReadonlyArray1<f64>,
    k: f64,
    ka_elem: f64,
    mainlobe_halfwidth_rad: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let st = steer_theta
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let obs = obs_theta
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        transducer::steering_grating_lobe_ratio_1d(ex, st, obs, k, ka_elem, mainlobe_halfwidth_rad);
    Ok(result.to_pyarray(py).unbind())
}

/// Safe steering half-angle — the largest steering excursion from broadside
/// over which the grating-lobe ratio stays at or below a safety threshold.
///
/// Starting from the steering angle closest to broadside, the safe region is
/// expanded outward to both sides while G <= threshold; the returned value is
/// the symmetric half-angle min(theta_right, |theta_left|) of that contiguous
/// run. With threshold 0.5 this is the -6 dB grating-lobe-safe steering
/// half-angle; the ratio of an aperiodic to a uniform half-angle quantifies
/// the steering-envelope expansion from sparse activation.
///
/// Args:
///     steer_theta: Steering-angle grid `rad` (monotonically increasing).
///     glr: Grating-lobe ratio at each steer_theta.
///     threshold: Grating-lobe safety threshold (e.g. 0.5).
///
/// Returns:
///     Safe steering half-angle `rad`.
#[pyfunction]
#[pyo3(signature = (steer_theta, glr, threshold))]
pub fn safe_steering_halfangle(
    steer_theta: PyReadonlyArray1<f64>,
    glr: PyReadonlyArray1<f64>,
    threshold: f64,
) -> PyResult<f64> {
    let st = steer_theta
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let g = glr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(transducer::safe_steering_halfangle(st, g, threshold))
}

/// Electronic-steering off-focus efficiency for a focused phased array.
///
/// ε(Δ_lat, Δ_ax) = exp[-(Δ_lat/R_lat)^2 - (Δ_ax/R_ax)^2], with the 1/e
/// ranges scaling linearly with wavelength λ = c / f0. Models the focal-pressure
/// derating when the focus is steered electronically off the mechanical focus
/// (element directivity + projected-aperture loss + grating-lobe roll-off).
///
/// Args:
///     dr_lat_m: Lateral steering offset from the mechanical focus `m`.
///     dr_ax_m: Axial steering offset from the mechanical focus `m`.
///     f0_hz: Drive frequency `Hz`.
///     c_m_s: Medium sound speed [m/s].
///     apodized: Whether cos-theta apodization is applied (wider window).
///
/// Returns:
///     Efficiency in (0, 1]; 1.0 at zero offset.
///
/// Reference:
///     Pernot et al. (2003) Ultrasound Med. Biol. 29, 1525; Hand et al. (2009)
///     Med. Phys. 36, 2107.
#[pyfunction]
#[pyo3(signature = (dr_lat_m, dr_ax_m, f0_hz, c_m_s, apodized=true))]
pub fn electronic_steering_efficiency(
    dr_lat_m: f64,
    dr_ax_m: f64,
    f0_hz: f64,
    c_m_s: f64,
    apodized: bool,
) -> PyResult<f64> {
    Ok(transducer::electronic_steering_efficiency(
        dr_lat_m, dr_ax_m, f0_hz, c_m_s, apodized,
    ))
}
