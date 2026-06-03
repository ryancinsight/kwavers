//! PyO3 bindings for `kwavers_physics::analytical::transducer`.

use kwavers_physics::analytical::transducer;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the circular-piston far-field directivity pattern.
///
/// D(theta) = 2 * J1(ka*sin(theta)) / (ka*sin(theta))
///
/// Args:
///     theta_rad: Observation angles [rad].
///     ka: Wave-number × radius product.
///
/// Returns:
///     Directivity array (normalised to unity on-axis).
#[pyfunction]
#[pyo3(signature = (theta_rad, ka))]
pub fn circular_piston_directivity(
    py: Python<'_>,
    theta_rad: PyReadonlyArray1<f64>,
    ka: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_slice = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::circular_piston_directivity(t_slice, ka);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the linear-array factor as a function of angle.
///
/// Args:
///     theta_rad: Observation angles [rad].
///     k: Wave number [rad/m].
///     d_m: Element pitch [m].
///     n: Number of elements.
///     steer_rad: Electronic steering angle [rad].
///
/// Returns:
///     Normalised array factor.
#[pyfunction]
#[pyo3(signature = (theta_rad, k, d_m, n, steer_rad))]
pub fn linear_array_factor(
    py: Python<'_>,
    theta_rad: PyReadonlyArray1<f64>,
    k: f64,
    d_m: f64,
    n: usize,
    steer_rad: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_slice = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::linear_array_factor(t_slice, k, d_m, n, steer_rad);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute grating-lobe angles for a uniform linear array.
///
/// Args:
///     k: Wave number [rad/m].
///     d_m: Element pitch [m].
///     steer_rad: Steering angle [rad].
///
/// Returns:
///     Array of grating-lobe angles [rad] (may be empty if none exist).
#[pyfunction]
#[pyo3(signature = (k, d_m, steer_rad))]
pub fn grating_lobe_angles(
    py: Python<'_>,
    k: f64,
    d_m: f64,
    steer_rad: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = transducer::grating_lobe_angles(k, d_m, steer_rad);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute element apodization weights for a given window type.
///
/// Supported window types: "uniform", "hann", "hamming", "blackman",
/// "nuttall", "tukey".
///
/// Args:
///     n: Number of elements.
///     window_type: Name of the window function.
///
/// Returns:
///     Weight array of length *n*.
#[pyfunction]
#[pyo3(signature = (n, window_type))]
pub fn apodization_weights(
    py: Python<'_>,
    n: usize,
    window_type: String,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = transducer::apodization_weights(n, &window_type);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute time-delay laws for 2-D geometric focusing.
///
/// Args:
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     x_f: Focal point x-coordinate [m].
///     z_f: Focal point z-coordinate [m].
///     c: Sound speed [m/s].
///
/// Returns:
///     Delay array [s], same length as *elem_x*.
#[pyfunction]
#[pyo3(signature = (elem_x, elem_z, x_f, z_f, c))]
pub fn delay_law_focus_2d(
    py: Python<'_>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    x_f: f64,
    z_f: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::delay_law_focus_2d(ex, ez, x_f, z_f, c);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the complex 2-D beam pattern for a phased array.
///
/// Returns a list [real_field, imag_field], each a 2-D ndarray of shape
/// (len(x_arr), len(z_arr)).
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     freq_hz: Frequency [Hz].
///     c: Sound speed [m/s].
///     weights: Apodization weights per element.
///     delays: Delay per element [s].
///
/// Returns:
///     [real_nx_nz, imag_nx_nz] — list of two 2-D arrays.
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, elem_x, elem_z, freq_hz, c, weights, delays))]
pub fn beam_pattern_2d(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    freq_hz: f64,
    c: f64,
    weights: PyReadonlyArray1<f64>,
    delays: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let w_s = weights
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let d_s = delays
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let (real_flat, imag_flat) =
        transducer::beam_pattern_2d(x_s, z_s, ex, ez, freq_hz, c, w_s, d_s);
    let real_arr = Array2::from_shape_vec((nx, nz), real_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let imag_arr = Array2::from_shape_vec((nx, nz), imag_flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        real_arr.into_pyarray(py).unbind(),
        imag_arr.into_pyarray(py).unbind(),
    ))
}

/// Compute the on-axis pressure of a circular piston transducer.
///
/// Args:
///     z_arr: Axial positions [m].
///     radius_m: Piston radius [m].
///     freq_hz: Frequency [Hz].
///     p0_pa: Surface pressure amplitude [Pa].
///     c: Sound speed [m/s].
///
/// Returns:
///     On-axis pressure magnitude [Pa].
#[pyfunction]
#[pyo3(signature = (z_arr, radius_m, freq_hz, p0_pa, c))]
pub fn circular_piston_onaxis(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    radius_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::circular_piston_onaxis(z_s, radius_m, freq_hz, p0_pa, c);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the on-axis pressure of a focused-bowl (spherically focused) transducer.
///
/// Args:
///     z_arr: Axial positions [m].
///     bowl_radius_m: Bowl aperture radius [m].
///     focal_length_m: Geometric focal length [m].
///     freq_hz: Frequency [Hz].
///     p0_pa: Source pressure [Pa].
///     c: Sound speed [m/s].
///
/// Returns:
///     On-axis pressure magnitude [Pa].
#[pyfunction]
#[pyo3(signature = (z_arr, bowl_radius_m, focal_length_m, freq_hz, p0_pa, c))]
pub fn focused_bowl_onaxis(
    py: Python<'_>,
    z_arr: PyReadonlyArray1<f64>,
    bowl_radius_m: f64,
    focal_length_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        transducer::focused_bowl_onaxis(z_s, bowl_radius_m, focal_length_m, freq_hz, p0_pa, c);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute band-limited interpolation (BLI) stencil weights.
///
/// Args:
///     delta: Sub-sample offsets (1-D array).
///     n_stencil: Number of stencil points.
///
/// Returns:
///     2-D array of shape (len(delta), n_stencil).
#[pyfunction]
#[pyo3(signature = (delta, n_stencil))]
pub fn bli_stencil_weights(
    py: Python<'_>,
    delta: PyReadonlyArray1<f64>,
    n_stencil: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let d_s = delta
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rows: Vec<Vec<f64>> = transducer::bli_stencil_weights(d_s, n_stencil);
    let n_delta = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let arr2d = Array2::from_shape_vec((n_delta, n_stencil), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr2d.into_pyarray(py).unbind())
}

/// Generate element positions for a centred linear array.
///
/// Element i is placed at x = (i − (N−1)/2)·pitch, z = 0.
///
/// Args:
///     n: Number of elements.
///     pitch_m: Inter-element pitch [m].
///
/// Returns:
///     [elem_x, elem_z] — two 1-D arrays of element coordinates [m].
#[pyfunction]
#[pyo3(signature = (n, pitch_m))]
pub fn linear_array_positions(
    py: Python<'_>,
    n: usize,
    pitch_m: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (x, z) = transducer::linear_array_positions(n, pitch_m);
    Ok((x.into_pyarray(py).unbind(), z.into_pyarray(py).unbind()))
}

/// Fresnel near-field transition distance — the natural focus of an aperture.
///
/// N = D² / (4λ), λ = c/f. Deepest range at which an unfocused aperture
/// naturally concentrates energy; electronic focusing is effective only for
/// z ≲ N.
///
/// Args:
///     aperture_m: Full aperture width D [m].
///     freq_hz: Frequency [Hz].
///     c: Sound speed [m/s].
///
/// Returns:
///     Natural-focus range N [m].
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
///     focal_range_m: Focal range R [m].
///     steer_rad: Steering angle from the array normal [rad].
///
/// Returns:
///     (x_f, z_f) — focal point [m].
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
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     focal_range_m: Focal range R [m].
///     steer_rad: Steering angle from the array normal [rad].
///     c: Sound speed [m/s].
///
/// Returns:
///     Delay array [s], same length as elem_x.
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
    Ok(result.into_pyarray(py).unbind())
}

/// Full far-field beam-pattern magnitude via the pattern-multiplication theorem.
///
/// |P(θ)| = |D(θ)| · |AF(θ)|, normalised to unit peak across the angle set.
///
/// Args:
///     theta_rad: Observation angles [rad].
///     k: Wave number [rad/m].
///     d_m: Element pitch [m].
///     n: Number of elements.
///     steer_rad: Steering angle [rad].
///     ka_elem: Element directivity parameter k·a_elem.
///
/// Returns:
///     Normalised pattern magnitude (linear), peak = 1.
#[pyfunction]
#[pyo3(signature = (theta_rad, k, d_m, n, steer_rad, ka_elem))]
pub fn beam_pattern_magnitude(
    py: Python<'_>,
    theta_rad: PyReadonlyArray1<f64>,
    k: f64,
    d_m: f64,
    n: usize,
    steer_rad: f64,
    ka_elem: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_slice = theta_rad
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = transducer::beam_pattern_magnitude(t_slice, k, d_m, n, steer_rad, ka_elem);
    Ok(result.into_pyarray(py).unbind())
}

/// Magnitude of the 2-D complex beam pattern, normalised to its peak.
///
/// Returns |p(x, z)| / max|p| as a 2-D ndarray of shape (len(x_arr), len(z_arr)).
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     freq_hz: Frequency [Hz].
///     c: Sound speed [m/s].
///     weights: Apodization weights per element.
///     delays: Delay per element [s].
///
/// Returns:
///     2-D magnitude field in [0, 1].
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, elem_x, elem_z, freq_hz, c, weights, delays))]
#[allow(clippy::too_many_arguments)]
pub fn beam_pattern_2d_magnitude(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    freq_hz: f64,
    c: f64,
    weights: PyReadonlyArray1<f64>,
    delays: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let w_s = weights
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let d_s = delays
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let flat = transducer::beam_pattern_2d_magnitude(x_s, z_s, ex, ez, freq_hz, c, w_s, d_s);
    let arr = Array2::from_shape_vec((nx, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).unbind())
}

/// Per-element geometric focusing delay laws for multiple focal sub-spots.
///
/// For each sub-spot j and element i the straight-ray (homogeneous-medium)
/// focusing delay is τ_ij = (max_i r_ij − r_ij) / c, so every element arrives
/// in phase at sub-spot j. The model for multi-spot histotripsy and
/// multi-target BBB opening.
///
/// Args:
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     spot_x: Sub-spot x-positions [m].
///     spot_z: Sub-spot z-positions [m].
///     c: Sound speed [m/s].
///
/// Returns:
///     2-D delay array of shape (n_spots, n_elem) [s]; row j focuses the full
///     aperture on sub-spot j.
#[pyfunction]
#[pyo3(signature = (elem_x, elem_z, spot_x, spot_z, c))]
pub fn multi_focus_delay_laws_2d(
    py: Python<'_>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    spot_x: PyReadonlyArray1<f64>,
    spot_z: PyReadonlyArray1<f64>,
    c: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let sx = spot_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let sz = spot_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n_spots = sx.len();
    let n_elem = ex.len();
    let flat = transducer::multi_focus_delay_laws_2d(ex, ez, sx, sz, c);
    let arr = Array2::from_shape_vec((n_spots, n_elem), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).unbind())
}

/// Simultaneous multi-focus CW field magnitude via phase-conjugation synthesis.
///
/// Each element is driven with w_i = Σ_j a_j·exp(+i·k·r_ij), the coherent
/// superposition of the phase-conjugate (time-reversed) fields that focus on
/// each sub-spot. Returns |p(x, z)| / max|p|, the field for parallel
/// multi-spot histotripsy / multi-target BBB opening.
///
/// Args:
///     x_arr: Lateral positions [m].
///     z_arr: Axial positions [m].
///     elem_x: Element x-positions [m].
///     elem_z: Element z-positions [m].
///     spot_x: Sub-spot x-positions [m].
///     spot_z: Sub-spot z-positions [m].
///     spot_amp: Per-spot drive amplitudes (length == n_spots).
///     freq_hz: Frequency [Hz].
///     c: Sound speed [m/s].
///
/// Returns:
///     2-D magnitude field of shape (len(x_arr), len(z_arr)) in [0, 1].
#[pyfunction]
#[pyo3(signature = (x_arr, z_arr, elem_x, elem_z, spot_x, spot_z, spot_amp, freq_hz, c))]
#[allow(clippy::too_many_arguments)]
pub fn multi_focus_field_magnitude_2d(
    py: Python<'_>,
    x_arr: PyReadonlyArray1<f64>,
    z_arr: PyReadonlyArray1<f64>,
    elem_x: PyReadonlyArray1<f64>,
    elem_z: PyReadonlyArray1<f64>,
    spot_x: PyReadonlyArray1<f64>,
    spot_z: PyReadonlyArray1<f64>,
    spot_amp: PyReadonlyArray1<f64>,
    freq_hz: f64,
    c: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_s = x_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let z_s = z_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ex = elem_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let ez = elem_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let sx = spot_x
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let sz = spot_z
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let sa = spot_amp
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let nx = x_s.len();
    let nz = z_s.len();
    let flat = transducer::multi_focus_field_magnitude_2d(x_s, z_s, ex, ez, sx, sz, sa, freq_hz, c);
    let arr = Array2::from_shape_vec((nx, nz), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).unbind())
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
///     aperture_m: Full aperture width [m].
///     jitter_frac: Dither amplitude as a fraction of the element pitch
///         (0 reproduces the uniform layout; ~1.0 suppresses grating lobes).
///
/// Returns:
///     Element x-positions [m], length n, centred at the origin.
#[pyfunction]
#[pyo3(signature = (n, aperture_m, jitter_frac))]
pub fn linear_array_aperiodic_positions(
    py: Python<'_>,
    n: usize,
    aperture_m: f64,
    jitter_frac: f64,
) -> Py<PyArray1<f64>> {
    let x = transducer::linear_array_aperiodic_positions(n, aperture_m, jitter_frac);
    x.into_pyarray(py).unbind()
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
///     elem_x: Element x-positions [m].
///     obs_theta: Observation angles [rad], from broadside.
///     k: Wavenumber 2*pi*f/c [rad/m].
///     steer_theta: Steering angle [rad], from broadside.
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
    Ok(result.into_pyarray(py).unbind())
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
///     elem_x: Element x-positions [m].
///     steer_theta: Steering-angle grid [rad].
///     obs_theta: Observation-angle grid [rad] for the lobe search.
///     k: Wavenumber 2*pi*f/c [rad/m].
///     ka_elem: Element directivity parameter k*a_elem.
///     mainlobe_halfwidth_rad: Half-width of the main-lobe exclusion window [rad].
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
    Ok(result.into_pyarray(py).unbind())
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
///     steer_theta: Steering-angle grid [rad] (monotonically increasing).
///     glr: Grating-lobe ratio at each steer_theta.
///     threshold: Grating-lobe safety threshold (e.g. 0.5).
///
/// Returns:
///     Safe steering half-angle [rad].
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
///     dr_lat_m: Lateral steering offset from the mechanical focus [m].
///     dr_ax_m: Axial steering offset from the mechanical focus [m].
///     f0_hz: Drive frequency [Hz].
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
