//! PyO3 wrappers for multi-focus transducer fields.

use kwavers_physics::analytical::transducer;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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
