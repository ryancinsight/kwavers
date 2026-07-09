//! PyO3 wrappers for aperture geometry and 3-D pressure helpers.

use kwavers_physics::analytical::transducer;
use leto::Array2;
use numpy::{ToPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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
    Ok((x.to_pyarray(py).unbind(), z.to_pyarray(py).unbind()))
}

/// Generate focused spherical-bowl element coordinates for a beam axis aligned
/// to +x. Returns an `(n_elem, 3)` array.
#[pyfunction]
#[pyo3(signature = (n_rings, elements_per_ring, aperture_radius_m, roc_m, focal_length_m))]
pub fn focused_bowl_element_positions_3d(
    py: Python<'_>,
    n_rings: usize,
    elements_per_ring: usize,
    aperture_radius_m: f64,
    roc_m: f64,
    focal_length_m: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let flat = transducer::focused_bowl_element_positions_3d(
        n_rings,
        elements_per_ring,
        aperture_radius_m,
        roc_m,
        focal_length_m,
    );
    let rows = flat.len() / 3;
    let arr = Array2::from_shape_vec((rows, 3), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.to_pyarray(py).unbind())
}

/// 3-D focusing delay law for arbitrary element and focus positions.
#[pyfunction]
#[pyo3(signature = (elem_pos, focus_xyz, c))]
pub fn delay_law_focus_3d(
    py: Python<'_>,
    elem_pos: PyReadonlyArray2<f64>,
    focus_xyz: PyReadonlyArray1<f64>,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let ep = elem_pos.as_array();
    let f = focus_xyz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if ep.ncols() != 3 || f.len() != 3 {
        return Err(PyRuntimeError::new_err(
            "elem_pos must have shape (n, 3) and focus_xyz length 3",
        ));
    }
    let flat: Vec<f64> = ep.iter().copied().collect();
    let result = py.detach(|| transducer::delay_law_focus_3d(&flat, [f[0], f[1], f[2]], c));
    Ok(result.to_pyarray(py).unbind())
}

/// Steered 3-D aperture pressure magnitude at sample points.
#[pyfunction]
#[pyo3(signature = (points_xyz, elem_pos, weights, delays_s, focus_xyz, freq_hz, c, alpha_np_m, focus_pressure_pa))]
#[allow(clippy::too_many_arguments)]
pub fn steered_aperture_pressure_3d(
    py: Python<'_>,
    points_xyz: PyReadonlyArray2<f64>,
    elem_pos: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    delays_s: PyReadonlyArray1<f64>,
    focus_xyz: PyReadonlyArray1<f64>,
    freq_hz: f64,
    c: f64,
    alpha_np_m: f64,
    focus_pressure_pa: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let pts = points_xyz.as_array();
    let ep = elem_pos.as_array();
    let w = weights
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let d = delays_s
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let f = focus_xyz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if pts.ncols() != 3 || ep.ncols() != 3 || f.len() != 3 {
        return Err(PyRuntimeError::new_err(
            "points_xyz and elem_pos must have shape (n, 3); focus_xyz length 3",
        ));
    }
    let pts_flat: Vec<f64> = pts.iter().copied().collect();
    let ep_flat: Vec<f64> = ep.iter().copied().collect();
    let result = py.detach(|| {
        transducer::steered_aperture_pressure_3d(
            &pts_flat,
            &ep_flat,
            w,
            d,
            [f[0], f[1], f[2]],
            freq_hz,
            c,
            alpha_np_m,
            focus_pressure_pa,
        )
    });
    Ok(result.to_pyarray(py).unbind())
}

/// Focused-bowl steered transverse pressure profile assembled in Rust.
#[pyfunction]
#[pyo3(signature = (
    radius_m, focus_xyz, focus_pressure_pa, n_rings, elements_per_ring,
    aperture_radius_m, roc_m, focal_length_m, freq_hz, c, alpha_np_m
))]
#[allow(clippy::too_many_arguments)]
pub fn focused_bowl_steered_pressure_profile(
    py: Python<'_>,
    radius_m: PyReadonlyArray1<f64>,
    focus_xyz: PyReadonlyArray1<f64>,
    focus_pressure_pa: f64,
    n_rings: usize,
    elements_per_ring: usize,
    aperture_radius_m: f64,
    roc_m: f64,
    focal_length_m: f64,
    freq_hz: f64,
    c: f64,
    alpha_np_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let r = radius_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let f = focus_xyz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if f.len() != 3 {
        return Err(PyRuntimeError::new_err("focus_xyz must have length 3"));
    }
    let result = py.detach(|| {
        transducer::focused_bowl_steered_pressure_profile(
            r,
            [f[0], f[1], f[2]],
            focus_pressure_pa,
            n_rings,
            elements_per_ring,
            aperture_radius_m,
            roc_m,
            focal_length_m,
            freq_hz,
            c,
            alpha_np_m,
        )
    });
    Ok(result.to_pyarray(py).unbind())
}

