use crate::Grid;
use numpy::PyArray3;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyfunction]
fn make_disc(
    py: Python<'_>,
    grid: &Grid,
    center: (f64, f64, f64),
    radius: f64,
) -> PyResult<Py<PyArray3<bool>>> {
    let center_arr = [center.0, center.1, center.2];
    let mask = kwavers_math::geometry::make_disc(
        (grid.inner.nx, grid.inner.ny, grid.inner.nz),
        (grid.inner.dx, grid.inner.dy, grid.inner.dz),
        center_arr,
        radius,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
    Ok(PyArray3::from_owned_array(py, mask).into())
}

#[pyfunction]
fn make_ball(
    py: Python<'_>,
    grid: &Grid,
    center: (f64, f64, f64),
    radius: f64,
) -> PyResult<Py<PyArray3<bool>>> {
    let center_arr = [center.0, center.1, center.2];
    let mask = kwavers_math::geometry::make_ball(
        (grid.inner.nx, grid.inner.ny, grid.inner.nz),
        (grid.inner.dx, grid.inner.dy, grid.inner.dz),
        center_arr,
        radius,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
    Ok(PyArray3::from_owned_array(py, mask).into())
}

#[pyfunction]
fn make_sphere(
    py: Python<'_>,
    grid: &Grid,
    center: (f64, f64, f64),
    radius: f64,
) -> PyResult<Py<PyArray3<bool>>> {
    make_ball(py, grid, center, radius)
}

#[pyfunction]
#[pyo3(signature = (grid, center, radius, thickness=1))]
fn make_circle(
    py: Python<'_>,
    grid: &Grid,
    center: (f64, f64, f64),
    radius: f64,
    thickness: usize,
) -> PyResult<Py<PyArray3<bool>>> {
    let center_arr = [center.0, center.1, center.2];
    let mask = kwavers_math::geometry::make_circle(
        (grid.inner.nx, grid.inner.ny, grid.inner.nz),
        (grid.inner.dx, grid.inner.dy, grid.inner.dz),
        center_arr,
        radius,
        thickness,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
    Ok(PyArray3::from_owned_array(py, mask).into())
}

#[pyfunction]
fn make_line(
    py: Python<'_>,
    grid: &Grid,
    start: (f64, f64, f64),
    end: (f64, f64, f64),
) -> PyResult<Py<PyArray3<bool>>> {
    let start_arr = [start.0, start.1, start.2];
    let end_arr = [end.0, end.1, end.2];
    let mask = kwavers_math::geometry::make_line(
        (grid.inner.nx, grid.inner.ny, grid.inner.nz),
        (grid.inner.dx, grid.inner.dy, grid.inner.dz),
        start_arr,
        end_arr,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
    Ok(PyArray3::from_owned_array(py, mask).into())
}

pub(super) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(make_disc, m)?)?;
    m.add_function(wrap_pyfunction!(make_ball, m)?)?;
    m.add_function(wrap_pyfunction!(make_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(make_circle, m)?)?;
    m.add_function(wrap_pyfunction!(make_line, m)?)?;
    Ok(())
}
