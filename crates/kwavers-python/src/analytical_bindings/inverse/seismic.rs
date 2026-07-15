//! Seismic-style inverse imaging bindings.

use kwavers_grid::Grid;
use kwavers_physics::acoustics::imaging::seismic::{EikonalSolver, KirchhoffMigrator, Trace};
use leto::Array3;
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Compute a 2-D first-arrival eikonal traveltime field.
///
/// The Rust path owns the fast-sweeping solve used by Chapter 17 Figure 18.6;
/// Python passes validated figure data and plots the returned NumPy array.
#[pyfunction]
#[pyo3(signature = (sound_speed, spacing_m, source_row, source_col, iterations=4))]
pub fn eikonal_traveltime_2d(
    py: Python<'_>,
    sound_speed: PyReadonlyArray2<f64>,
    spacing_m: f64,
    source_row: usize,
    source_col: usize,
    iterations: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let speed = sound_speed.as_array();
    let (rows, cols) = speed.dim();
    let grid = Grid::new(rows, cols, 1, spacing_m, spacing_m, spacing_m)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let speed3 = Array3::from_shape_fn((rows, cols, 1), |[row, col, _]| speed[[row, col]]);
    let solver = EikonalSolver::from_sound_speed(&grid, &speed3)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .with_iterations(iterations);
    let travel3 = solver
        .solve((source_row, source_col, 0))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let travel2 = Array2::from_shape_fn((rows, cols), |(row, col)| travel3[[row, col, 0]]);
    Ok(travel2.to_pyarray(py).unbind())
}

/// Migrate synthetic point-scatterer traces with Rust eikonal and Kirchhoff kernels.
///
/// This is the Rust-owned numerical backend for Chapter 17 Figure 18.6(b).
/// Python supplies aperture and scatterer coordinates, then plots the returned
/// migrated image.
#[pyfunction]
#[pyo3(signature = (
    rows,
    cols,
    spacing_m,
    sound_speed_m_s,
    aperture_rows,
    aperture_cols,
    scatterer_rows,
    scatterer_cols,
    dt_s,
    sample_count,
    center_frequency_hz,
    iterations=4
))]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 boundary mirrors chapter figure parameters"
)]
pub fn kirchhoff_point_scatterer_image_2d(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    spacing_m: f64,
    sound_speed_m_s: f64,
    aperture_rows: PyReadonlyArray1<i64>,
    aperture_cols: PyReadonlyArray1<i64>,
    scatterer_rows: PyReadonlyArray1<i64>,
    scatterer_cols: PyReadonlyArray1<i64>,
    dt_s: f64,
    sample_count: usize,
    center_frequency_hz: f64,
    iterations: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let aperture = paired_indices(aperture_rows, aperture_cols, "aperture")?;
    let scatterers = paired_indices(scatterer_rows, scatterer_cols, "scatterer")?;
    if aperture.is_empty() {
        return Err(PyValueError::new_err(
            "aperture must contain at least one point",
        ));
    }
    if scatterers.is_empty() {
        return Err(PyValueError::new_err(
            "scatterers must contain at least one point",
        ));
    }
    if sample_count < 2 {
        return Err(PyValueError::new_err("sample_count must be at least 2"));
    }
    if center_frequency_hz <= 0.0 {
        return Err(PyValueError::new_err(
            "center_frequency_hz must be positive",
        ));
    }

    let grid = Grid::new(rows, cols, 1, spacing_m, spacing_m, spacing_m)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let speed = Array3::from_elem((rows, cols, 1), sound_speed_m_s);
    let solver = EikonalSolver::from_sound_speed(&grid, &speed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .with_iterations(iterations);
    let travel_tables: Vec<Array3<f64>> = aperture
        .iter()
        .map(|&(row, col)| solver.solve((row, col, 0)))
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut traces = Vec::with_capacity(aperture.len() * aperture.len());
    for source in 0..aperture.len() {
        for receiver in 0..aperture.len() {
            let mut samples = vec![0.0; sample_count];
            for &(row, col) in &scatterers {
                let two_way =
                    travel_tables[source][[row, col, 0]] + travel_tables[receiver][[row, col, 0]];
                add_ricker(&mut samples, two_way, dt_s, center_frequency_hz);
            }
            traces.push(Trace {
                source,
                receiver,
                samples,
            });
        }
    }

    let image3 = KirchhoffMigrator::new(dt_s)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .migrate(&traces, &travel_tables, &travel_tables)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let image2 = Array2::from_shape_fn((rows, cols), |(row, col)| image3[[row, col, 0]]);
    Ok(image2.to_pyarray(py).unbind())
}

fn paired_indices(
    rows: PyReadonlyArray1<'_, i64>,
    cols: PyReadonlyArray1<'_, i64>,
    label: &str,
) -> PyResult<Vec<(usize, usize)>> {
    let row_slice = rows
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let col_slice = cols
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if row_slice.len() != col_slice.len() {
        return Err(PyValueError::new_err(format!(
            "{label} row and column arrays must have equal length"
        )));
    }
    row_slice
        .iter()
        .copied()
        .zip(col_slice.iter().copied())
        .map(|(row, col)| {
            if row < 0 || col < 0 {
                return Err(PyValueError::new_err(format!(
                    "{label} indices must be non-negative"
                )));
            }
            Ok((row as usize, col as usize))
        })
        .collect()
}

fn add_ricker(samples: &mut [f64], arrival_s: f64, dt_s: f64, center_frequency_hz: f64) {
    for (index, sample) in samples.iter_mut().enumerate() {
        let tau = index as f64 * dt_s - arrival_s;
        let scaled = std::f64::consts::PI * center_frequency_hz * tau;
        let squared = scaled * scaled;
        *sample += (1.0 - 2.0 * squared) * (-squared).exp();
    }
}
