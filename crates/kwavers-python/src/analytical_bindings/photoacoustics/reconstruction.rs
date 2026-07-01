//! Axial-resolution and spectroscopic-unmixing photoacoustic bindings.

use kwavers_physics::analytical::photoacoustics;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compute the photoacoustic axial resolution.
///
/// δz ≈ 0.88 * c / bandwidth
///
/// Args:
///     bandwidth_hz: Transducer bandwidth [Hz].
///     c: Sound speed [m/s].
///
/// Returns:
///     Axial resolution [m].
#[pyfunction]
#[pyo3(signature = (bandwidth_hz, c))]
pub fn pa_axial_resolution(bandwidth_hz: f64, c: f64) -> PyResult<f64> {
    Ok(photoacoustics::pa_axial_resolution(bandwidth_hz, c))
}

/// Estimate chromophore concentrations by least-squares spectral unmixing.
///
/// Solves: spectra_matrix @ concentrations ≈ measurements
///
/// Args:
///     spectra_matrix: Absorption spectra matrix (n_wavelengths × n_chromophores).
///     measurements: Measured PA signals (length n_wavelengths).
///
/// Returns:
///     Concentration vector (length n_chromophores).
#[pyfunction]
#[pyo3(signature = (spectra_matrix, measurements))]
pub fn spectroscopic_unmixing_lstsq(
    py: Python<'_>,
    spectra_matrix: PyReadonlyArray2<f64>,
    measurements: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let sm = spectra_matrix.as_array();
    let (nrows, _ncols) = sm.dim();
    let spectra_vecs: Vec<Vec<f64>> = (0..nrows).map(|i| sm.row(i).to_vec()).collect();
    let meas_slice = measurements
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::spectroscopic_unmixing_lstsq(&spectra_vecs, meas_slice);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute deterministic spectroscopic-unmixing sO₂ curves.
#[pyfunction]
#[pyo3(signature = (wavelengths_nm, min_so2, max_so2, n_samples, perturbations))]
pub fn spectroscopic_unmixing_so2_sweep<'py>(
    py: Python<'py>,
    wavelengths_nm: PyReadonlyArray1<f64>,
    min_so2: f64,
    max_so2: f64,
    n_samples: usize,
    perturbations: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let wavelengths = wavelengths_nm
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let perturbations = perturbations
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = photoacoustics::spectroscopic_unmixing_so2_sweep(
        wavelengths,
        min_so2,
        max_so2,
        n_samples,
        perturbations,
    )
    .map_err(PyValueError::new_err)?;

    let estimates = PyArray2::from_vec2(py, &result.estimated_so2_by_perturbation)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let out = PyDict::new(py);
    out.set_item("true_so2", result.true_so2.into_pyarray(py))?;
    out.set_item("estimated_so2_by_perturbation", estimates)?;
    Ok(out)
}
