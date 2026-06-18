//! RITK-backed medical image loading shared by PyO3 bindings.

use burn::backend::NdArray as NdArrayBackend;
use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::path::Path;

/// Load a NIfTI volume (CT, segmentation, …) via RITK (`ritk-io`) — the kwavers
/// medical-image I/O path, the supported alternative to `nibabel`. Returns the
/// volume as a `(x, y, z)` float64 array and voxel spacing `(dx, dy, dz)` in mm.
///
/// Args:
///     path: NIfTI file path (.nii / .nii.gz).
///
/// Returns:
///     (volume, (dx_mm, dy_mm, dz_mm)).
#[pyfunction]
#[pyo3(signature = (path,))]
pub fn load_ct_nifti(py: Python<'_>, path: &str) -> PyResult<(Py<PyArray3<f64>>, (f64, f64, f64))> {
    let (volume, spacing) = load_ritk_nifti(Path::new(path))?;
    Ok((
        volume.into_pyarray(py).unbind(),
        (spacing[0], spacing[1], spacing[2]),
    ))
}

pub fn load_ritk_nifti(path: &Path) -> PyResult<(Array3<f64>, [f64; 3])> {
    type Backend = NdArrayBackend<f32>;
    let device = Default::default();
    let image = ritk_io::read_nifti::<Backend, _>(path, &device).map_err(|err| {
        PyRuntimeError::new_err(format!(
            "RITK NIfTI load failed for '{}': {err}",
            path.display()
        ))
    })?;
    let [depth, rows, cols] = image.shape();
    let spacing = image.spacing().as_slice().to_vec();
    if spacing.len() != 3 {
        return Err(PyRuntimeError::new_err(format!(
            "RITK image spacing rank {} is not 3",
            spacing.len()
        )));
    }

    let tensor_data = image.data().clone().into_data();
    let values = tensor_data
        .as_slice::<f32>()
        .map_err(|err| PyRuntimeError::new_err(format!("RITK tensor is not f32: {err:?}")))?;
    if values.len() != depth * rows * cols {
        return Err(PyRuntimeError::new_err(format!(
            "RITK tensor length {} does not match shape [{depth}, {rows}, {cols}]",
            values.len()
        )));
    }

    let mut volume = Array3::<f64>::zeros((cols, rows, depth));
    for z in 0..depth {
        for y in 0..rows {
            for x in 0..cols {
                volume[[x, y, z]] = f64::from(values[z * rows * cols + y * cols + x]);
            }
        }
    }
    Ok((volume, [spacing[2], spacing[1], spacing[0]]))
}
