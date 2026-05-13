//! RITK-backed medical image loading shared by PyO3 bindings.

use burn::backend::NdArray as NdArrayBackend;
use ndarray::Array3;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::path::Path;

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
    let spacing = image.spacing().to_vec();
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
