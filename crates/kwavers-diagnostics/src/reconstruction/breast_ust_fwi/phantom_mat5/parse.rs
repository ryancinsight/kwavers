//! Select the MRI variable and convert its normalized samples for interpolation.

use consus_mat::{v5::read_mat_v5_variable, MatArray};
use kwavers_core::error::{KwaversError, KwaversResult};
use std::path::Path;

pub(super) struct Mat5NumericVolume {
    pub dims: [usize; 3],
    pub values: Vec<f64>,
}

pub(super) fn read_mat5_numeric_volume(
    path: &Path,
    requested_name: &str,
) -> KwaversResult<Mat5NumericVolume> {
    let bytes = std::fs::read(path)?;
    let array = read_mat_v5_variable(&bytes, requested_name).map_err(|error| {
        KwaversError::InvalidInput(format!("MAT5 variable '{requested_name}': {error}"))
    })?;
    let Some(MatArray::Numeric(array)) = array else {
        return Err(KwaversError::InvalidInput(format!(
            "MAT5 variable '{requested_name}' was not found as a real numeric 3-D array"
        )));
    };
    let dims: [usize; 3] = array.shape.as_slice().try_into().map_err(|_| {
        KwaversError::InvalidInput(format!(
            "MAT5 variable '{requested_name}' must be rank 3, got dimensions {:?}",
            array.shape
        ))
    })?;
    if array.is_complex() || dims.contains(&0) {
        return Err(KwaversError::InvalidInput(format!(
            "MAT5 variable '{requested_name}' must be real with positive dimensions"
        )));
    }

    // MRI interpolation operates on f64 intensities. Decode once from the
    // source scalar type without changing MATLAB column-major storage order.
    let values =
        consus_core::decode_to_f64(&array.real_data, &array.class.datatype()).map_err(|error| {
            KwaversError::InvalidInput(format!("MAT5 variable '{requested_name}': {error}"))
        })?;
    Ok(Mat5NumericVolume { dims, values })
}
