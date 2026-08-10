//! MATLAB Level-5 numeric-volume parsing via Consus MAT provider.

use consus_mat::{MatArray, MatNumericArray, MatNumericClass};
use kwavers_core::error::{KwaversError, KwaversResult};
use std::path::Path;

#[derive(Clone, Debug)]
pub(super) struct Mat5NumericVolume {
    pub dims: [usize; 3],
    pub values: Vec<f64>,
    pub name: String,
}

pub(super) fn read_mat5_numeric_volume(
    path: &Path,
    requested_name: &str,
) -> KwaversResult<Mat5NumericVolume> {
    let bytes = std::fs::read(path)?;
    let mat = consus_mat::loadmat_bytes(&bytes).map_err(|err| {
        KwaversError::InvalidInput(format!(
            "MAT5 file `{}` failed Consus MAT decode: {err}",
            path.display()
        ))
    })?;

    for (name, array) in mat.variables {
        if name != requested_name {
            continue;
        }

        let MatArray::Numeric(numeric) = array else {
            return Err(KwaversError::InvalidInput(format!(
                "MAT5 variable '{name}' is not numeric"
            )));
        };
        if numeric.is_complex() {
            return Err(KwaversError::InvalidInput(format!(
                "MAT5 variable '{name}' must be real-valued"
            )));
        }
        let dims = decode_dims_3d(&name, &numeric)?;
        let values = decode_numeric_values(&name, &numeric)?;
        let expected = dims.iter().product::<usize>();
        if values.len() != expected {
            return Err(KwaversError::DimensionMismatch(format!(
                "MAT5 variable '{name}' decoded {} elements for dims {:?}",
                values.len(),
                dims
            )));
        }

        return Ok(Mat5NumericVolume { dims, values, name });
    }

    Err(KwaversError::InvalidInput(format!(
        "MAT5 variable '{requested_name}' was not found as a real numeric 3-D array"
    )))
}

fn decode_dims_3d(name: &str, numeric: &MatNumericArray) -> KwaversResult<[usize; 3]> {
    if numeric.shape.len() != 3 {
        return Err(KwaversError::InvalidInput(format!(
            "MAT5 variable '{name}' must be rank 3, got dimensions {:?}",
            numeric.shape
        )));
    }
    let dims = [numeric.shape[0], numeric.shape[1], numeric.shape[2]];
    if dims.iter().any(|&d| d == 0) {
        return Err(KwaversError::InvalidInput(format!(
            "MAT5 variable '{name}' has a zero-sized dimension: {:?}",
            dims
        )));
    }
    Ok(dims)
}

fn decode_numeric_values(name: &str, numeric: &MatNumericArray) -> KwaversResult<Vec<f64>> {
    match numeric.class {
        MatNumericClass::Uint8 => Ok(numeric.real_data.iter().copied().map(f64::from).collect()),
        MatNumericClass::Int8 => Ok(numeric
            .real_data
            .iter()
            .copied()
            .map(|v| f64::from(v as i8))
            .collect()),
        MatNumericClass::Uint16 => decode_chunks(name, &numeric.real_data, 2, |c| {
            f64::from(u16::from_le_bytes([c[0], c[1]]))
        }),
        MatNumericClass::Int16 => decode_chunks(name, &numeric.real_data, 2, |c| {
            f64::from(i16::from_le_bytes([c[0], c[1]]))
        }),
        MatNumericClass::Uint32 => decode_chunks(name, &numeric.real_data, 4, |c| {
            f64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        }),
        MatNumericClass::Int32 => decode_chunks(name, &numeric.real_data, 4, |c| {
            f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        }),
        MatNumericClass::Single => decode_chunks(name, &numeric.real_data, 4, |c| {
            f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        }),
        MatNumericClass::Double => decode_chunks(name, &numeric.real_data, 8, |c| {
            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
        }),
        MatNumericClass::Uint64 => decode_chunks(name, &numeric.real_data, 8, |c| {
            u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64
        }),
        MatNumericClass::Int64 => decode_chunks(name, &numeric.real_data, 8, |c| {
            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64
        }),
    }
}

fn decode_chunks<F>(name: &str, payload: &[u8], width: usize, decode: F) -> KwaversResult<Vec<f64>>
where
    F: Fn(&[u8]) -> f64,
{
    if !payload.len().is_multiple_of(width) {
        return Err(KwaversError::InvalidInput(format!(
            "MAT5 variable '{name}' payload width mismatch: {} bytes not divisible by {width}",
            payload.len()
        )));
    }
    Ok(payload.chunks_exact(width).map(decode).collect())
}
