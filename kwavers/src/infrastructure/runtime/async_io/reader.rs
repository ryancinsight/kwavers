use std::path::{Path, PathBuf};

use ndarray::Array3;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

use crate::core::error::{KwaversError, KwaversResult};

/// Async file reader for simulation data.
#[derive(Debug)]
pub struct AsyncFileReader {
    path: PathBuf,
}

impl AsyncFileReader {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new<P: AsRef<Path>>(path: P) -> KwaversResult<Self> {
        Ok(Self {
            path: path.as_ref().to_path_buf(),
        })
    }

    /// Read a 3D array of `f64` values.
    ///
    /// Reads the wire format written by [`AsyncFileWriter::write_array3`]:
    /// 24-byte little-endian dimension header followed by `f64` payload.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub async fn read_array3(&self) -> KwaversResult<Array3<f64>> {
        let mut file = File::open(&self.path).await.map_err(KwaversError::Io)?;

        let mut dim_buf = [0u8; 24];
        file.read_exact(&mut dim_buf)
            .await
            .map_err(KwaversError::Io)?;

        let nx = usize::from_le_bytes(dim_buf[0..8].try_into().unwrap());
        let ny = usize::from_le_bytes(dim_buf[8..16].try_into().unwrap());
        let nz = usize::from_le_bytes(dim_buf[16..24].try_into().unwrap());

        let cells = nx
            .checked_mul(ny)
            .and_then(|v| v.checked_mul(nz))
            .ok_or_else(|| KwaversError::ResourceLimitExceeded {
                message: format!("Array dimensions overflow allocation: ({nx}, {ny}, {nz})"),
            })?;
        let byte_len = cells
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| KwaversError::ResourceLimitExceeded {
                message: format!("Byte length overflow allocation for {cells} f64 values"),
            })?;

        let mut data_buf = vec![0u8; byte_len];
        file.read_exact(&mut data_buf)
            .await
            .map_err(KwaversError::Io)?;

        let data: Vec<f64> = data_buf
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Array3::from_shape_vec((nx, ny, nz), data).map_err(|error| {
            KwaversError::InvalidInput(format!("Invalid array shape in async reader: {error}"))
        })
    }
}
