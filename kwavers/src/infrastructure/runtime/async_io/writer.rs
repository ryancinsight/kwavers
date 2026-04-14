use std::path::{Path, PathBuf};

use ndarray::Array3;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

use crate::core::error::{KwaversError, KwaversResult};

/// Async file writer for simulation results.
#[derive(Debug)]
pub struct AsyncFileWriter {
    path: PathBuf,
}

impl AsyncFileWriter {
    pub fn new<P: AsRef<Path>>(path: P) -> KwaversResult<Self> {
        Ok(Self {
            path: path.as_ref().to_path_buf(),
        })
    }

    /// Write a 3D array of `f64` values.
    pub async fn write_array3_f64(&self, array: &Array3<f64>) -> KwaversResult<()> {
        let mut file = File::create(&self.path).await.map_err(KwaversError::Io)?;

        let shape = array.shape();
        let mut dim_buf = Vec::with_capacity(24);
        dim_buf.extend_from_slice(&shape[0].to_le_bytes());
        dim_buf.extend_from_slice(&shape[1].to_le_bytes());
        dim_buf.extend_from_slice(&shape[2].to_le_bytes());
        file.write_all(&dim_buf).await.map_err(KwaversError::Io)?;

        let byte_len = array
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| KwaversError::ResourceLimitExceeded {
                message: format!(
                    "Byte length overflow allocation for {} f64 values",
                    array.len()
                ),
            })?;
        let mut data_buf = Vec::with_capacity(byte_len);
        for &value in array {
            data_buf.extend_from_slice(&value.to_le_bytes());
        }

        file.write_all(&data_buf).await.map_err(KwaversError::Io)?;
        file.sync_all().await.map_err(KwaversError::Io)?;
        Ok(())
    }
}

/// Spawn an async task on the Tokio runtime.
pub fn spawn_task<F>(future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    tokio::spawn(future)
}
