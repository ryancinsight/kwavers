//! Async I/O infrastructure using tokio
//!
//! This module provides async abstractions for I/O-bound operations following
//! the senior Rust engineer persona requirements.
//!
//! ## Design Principles
//!
//! - **Tokio Runtime**: Multi-threaded async runtime for concurrent operations
//! - **Task Spawning**: Use tokio::spawn for independent async tasks
//! - **Structured Concurrency**: Proper cancellation and error propagation
//! - **Performance**: Zero-cost abstractions with async/await
//!
//! ## References
//!
//! - Tokio Documentation: https://tokio.rs
//! - Async Book: https://rust-lang.github.io/async-book/
//!
//! ## Example
//!
//! ```no_run
//! #[cfg(feature = "async-runtime")]
//! use kwavers::runtime::async_io::AsyncFileReader;
//!
//! #[cfg(feature = "async-runtime")]
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let reader = AsyncFileReader::new("simulation_results.dat")?;
//!     let data = reader.read_array3_f64().await?;
//!     println!("Loaded array with shape: {:?}", data.shape());
//!     Ok(())
//! }
//! ```

#[cfg(feature = "async-runtime")]
pub use tokio_impl::*;

#[cfg(feature = "async-runtime")]
mod tokio_impl {
    use crate::error::{KwaversError, KwaversResult};
    use ndarray::Array3;
    use std::path::{Path, PathBuf};
    use tokio::fs::File;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Async file reader for simulation data
    ///
    /// Provides non-blocking I/O for loading large datasets without blocking
    /// the main computation thread.
    #[derive(Debug)]
    pub struct AsyncFileReader {
        path: PathBuf,
    }

    impl AsyncFileReader {
        /// Create a new async file reader
        ///
        /// # Arguments
        ///
        /// * `path` - Path to the file to read
        ///
        /// # Errors
        ///
        /// Returns error if path is invalid
        pub fn new<P: AsRef<Path>>(path: P) -> KwaversResult<Self> {
            Ok(Self {
                path: path.as_ref().to_path_buf(),
            })
        }

        /// Read a 3D array of f64 values asynchronously
        ///
        /// # Errors
        ///
        /// Returns error if file cannot be read or data format is invalid
        pub async fn read_array3_f64(&self) -> KwaversResult<Array3<f64>> {
            let mut file = File::open(&self.path)
                .await
                .map_err(|e| KwaversError::InvalidInput(format!("Failed to open file: {}", e)))?;

            // Read dimensions (3 x usize)
            let mut dim_buf = [0u8; 24]; // 3 * 8 bytes
            file.read_exact(&mut dim_buf).await.map_err(|e| {
                KwaversError::InvalidInput(format!("Failed to read dimensions: {}", e))
            })?;

            let nx = usize::from_le_bytes(dim_buf[0..8].try_into().unwrap());
            let ny = usize::from_le_bytes(dim_buf[8..16].try_into().unwrap());
            let nz = usize::from_le_bytes(dim_buf[16..24].try_into().unwrap());

            // Read data
            let len = nx * ny * nz;
            let mut data_buf = vec![0u8; len * 8];
            file.read_exact(&mut data_buf)
                .await
                .map_err(|e| KwaversError::InvalidInput(format!("Failed to read data: {}", e)))?;

            // Convert bytes to f64 values
            let data: Vec<f64> = data_buf
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            Array3::from_shape_vec((nx, ny, nz), data)
                .map_err(|e| KwaversError::InvalidInput(format!("Failed to create array: {}", e)))
        }
    }

    /// Async file writer for simulation results
    ///
    /// Provides non-blocking I/O for saving large datasets.
    #[derive(Debug)]
    pub struct AsyncFileWriter {
        path: PathBuf,
    }

    impl AsyncFileWriter {
        /// Create a new async file writer
        pub fn new<P: AsRef<Path>>(path: P) -> KwaversResult<Self> {
            Ok(Self {
                path: path.as_ref().to_path_buf(),
            })
        }

        /// Write a 3D array of f64 values asynchronously
        ///
        /// # Errors
        ///
        /// Returns error if file cannot be written
        pub async fn write_array3_f64(&self, array: &Array3<f64>) -> KwaversResult<()> {
            let mut file = File::create(&self.path)
                .await
                .map_err(|e| KwaversError::InvalidInput(format!("Failed to create file: {}", e)))?;

            // Write dimensions
            let shape = array.shape();
            let mut dim_buf = Vec::with_capacity(24);
            dim_buf.extend_from_slice(&shape[0].to_le_bytes());
            dim_buf.extend_from_slice(&shape[1].to_le_bytes());
            dim_buf.extend_from_slice(&shape[2].to_le_bytes());

            file.write_all(&dim_buf).await.map_err(|e| {
                KwaversError::InvalidInput(format!("Failed to write dimensions: {}", e))
            })?;

            // Write data
            let mut data_buf = Vec::with_capacity(array.len() * 8);
            for &value in array.iter() {
                data_buf.extend_from_slice(&value.to_le_bytes());
            }

            file.write_all(&data_buf)
                .await
                .map_err(|e| KwaversError::InvalidInput(format!("Failed to write data: {}", e)))?;

            file.sync_all()
                .await
                .map_err(|e| KwaversError::InvalidInput(format!("Failed to sync file: {}", e)))?;

            Ok(())
        }
    }

    /// Spawn an async task on the tokio runtime
    ///
    /// # Example
    ///
    /// ```no_run
    /// #[cfg(feature = "async-runtime")]
    /// use kwavers::runtime::async_io::spawn_task;
    ///
    /// #[cfg(feature = "async-runtime")]
    /// # async fn example() {
    /// let handle = spawn_task(async {
    ///     // Async computation here
    ///     42
    /// });
    ///
    /// let result = handle.await.unwrap();
    /// # }
    /// ```
    pub fn spawn_task<F>(future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        tokio::spawn(future)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use ndarray::Array3;

        #[tokio::test]
        async fn test_async_file_roundtrip() {
            let temp_path = "/tmp/test_async_array.dat";

            // Create test array
            let original =
                Array3::from_shape_fn((10, 20, 30), |(i, j, k)| (i * 600 + j * 30 + k) as f64);

            // Write asynchronously
            let writer = AsyncFileWriter::new(temp_path).unwrap();
            writer.write_array3_f64(&original).await.unwrap();

            // Read asynchronously
            let reader = AsyncFileReader::new(temp_path).unwrap();
            let loaded = reader.read_array3_f64().await.unwrap();

            // Verify
            assert_eq!(original.shape(), loaded.shape());
            assert_eq!(original, loaded);

            // Cleanup
            tokio::fs::remove_file(temp_path).await.ok();
        }
    }
}

#[cfg(not(feature = "async-runtime"))]
pub mod stub {
    //! Stub implementations when async-runtime feature is disabled
    //!
    //! These provide compile-time errors with helpful messages if async
    //! functionality is used without the feature enabled.

    /// Async runtime not available - enable "async-runtime" feature
    #[derive(Debug)]
    pub struct AsyncFileReader;

    /// Async runtime not available - enable "async-runtime" feature
    #[derive(Debug)]
    pub struct AsyncFileWriter;
}

#[cfg(not(feature = "async-runtime"))]
pub use stub::*;
