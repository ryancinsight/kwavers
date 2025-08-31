//! File-based storage backend

use super::StorageBackend;
use crate::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

/// File storage backend
pub struct FileStorage {
    base_path: PathBuf,
    files: Vec<File>,
    shape: Option<(usize, usize, usize)>,
}

impl FileStorage {
    /// Create file storage
    pub fn create(base_path: PathBuf) -> Self {
        Self {
            base_path,
            files: Vec::new(),
            shape: None,
        }
    }
}

impl StorageBackend for FileStorage {
    fn initialize(&mut self, shape: (usize, usize, usize)) -> KwaversResult<()> {
        self.shape = Some(shape);
        Ok(())
    }

    fn store_field(&mut self, name: &str, field: &Array3<f64>, step: usize) -> KwaversResult<()> {
        let filename = self.base_path.join(format!("{}_{:06}.dat", name, step));
        let mut file = File::create(&filename).map_err(KwaversError::Io)?;

        // Write binary data
        for value in field.iter() {
            let bytes = value.to_le_bytes();
            file.write_all(&bytes).map_err(KwaversError::Io)?;
        }

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        // Write metadata
        let metadata_path = self.base_path.join("metadata.json");
        let metadata = serde_json::json!({
            "shape": self.shape,
            "format": "binary_f64_le"
        });

        let mut file = File::create(metadata_path).map_err(KwaversError::Io)?;
        file.write_all(metadata.to_string().as_bytes())
            .map_err(KwaversError::Io)?;

        Ok(())
    }
}
