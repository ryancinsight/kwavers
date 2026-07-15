use super::UnifiedMedicalImageLoader;
use crate::medical::MedicalImageMetadata;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;
use std::path::Path;

/// Medical image batch loader for processing multiple files
///
/// Useful for loading entire datasets or patient cohorts.
///
/// # Example
///
/// ```no_run
/// # use kwavers_imaging::unified_loader::MedicalImageBatchLoader;
/// # use kwavers_core::error::KwaversResult;
/// let mut batch = MedicalImageBatchLoader::new();
/// batch.add("patient1.nii.gz")?;
/// batch.add("patient2.dcm")?;
/// batch.load_all()?;
/// # Ok::<(), kwavers_core::error::KwaversError>(())
/// ```
#[derive(Debug)]
pub struct MedicalImageBatchLoader {
    /// Paths queued for loading
    pub(super) paths: Vec<String>,
    /// Loaded images (path -> data)
    images: Vec<(String, Array3<f64>)>,
    /// Metadata per image
    metadata: Vec<MedicalImageMetadata>,
}

impl MedicalImageBatchLoader {
    /// Create new batch loader
    #[must_use]
    pub fn new() -> Self {
        Self {
            paths: Vec::new(),
            images: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Add file to batch queue
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn add(&mut self, path: &str) -> KwaversResult<()> {
        if !Path::new(path).exists() {
            return Err(KwaversError::InvalidInput(format!(
                "Medical image file not found: {}",
                path
            )));
        }
        self.paths.push(path.to_owned());
        Ok(())
    }

    /// Load all queued files (fails fast on first error).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn load_all(&mut self) -> KwaversResult<()> {
        self.images.clear();
        self.metadata.clear();

        for path in &self.paths {
            let mut loader = UnifiedMedicalImageLoader::from_path(path)?;
            let data = loader.load()?;
            let meta = loader.metadata();
            self.images.push((path.clone(), data));
            self.metadata.push(meta);
        }

        Ok(())
    }

    /// Get number of queued files
    #[must_use]
    pub fn queued_count(&self) -> usize {
        self.paths.len()
    }

    /// Get number of loaded images
    #[must_use]
    pub fn loaded_count(&self) -> usize {
        self.images.len()
    }

    /// Get loaded image by index
    #[must_use]
    pub fn get_image(&self, index: usize) -> Option<&Array3<f64>> {
        self.images.get(index).map(|(_, data)| data)
    }

    /// Get metadata by index
    #[must_use]
    pub fn get_metadata(&self, index: usize) -> Option<&MedicalImageMetadata> {
        self.metadata.get(index)
    }

    /// Get all queued paths
    #[must_use]
    pub fn paths(&self) -> &[String] {
        &self.paths[..self.loaded_count()]
    }

    /// Clear all loaded images and metadata
    pub fn clear(&mut self) {
        self.paths.clear();
        self.images.clear();
        self.metadata.clear();
    }
}

impl Default for MedicalImageBatchLoader {
    fn default() -> Self {
        Self::new()
    }
}
