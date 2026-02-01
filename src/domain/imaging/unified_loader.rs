//! Unified Medical Image Loading Interface
//!
//! This module provides a unified, polymorphic interface for loading and managing
//! multiple medical imaging formats (CT NIFTI, DICOM) through a consistent abstraction.
//!
//! ## Design Pattern: Strategy Pattern
//!
//! The unified loader uses the Strategy pattern to encapsulate different loading algorithms:
//! ```text
//! UnifiedMedicalImageLoader
//!   ├── Strategy 1: CT NIFTI Loader
//!   ├── Strategy 2: DICOM Loader
//!   └── Strategy N: Future formats
//! ```
//!
//! This allows clients to:
//! - Load different formats transparently
//! - Switch formats without changing application code
//! - Validate format compatibility upfront
//!
//! ## Workflow
//!
//! ```text
//! File Path → Format Detection → Loader Selection → Load & Validate → Unified Metadata
//! ```
//!
//! ## Example Usage
//!
//! ```no_run
//! # use kwavers::domain::imaging::unified_loader::UnifiedMedicalImageLoader;
//! # use kwavers::core::error::KwaversResult;
//! // Load any supported format transparently
//! let mut loader = UnifiedMedicalImageLoader::from_path("patient_scan.nii.gz")?;
//! let image_data = loader.load()?;
//! let metadata = loader.metadata();
//!
//! println!("Loaded {} image: {:?}", metadata.modality, metadata.dimensions);
//! # Ok::<(), kwavers::core::error::KwaversError>(())
//! ```

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::imaging::medical::{
    create_loader, CTImageLoader, DicomImageLoader, MedicalImageLoader, MedicalImageMetadata,
};
use ndarray::Array3;
use std::path::Path;

/// Unified medical image loader supporting multiple formats
///
/// This loader provides a single interface for all supported medical imaging formats.
/// It automatically detects format from file extension and selects appropriate loader.
///
/// # Supported Formats
///
/// - NIFTI (`.nii`, `.nii.gz`) - CT scans
/// - DICOM (`.dcm`, `.dicom`) - Multi-modality medical images
///
/// # Example
///
/// ```no_run
/// # use kwavers::domain::imaging::unified_loader::UnifiedMedicalImageLoader;
/// # use kwavers::core::error::KwaversResult;
/// let mut loader = UnifiedMedicalImageLoader::from_path("scan.nii.gz")?;
/// let data = loader.load()?;
/// assert_eq!(loader.metadata().modality, "CT");
/// # Ok::<(), kwavers::core::error::KwaversError>(())
/// ```
pub struct UnifiedMedicalImageLoader {
    /// Underlying strategy (loader implementation)
    strategy: Box<dyn MedicalImageLoader>,
    /// File path that was loaded
    file_path: String,
}

impl std::fmt::Debug for UnifiedMedicalImageLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedMedicalImageLoader")
            .field("file_path", &self.file_path)
            .field("strategy", &self.strategy.modality())
            .finish()
    }
}

impl UnifiedMedicalImageLoader {
    /// Create loader from file path (auto-detect format)
    ///
    /// # Arguments
    ///
    /// * `path` - File path to medical image
    ///
    /// # Returns
    ///
    /// Unified loader configured for the detected format
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use kwavers::domain::imaging::unified_loader::UnifiedMedicalImageLoader;
    /// let loader = UnifiedMedicalImageLoader::from_path("ct.nii.gz")?;
    /// # Ok::<(), kwavers::core::error::KwaversError>(())
    /// ```
    pub fn from_path(path: &str) -> KwaversResult<Self> {
        // Validate path exists
        if !Path::new(path).exists() {
            return Err(KwaversError::InvalidInput(format!(
                "Medical image file not found: {}",
                path
            )));
        }

        // Create appropriate loader based on extension
        let strategy = create_loader(path)?;

        Ok(Self {
            strategy,
            file_path: path.to_string(),
        })
    }

    /// Create CT NIFTI loader
    pub fn ct_loader() -> Self {
        Self {
            strategy: Box::new(CTImageLoader::new()),
            file_path: String::new(),
        }
    }

    /// Create DICOM loader
    pub fn dicom_loader() -> Self {
        Self {
            strategy: Box::new(DicomImageLoader::new()),
            file_path: String::new(),
        }
    }

    /// Load medical image from file
    ///
    /// # Returns
    ///
    /// 3D image array with shape (nx, ny, nz)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use kwavers::domain::imaging::unified_loader::UnifiedMedicalImageLoader;
    /// let mut loader = UnifiedMedicalImageLoader::from_path("scan.dcm")?;
    /// let image = loader.load()?;
    /// println!("Image shape: {:?}", image.dim());
    /// # Ok::<(), kwavers::core::error::KwaversError>(())
    /// ```
    pub fn load(&mut self) -> KwaversResult<Array3<f64>> {
        if self.file_path.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No file path set. Use from_path() or set_path()".to_string(),
            ));
        }

        self.strategy.load(&self.file_path)
    }

    /// Load from custom path
    ///
    /// Useful when the loader was created with ct_loader() or dicom_loader()
    /// and you want to load from a different path without recreating.
    pub fn load_from(&mut self, path: &str) -> KwaversResult<Array3<f64>> {
        if !Path::new(path).exists() {
            return Err(KwaversError::InvalidInput(format!(
                "Medical image file not found: {}",
                path
            )));
        }

        self.file_path = path.to_string();
        self.strategy.load(path)
    }

    /// Get loaded image metadata
    pub fn metadata(&self) -> MedicalImageMetadata {
        self.strategy.metadata()
    }

    /// Get image name/identifier
    pub fn name(&self) -> &str {
        self.strategy.name()
    }

    /// Get image modality
    pub fn modality(&self) -> &str {
        self.strategy.modality()
    }

    /// Get current file path
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    /// Check if image has been loaded
    pub fn is_loaded(&self) -> bool {
        !self.file_path.is_empty()
    }
}

/// Medical image batch loader for processing multiple files
///
/// Useful for loading entire datasets or patient cohorts.
///
/// # Example
///
/// ```no_run
/// # use kwavers::domain::imaging::unified_loader::MedicalImageBatchLoader;
/// # use kwavers::core::error::KwaversResult;
/// let mut batch = MedicalImageBatchLoader::new();
/// batch.add("patient1.nii.gz")?;
/// batch.add("patient2.dcm")?;
/// batch.load_all()?;
/// # Ok::<(), kwavers::core::error::KwaversError>(())
/// ```
#[derive(Debug)]
pub struct MedicalImageBatchLoader {
    /// Paths queued for loading
    paths: Vec<String>,
    /// Loaded images (path -> data)
    images: Vec<(String, Array3<f64>)>,
    /// Metadata per image
    metadata: Vec<MedicalImageMetadata>,
}

impl MedicalImageBatchLoader {
    /// Create new batch loader
    pub fn new() -> Self {
        Self {
            paths: Vec::new(),
            images: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Add file to batch queue
    pub fn add(&mut self, path: &str) -> KwaversResult<()> {
        if !Path::new(path).exists() {
            return Err(KwaversError::InvalidInput(format!(
                "Medical image file not found: {}",
                path
            )));
        }

        self.paths.push(path.to_string());
        Ok(())
    }

    /// Load all queued files
    ///
    /// Stops at first error (fails fast).
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
    pub fn queued_count(&self) -> usize {
        self.paths.len()
    }

    /// Get number of loaded images
    pub fn loaded_count(&self) -> usize {
        self.images.len()
    }

    /// Get loaded image by index
    pub fn get_image(&self, index: usize) -> Option<&Array3<f64>> {
        self.images.get(index).map(|(_, data)| data)
    }

    /// Get metadata by index
    pub fn get_metadata(&self, index: usize) -> Option<&MedicalImageMetadata> {
        self.metadata.get(index)
    }

    /// Get all loaded image paths
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_loader_creation_ct() {
        let _loader = UnifiedMedicalImageLoader::ct_loader();
        // Should not panic
    }

    #[test]
    fn test_unified_loader_creation_dicom() {
        let _loader = UnifiedMedicalImageLoader::dicom_loader();
        // Should not panic
    }

    #[test]
    fn test_unified_loader_invalid_path() {
        let result = UnifiedMedicalImageLoader::from_path("nonexistent.nii.gz");
        assert!(result.is_err());
    }

    #[test]
    fn test_unified_loader_unsupported_format() {
        // Create temp file with unsupported extension
        let result = UnifiedMedicalImageLoader::from_path("test.xyz");
        assert!(result.is_err());
    }

    #[test]
    fn test_unified_loader_is_loaded() {
        let loader_ct = UnifiedMedicalImageLoader::ct_loader();
        assert!(!loader_ct.is_loaded());

        let loader_dicom = UnifiedMedicalImageLoader::dicom_loader();
        assert!(!loader_dicom.is_loaded());
    }

    #[test]
    fn test_batch_loader_new() {
        let batch = MedicalImageBatchLoader::new();
        assert_eq!(batch.queued_count(), 0);
        assert_eq!(batch.loaded_count(), 0);
    }

    #[test]
    fn test_batch_loader_add_invalid() {
        let mut batch = MedicalImageBatchLoader::new();
        let result = batch.add("nonexistent.nii");
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_loader_clear() {
        let mut batch = MedicalImageBatchLoader::new();
        batch.paths.push("test.nii".to_string());
        assert_eq!(batch.queued_count(), 1);

        batch.clear();
        assert_eq!(batch.queued_count(), 0);
    }

    #[test]
    fn test_batch_loader_default() {
        let batch = MedicalImageBatchLoader::default();
        assert_eq!(batch.queued_count(), 0);
    }

    #[test]
    fn test_batch_loader_get_nonexistent() {
        let batch = MedicalImageBatchLoader::new();
        assert!(batch.get_image(0).is_none());
        assert!(batch.get_metadata(0).is_none());
    }
}
