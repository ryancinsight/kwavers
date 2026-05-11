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
//! ## Workflow
//!
//! ```text
//! File Path → Format Detection → Loader Selection → Load & Validate → Unified Metadata
//! ```

mod batch;
#[cfg(test)]
mod tests;

pub use batch::MedicalImageBatchLoader;

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::imaging::medical::{
    create_loader, CTImageLoader, DicomImageLoader, MedicalImageLoader, MedicalImageMetadata,
};
use ndarray::Array3;
use std::path::Path;

/// Unified medical image loader supporting multiple formats.
///
/// Automatically detects format from file extension and selects the appropriate loader.
///
/// # Supported Formats
///
/// - NIFTI (`.nii`, `.nii.gz`) - CT scans
/// - DICOM (`.dcm`, `.dicom`) - Multi-modality medical images
pub struct UnifiedMedicalImageLoader {
    strategy: Box<dyn MedicalImageLoader>,
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
    /// Create loader from file path (auto-detect format).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if file not found or format unsupported.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn from_path(path: &str) -> KwaversResult<Self> {
        if !Path::new(path).exists() {
            return Err(KwaversError::InvalidInput(format!(
                "Medical image file not found: {}",
                path
            )));
        }
        let strategy = create_loader(path)?;
        Ok(Self { strategy, file_path: path.to_owned() })
    }

    /// Create CT NIFTI loader
    #[must_use]
    pub fn ct_loader() -> Self {
        Self {
            strategy: Box::new(CTImageLoader::new()),
            file_path: String::new(),
        }
    }

    /// Create DICOM loader
    #[must_use]
    pub fn dicom_loader() -> Self {
        Self {
            strategy: Box::new(DicomImageLoader::new()),
            file_path: String::new(),
        }
    }

    /// Load medical image from the configured file path.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if no file path has been set.
    ///
    pub fn load(&mut self) -> KwaversResult<Array3<f64>> {
        if self.file_path.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No file path set. Use from_path() or set_path()".to_owned(),
            ));
        }
        self.strategy.load(&self.file_path)
    }

    /// Load from a custom path (overrides internal path).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if file not found.
    ///
    pub fn load_from(&mut self, path: &str) -> KwaversResult<Array3<f64>> {
        if !Path::new(path).exists() {
            return Err(KwaversError::InvalidInput(format!(
                "Medical image file not found: {}",
                path
            )));
        }
        self.file_path = path.to_owned();
        self.strategy.load(path)
    }

    /// Get loaded image metadata
    #[must_use]
    pub fn metadata(&self) -> MedicalImageMetadata {
        self.strategy.metadata()
    }

    /// Get image name/identifier
    #[must_use]
    pub fn name(&self) -> &str {
        self.strategy.name()
    }

    /// Get image modality
    #[must_use]
    pub fn modality(&self) -> &str {
        self.strategy.modality()
    }

    /// Get current file path
    #[must_use]
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    /// Check if a file path has been set
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        !self.file_path.is_empty()
    }
}
