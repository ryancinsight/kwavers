//! Medical Imaging Module
//!
//! This module provides domain abstractions for loading and processing medical imaging formats
//! including CT (NIFTI) and DICOM images. It represents the **ubiquitous language** for medical
//! imaging workflows across diagnostic and therapeutic ultrasound applications.
//!
//! ## Ubiquitous Language
//!
//! - **Medical Image**: Raw imaging data with geometric and spatial metadata
//! - **Hounsfield Units (HU)**: Standard intensity values in CT scans (range: -1024 to +3071)
//! - **Voxel Spacing**: Physical dimensions of image voxels (dx, dy, dz) in meters
//! - **Affine Transform**: 4×4 matrix mapping voxel indices to physical coordinates
//! - **Image Modality**: CT, MRI, DICOM (DICOM can encode multiple modalities)
//!
//! ## Architecture
//!
//! This module follows **Domain-Driven Design** principles:
//! - Domain abstractions (traits) define **what** imaging systems do
//! - Physics and clinical layers implement these abstractions
//! - No implementation details leak into domain definitions
//!
//! ## Features
//!
//! - `ct_loader` - NIFTI CT image loading with HU validation
//! - `dicom_loader` - DICOM format support (when `dicom-rs` feature enabled)
//! - Unified `MedicalImageLoader` trait for polymorphic access
//! - Metadata extraction: dimensions, voxel spacing, modality, intensity ranges
//! - Material property conversion: HU → acoustic/elastic properties
//!
//! ## References
//!
//! - Marquet et al. (2009) "Non-invasive transcranial ultrasound therapy based on 3D CT scan"
//! - Aubry et al. (2003) "Experimental validation of numerical simulations of transcranial
//!   ultrasound waves"
//! - Cristescu et al. (2017) "Automated identification of skull bones from CT images"

pub mod ct_loader;
pub mod dicom_loader;

pub use ct_loader::{CTImageLoader, CTMetadata};
pub use dicom_loader::{DicomImageLoader, DicomMetadata, DicomModality};

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Metadata common to all medical imaging formats
#[derive(Debug, Clone)]
pub struct MedicalImageMetadata {
    /// Image dimensions (nx, ny, nz)
    pub dimensions: (usize, usize, usize),
    /// Voxel spacing in meters (dx, dy, dz)
    pub voxel_spacing_m: (f64, f64, f64),
    /// Voxel spacing in millimeters (dx, dy, dz)
    pub voxel_spacing_mm: (f64, f64, f64),
    /// Affine transformation matrix (4×4) mapping voxel to physical coordinates
    pub affine: [[f64; 4]; 4],
    /// Data type description (e.g., "Hounsfield Units (f64)")
    pub data_type: String,
    /// Min/max intensity values in the image
    pub intensity_range: (f64, f64),
    /// Imaging modality (CT, MRI, DICOM)
    pub modality: String,
}

/// Unified trait for medical image loaders
///
/// This trait provides a polymorphic interface for loading medical images in different formats.
/// Implementations must provide:
/// - File loading with validation
/// - Metadata extraction
/// - 3D volumetric data as Array3<f64>
///
/// # Example
///
/// ```no_run
/// # use kwavers::domain::imaging::medical::{MedicalImageLoader, CTImageLoader};
/// # use kwavers::core::error::KwaversResult;
/// let loader = CTImageLoader::new();
/// let image = loader.load("patient_ct.nii.gz")?;
/// let metadata = loader.metadata();
/// # Ok::<(), kwavers::core::error::KwaversError>(())
/// ```
pub trait MedicalImageLoader: Send + Sync {
    /// Load image from file
    fn load(&mut self, path: &str) -> KwaversResult<Array3<f64>>;

    /// Get metadata from loaded image
    fn metadata(&self) -> MedicalImageMetadata;

    /// Get image name/identifier
    fn name(&self) -> &str;

    /// Get modality type
    fn modality(&self) -> &str;
}

/// Factory function to create appropriate loader based on file extension
///
/// # Arguments
///
/// * `path` - File path to medical image
///
/// # Returns
///
/// Box containing appropriate loader for the file format
///
/// # Example
///
/// ```no_run
/// # use kwavers::domain::imaging::medical::create_loader;
/// let loader = create_loader("patient_ct.nii.gz")?;
/// # Ok::<(), kwavers::core::error::KwaversError>(())
/// ```
pub fn create_loader(path: &str) -> KwaversResult<Box<dyn MedicalImageLoader>> {
    if path.ends_with(".nii") || path.ends_with(".nii.gz") {
        Ok(Box::new(CTImageLoader::new()))
    } else if path.ends_with(".dcm") || path.ends_with(".dicom") {
        Ok(Box::new(DicomImageLoader::new()))
    } else {
        Err(crate::core::error::KwaversError::InvalidInput(format!(
            "Unsupported medical image format: {}. Supported: .nii, .nii.gz, .dcm, .dicom",
            path
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_medical_image_metadata_creation() {
        let metadata = MedicalImageMetadata {
            dimensions: (512, 512, 256),
            voxel_spacing_m: (0.5e-3, 0.5e-3, 1.0e-3),
            voxel_spacing_mm: (0.5, 0.5, 1.0),
            affine: [
                [0.5e-3, 0.0, 0.0, 0.0],
                [0.0, 0.5e-3, 0.0, 0.0],
                [0.0, 0.0, 1.0e-3, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            data_type: "Hounsfield Units (f64)".to_string(),
            intensity_range: (-1024.0, 2000.0),
            modality: "CT".to_string(),
        };

        assert_eq!(metadata.dimensions, (512, 512, 256));
        assert_eq!(metadata.modality, "CT");
    }

    #[test]
    fn test_loader_factory_nifti() {
        let result = create_loader("test.nii.gz");
        assert!(result.is_ok());
    }

    #[test]
    fn test_loader_factory_dicom() {
        let result = create_loader("test.dcm");
        assert!(result.is_ok());
    }

    #[test]
    fn test_loader_factory_invalid() {
        let result = create_loader("test.xyz");
        assert!(result.is_err());
    }
}
