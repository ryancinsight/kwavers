//! CT Image Loader - NIFTI Format Support
//!
//! This module provides domain-level CT image loading functionality, refactored from the
//! physics layer to properly reside in the domain layer. CT images are fundamental domain
//! concepts used in both diagnostic and therapeutic ultrasound applications.
//!
//! ## Architecture Rationale
//!
//! CT image loading is a **domain concern** because:
//! - CT scans define material properties (skull bone density, heterogeneity)
//! - CT metadata (voxel spacing, affine) is essential domain information
//! - HU-to-property conversion is a domain rule, not a physics detail
//! - Multiple physics implementations (FEM, BEM, FDM) use CT data
//!
//! Therefore, CT loading belongs in the domain layer where all solvers can access it,
//! rather than in the physics layer where it's tied to specific implementations.
//!
//! ## Reference
//!
//! Marquet et al. (2009) "Non-invasive transcranial ultrasound therapy based on a 3D CT scan"

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use log::warn;
use crate::domain::imaging::medical::{MedicalImageLoader, MedicalImageMetadata};
use ndarray::Array3;

#[cfg(feature = "nifti")]
use nifti::{InMemNiftiObject, IntoNdArray, NiftiObject, NiftiVolume, ReaderOptions};

/// CT Image Loader - Loads NIFTI format CT scans
///
/// # Features
///
/// - Supports `.nii` and `.nii.gz` (gzip-compressed) NIFTI files
/// - Extracts Hounsfield Units (HU) with validation
/// - Provides voxel spacing and affine transformation matrix
/// - Validates HU range: -2000 to +4000 (extended range for various scanners)
///
/// # Example
///
/// ```no_run
/// # use kwavers::domain::imaging::medical::{CTImageLoader, MedicalImageLoader};
/// let mut loader = CTImageLoader::new();
/// let ct_data = loader.load("patient_ct.nii.gz")?;
/// let metadata = loader.metadata();
/// println!("CT dimensions: {:?}", metadata.dimensions);
/// # Ok::<(), kwavers::core::error::KwaversError>(())
/// ```
#[derive(Debug)]
pub struct CTImageLoader {
    /// Loaded CT data (Hounsfield Units)
    data: Option<Array3<f64>>,
    /// Metadata extracted from NIFTI header
    metadata: Option<CTMetadata>,
}

/// Metadata extracted from CT NIFTI header
#[derive(Debug, Clone)]
pub struct CTMetadata {
    /// Image dimensions (nx, ny, nz)
    pub dimensions: (usize, usize, usize),
    /// Voxel spacing in mm (dx, dy, dz)
    pub voxel_spacing_mm: (f64, f64, f64),
    /// Voxel spacing in meters (dx, dy, dz)
    pub voxel_spacing_m: (f64, f64, f64),
    /// Affine transformation matrix (4×4)
    pub affine: [[f64; 4]; 4],
    /// Data type description
    pub data_type: String,
    /// Min/Max HU values in volume
    pub hu_range: (f64, f64),
}

impl CTImageLoader {
    /// Create new CT image loader
    pub fn new() -> Self {
        Self {
            data: None,
            metadata: None,
        }
    }

    /// Get loaded CT data
    pub fn data(&self) -> Option<&Array3<f64>> {
        self.data.as_ref()
    }

    /// Get CT metadata
    pub fn ct_metadata(&self) -> Option<&CTMetadata> {
        self.metadata.as_ref()
    }
}

impl Default for CTImageLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl MedicalImageLoader for CTImageLoader {
    fn load(&mut self, path: &str) -> KwaversResult<Array3<f64>> {
        #[cfg(feature = "nifti")]
        {
            use std::path::Path;

            // Validate file exists
            if !Path::new(path).exists() {
                return Err(KwaversError::InvalidInput(format!(
                    "NIFTI file not found: {}",
                    path
                )));
            }

            // Load NIFTI file
            let nifti_obj = ReaderOptions::new().read_file(path).map_err(|e| {
                KwaversError::InvalidInput(format!("Failed to read NIFTI file: {}", e))
            })?;

            // Extract header information
            let header = nifti_obj.header();
            let dims = header.dim;

            // Validate this is a 3D volume
            if dims[0] != 3 {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "dimensions".to_string(),
                    value: dims[0].to_string(),
                    constraint: "must be 3D volume".to_string(),
                }));
            }

            let (nx, ny, nz) = (dims[1] as usize, dims[2] as usize, dims[3] as usize);

            // Get voxel spacing (convert from header units to meters)
            let pixdim = header.pixdim;
            let voxel_spacing_mm = (pixdim[1] as f64, pixdim[2] as f64, pixdim[3] as f64);
            let voxel_spacing_m = (
                voxel_spacing_mm.0 * 1e-3,
                voxel_spacing_mm.1 * 1e-3,
                voxel_spacing_mm.2 * 1e-3,
            );

            // Extract affine transformation matrix
            let affine = Self::extract_affine_matrix(&nifti_obj)?;

            // Load volume data
            let hounsfield = Self::extract_hounsfield_units(nifti_obj.volume(), nx, ny, nz)?;

            // Validate HU range
            let (min_hu, max_hu) = Self::compute_hu_range(&hounsfield);
            Self::validate_hu_range(min_hu, max_hu)?;

            // Store metadata and data
            self.metadata = Some(CTMetadata {
                dimensions: (nx, ny, nz),
                voxel_spacing_mm,
                voxel_spacing_m,
                affine,
                data_type: "Hounsfield Units (f64)".to_string(),
                hu_range: (min_hu, max_hu),
            });

            self.data = Some(hounsfield.clone());
            Ok(hounsfield)
        }

        #[cfg(not(feature = "nifti"))]
        {
            Err(KwaversError::InvalidInput(format!(
                "NIFTI support not enabled. Rebuild with --features nifti. Path: {}",
                path
            )))
        }
    }

    fn metadata(&self) -> MedicalImageMetadata {
        if let Some(ct_meta) = &self.metadata {
            MedicalImageMetadata {
                dimensions: ct_meta.dimensions,
                voxel_spacing_m: ct_meta.voxel_spacing_m,
                voxel_spacing_mm: ct_meta.voxel_spacing_mm,
                affine: ct_meta.affine,
                data_type: ct_meta.data_type.clone(),
                intensity_range: ct_meta.hu_range,
                modality: "CT".to_string(),
            }
        } else {
            // Default metadata for unloaded image
            MedicalImageMetadata {
                dimensions: (0, 0, 0),
                voxel_spacing_m: (1e-3, 1e-3, 1e-3),
                voxel_spacing_mm: (1.0, 1.0, 1.0),
                affine: Self::identity_affine(),
                data_type: "Hounsfield Units (f64)".to_string(),
                intensity_range: (0.0, 0.0),
                modality: "CT".to_string(),
            }
        }
    }

    fn name(&self) -> &str {
        "CT (NIFTI)"
    }

    fn modality(&self) -> &str {
        "CT"
    }
}

// ==================== Private Implementation ====================

impl CTImageLoader {
    #[cfg(feature = "nifti")]
    fn extract_affine_matrix(nifti_obj: &InMemNiftiObject) -> KwaversResult<[[f64; 4]; 4]> {
        let header = nifti_obj.header();

        // Try to get the sform (scanner anatomical coordinates) first
        if header.sform_code > 0 {
            let s = &header.srow_x;
            let affine = [
                [s[0] as f64, s[1] as f64, s[2] as f64, s[3] as f64],
                [
                    header.srow_y[0] as f64,
                    header.srow_y[1] as f64,
                    header.srow_y[2] as f64,
                    header.srow_y[3] as f64,
                ],
                [
                    header.srow_z[0] as f64,
                    header.srow_z[1] as f64,
                    header.srow_z[2] as f64,
                    header.srow_z[3] as f64,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ];
            return Ok(affine);
        }

        // Fallback to qform (aligned coordinates)
        if header.qform_code > 0 {
            // For simplicity, construct diagonal affine from pixdim
            let pixdim = header.pixdim;
            let affine = [
                [pixdim[1] as f64, 0.0, 0.0, 0.0],
                [0.0, pixdim[2] as f64, 0.0, 0.0],
                [0.0, 0.0, pixdim[3] as f64, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            return Ok(affine);
        }

        // Default: identity matrix with pixel dimensions
        Ok(Self::identity_affine())
    }

    #[cfg(feature = "nifti")]
    fn extract_hounsfield_units<V: NiftiVolume + IntoNdArray>(
        volume: V,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<Array3<f64>> {
        // Convert volume data to f64 HU values
        let raw_data = volume.into_ndarray::<f64>().map_err(|e| {
            KwaversError::InvalidInput(format!("Failed to convert NIFTI data: {}", e))
        })?;

        // Check dimensions match
        let data_shape = raw_data.shape();
        if data_shape.len() != 3
            || data_shape[0] != nx
            || data_shape[1] != ny
            || data_shape[2] != nz
        {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "dimensions".to_string(),
                value: format!("{:?}", data_shape),
                constraint: format!("must be 3D with shape ({},{},{})", nx, ny, nz),
            }));
        }

        // Convert to Array3<f64>
        let hounsfield = raw_data
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| {
                KwaversError::InvalidInput(format!("Failed to convert to 3D array: {}", e))
            })?;

        Ok(hounsfield)
    }

    /// Compute the minimum and maximum Hounsfield Unit (HU) values in the CT volume
    ///
    /// # Arguments
    /// * `hounsfield` - 3D array of HU values
    ///
    /// # Returns
    /// Tuple of (min_hu, max_hu)
    pub fn compute_hu_range(hounsfield: &Array3<f64>) -> (f64, f64) {
        let mut min_hu = f64::INFINITY;
        let mut max_hu = f64::NEG_INFINITY;

        for &val in hounsfield.iter() {
            if val < min_hu {
                min_hu = val;
            }
            if val > max_hu {
                max_hu = val;
            }
        }

        (min_hu, max_hu)
    }

    /// Validate that HU values are within physically reasonable range
    ///
    /// # Arguments
    /// * `min_hu` - Minimum HU value
    /// * `max_hu` - Maximum HU value
    ///
    /// # Returns
    /// Ok if valid, Error if out of range
    pub fn validate_hu_range(min_hu: f64, max_hu: f64) -> KwaversResult<()> {
        // Valid CT HU range: -1024 (air) to +3071 (dense bone)
        // Extended range tolerance for various scanner manufacturers
        const MIN_VALID_HU: f64 = -2000.0;
        const MAX_VALID_HU: f64 = 4000.0;

        if min_hu < MIN_VALID_HU || max_hu > MAX_VALID_HU {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "hounsfield_units".to_string(),
                value: format!("({:.0}, {:.0})", min_hu, max_hu),
                constraint: format!(
                    "must be within valid CT range ({:.0}, {:.0})",
                    MIN_VALID_HU, MAX_VALID_HU
                ),
            }));
        }

        // Warn if no bone is present (max_hu < 700)
        if max_hu < 700.0 {
            warn!(
                "No bone detected (max HU = {:.0}). Expected skull HU > 700.",
                max_hu
            );
        }

        Ok(())
    }

    fn identity_affine() -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Convert HU to sound speed using empirical relations
    ///
    /// Based on Aubry et al. (2003) "Experimental validation of numerical simulations
    /// of transcranial ultrasound waves"
    ///
    /// # Formula
    ///
    /// For bone (HU > 700):
    /// c(HU) = 2800 + (HU - 700) × 2.0 m/s
    ///
    /// For soft tissue (HU ≤ 700):
    /// c = 1500 m/s
    pub fn hu_to_sound_speed(hu: f64) -> f64 {
        if hu > 700.0 {
            // Bone: strong correlation with density
            2800.0 + (hu - 700.0) * 2.0
        } else {
            // Soft tissue: relatively uniform
            1500.0
        }
    }

    /// Convert HU to density using empirical relations
    ///
    /// # Formula
    ///
    /// For bone (HU > 700):
    /// ρ(HU) = 1700 + (HU - 700) × 0.2 kg/m³
    ///
    /// For soft tissue (HU ≤ 700):
    /// ρ = 1000 kg/m³
    pub fn hu_to_density(hu: f64) -> f64 {
        if hu > 700.0 {
            // Bone
            1700.0 + (hu - 700.0) * 0.2
        } else {
            // Soft tissue/water
            1000.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ct_loader_creation() {
        let loader = CTImageLoader::new();
        assert!(loader.data().is_none());
        assert!(loader.ct_metadata().is_none());
    }

    #[test]
    fn test_ct_loader_metadata_default() {
        let loader = CTImageLoader::new();
        let metadata = loader.metadata();
        assert_eq!(metadata.dimensions, (0, 0, 0));
        assert_eq!(metadata.modality, "CT");
    }

    #[test]
    fn test_ct_loader_name() {
        let loader = CTImageLoader::new();
        assert_eq!(loader.name(), "CT (NIFTI)");
        assert_eq!(loader.modality(), "CT");
    }

    #[test]
    fn test_hu_to_sound_speed_bone() {
        // High HU (bone) should give higher sound speed
        let c_bone = CTImageLoader::hu_to_sound_speed(1500.0);
        assert!(c_bone > 3000.0);

        let c_very_dense = CTImageLoader::hu_to_sound_speed(2500.0);
        assert!(c_very_dense > c_bone);
    }

    #[test]
    fn test_hu_to_sound_speed_soft_tissue() {
        // Soft tissue (HU < 700) should give ~1500 m/s
        let c_water = CTImageLoader::hu_to_sound_speed(0.0);
        assert!((c_water - 1500.0).abs() < 1e-6);

        let c_tissue = CTImageLoader::hu_to_sound_speed(100.0);
        assert_eq!(c_tissue, 1500.0);
    }

    #[test]
    fn test_hu_to_density_bone() {
        // High HU (bone) should give higher density
        let rho_bone = CTImageLoader::hu_to_density(1000.0);
        assert!(rho_bone > 1500.0);

        let rho_very_dense = CTImageLoader::hu_to_density(2000.0);
        assert!(rho_very_dense > rho_bone);
    }

    #[test]
    fn test_hu_to_density_soft_tissue() {
        // Soft tissue (HU < 700) should give ~1000 kg/m³
        let rho_water = CTImageLoader::hu_to_density(0.0);
        assert_eq!(rho_water, 1000.0);

        let rho_tissue = CTImageLoader::hu_to_density(50.0);
        assert_eq!(rho_tissue, 1000.0);
    }

    #[test]
    fn test_identity_affine() {
        let affine = CTImageLoader::identity_affine();
        // Check diagonal is 1
        assert_eq!(affine[0][0], 1.0);
        assert_eq!(affine[1][1], 1.0);
        assert_eq!(affine[2][2], 1.0);
        assert_eq!(affine[3][3], 1.0);
        // Check off-diagonal is 0
        assert_eq!(affine[0][1], 0.0);
        assert_eq!(affine[3][0], 0.0);
    }
}
