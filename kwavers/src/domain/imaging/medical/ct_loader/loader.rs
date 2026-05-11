//! CT image loader implementation.

use super::types::CTMetadata;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::imaging::medical::{MedicalImageLoader, MedicalImageMetadata};
use log::warn;
use ndarray::Array3;

#[cfg(feature = "nifti")]
use nifti::{InMemNiftiObject, IntoNdArray, NiftiObject, NiftiVolume, ReaderOptions};

/// CT Image Loader — loads NIFTI format CT scans.
///
/// Supports `.nii` and `.nii.gz` (gzip-compressed) NIFTI files.
/// Extracts Hounsfield Units (HU) with validation.
/// Validates HU range: -2000 to +4000 (extended range for various scanners).
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

impl CTImageLoader {
    /// Create new CT image loader.
    #[must_use] 
    pub fn new() -> Self {
        Self {
            data: None,
            metadata: None,
        }
    }

    /// Get loaded CT data.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use] 
    pub fn data(&self) -> Option<&Array3<f64>> {
        self.data.as_ref()
    }

    /// Get CT metadata.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use] 
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

            if !Path::new(path).exists() {
                return Err(KwaversError::InvalidInput(format!(
                    "NIFTI file not found: {}",
                    path
                )));
            }

            let nifti_obj = ReaderOptions::new().read_file(path).map_err(|e| {
                KwaversError::InvalidInput(format!("Failed to read NIFTI file: {}", e))
            })?;

            let header = nifti_obj.header();
            let dims = header.dim;

            if dims[0] != 3 {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "dimensions".to_string(),
                    value: dims[0].to_string(),
                    constraint: "must be 3D volume".to_string(),
                }));
            }

            let (nx, ny, nz) = (dims[1] as usize, dims[2] as usize, dims[3] as usize);

            let pixdim = header.pixdim;
            let voxel_spacing_mm = (pixdim[1] as f64, pixdim[2] as f64, pixdim[3] as f64);
            let voxel_spacing_m = (
                voxel_spacing_mm.0 * 1e-3,
                voxel_spacing_mm.1 * 1e-3,
                voxel_spacing_mm.2 * 1e-3,
            );

            let affine = Self::extract_affine_matrix(&nifti_obj)?;
            let hounsfield = Self::extract_hounsfield_units(nifti_obj.volume(), nx, ny, nz)?;

            let (min_hu, max_hu) = Self::compute_hu_range(&hounsfield);
            Self::validate_hu_range(min_hu, max_hu)?;

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
                modality: "CT".to_owned(),
            }
        } else {
            MedicalImageMetadata {
                dimensions: (0, 0, 0),
                voxel_spacing_m: (1e-3, 1e-3, 1e-3),
                voxel_spacing_mm: (1.0, 1.0, 1.0),
                affine: Self::identity_affine(),
                data_type: "Hounsfield Units (f64)".to_owned(),
                intensity_range: (0.0, 0.0),
                modality: "CT".to_owned(),
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

        if header.qform_code > 0 {
            let pixdim = header.pixdim;
            let affine = [
                [pixdim[1] as f64, 0.0, 0.0, 0.0],
                [0.0, pixdim[2] as f64, 0.0, 0.0],
                [0.0, 0.0, pixdim[3] as f64, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            return Ok(affine);
        }

        Ok(Self::identity_affine())
    }

    #[cfg(feature = "nifti")]
    fn extract_hounsfield_units<V: NiftiVolume + IntoNdArray>(
        volume: V,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<Array3<f64>> {
        let raw_data = volume.into_ndarray::<f64>().map_err(|e| {
            KwaversError::InvalidInput(format!("Failed to convert NIFTI data: {}", e))
        })?;

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

        let hounsfield = raw_data
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| {
                KwaversError::InvalidInput(format!("Failed to convert to 3D array: {}", e))
            })?;

        Ok(hounsfield)
    }

    /// Compute min/max Hounsfield Unit values in the CT volume.
    #[must_use] 
    pub fn compute_hu_range(hounsfield: &Array3<f64>) -> (f64, f64) {
        let mut min_hu = f64::INFINITY;
        let mut max_hu = f64::NEG_INFINITY;

        for &val in hounsfield {
            if val < min_hu {
                min_hu = val;
            }
            if val > max_hu {
                max_hu = val;
            }
        }

        (min_hu, max_hu)
    }

    /// Validate that HU values are within physically reasonable range (-2000 to +4000).
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn validate_hu_range(min_hu: f64, max_hu: f64) -> KwaversResult<()> {
        const MIN_VALID_HU: f64 = -2000.0;
        const MAX_VALID_HU: f64 = 4000.0;

        if min_hu < MIN_VALID_HU || max_hu > MAX_VALID_HU {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "hounsfield_units".to_owned(),
                value: format!("({:.0}, {:.0})", min_hu, max_hu),
                constraint: format!(
                    "must be within valid CT range ({:.0}, {:.0})",
                    MIN_VALID_HU, MAX_VALID_HU
                ),
            }));
        }

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

    /// Convert HU to sound speed (Aubry et al. 2003).
    ///
    /// Bone (HU > 700): c(HU) = 2800 + (HU - 700) × 2.0 m/s
    /// Soft tissue: c = 1500 m/s
    #[must_use] 
    pub fn hu_to_sound_speed(hu: f64) -> f64 {
        if hu > 700.0 {
            (hu - 700.0).mul_add(2.0, 2800.0)
        } else {
            1500.0
        }
    }

    /// Convert HU to density.
    ///
    /// Bone (HU > 700): ρ(HU) = 1700 + (HU - 700) × 0.2 kg/m³
    /// Soft tissue: ρ = 1000 kg/m³
    #[must_use] 
    pub fn hu_to_density(hu: f64) -> f64 {
        if hu > 700.0 {
            (hu - 700.0).mul_add(0.2, 1700.0)
        } else {
            1000.0
        }
    }
}
