//! CT image loader implementation.

use super::types::CTMetadata;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use crate::imaging::medical::{MedicalImageLoader, MedicalImageMetadata};
use log::warn;
use ndarray::Array3;

/// CT Image Loader — loads NIFTI format CT scans.
///
/// Supports `.nii` and `.nii.gz` (gzip-compressed) NIFTI files.
/// Extracts Hounsfield Units (HU) with validation.
/// Validates HU range: -2000 to +4000 (extended range for various scanners).
///
/// # Example
///
/// ```no_run
/// # use kwavers_domain::imaging::medical::{CTImageLoader, MedicalImageLoader};
/// let mut loader = CTImageLoader::new();
/// let ct_data = loader.load("patient_ct.nii.gz")?;
/// let metadata = loader.metadata();
/// println!("CT dimensions: {:?}", metadata.dimensions);
/// # Ok::<(), kwavers_core::error::KwaversError>(())
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
        use crate::imaging::medical::ritk_bridge::{image_to_volume, AdapterBackend};

        if !std::path::Path::new(path).exists() {
            return Err(KwaversError::InvalidInput(format!(
                "NIfTI file not found: {path}"
            )));
        }

        // ritk owns medical-image I/O: decode the .nii/.nii.gz via ritk-io into a
        // ritk-core image, then bridge it to kwavers' `(x, y, z)` Array3<f64>.
        let device = Default::default();
        let image = ritk_io::read_nifti::<AdapterBackend, _>(path, &device).map_err(|e| {
            KwaversError::InvalidInput(format!("ritk-io failed to read NIfTI '{path}': {e}"))
        })?;
        let vol = image_to_volume(&image)?;

        if vol.dimensions.0 == 0 || vol.dimensions.1 == 0 || vol.dimensions.2 == 0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "dimensions".to_owned(),
                value: format!("{:?}", vol.dimensions),
                constraint: "must be a non-empty 3D volume".to_owned(),
            }));
        }

        let (min_hu, max_hu) = vol.intensity_range;
        Self::validate_hu_range(min_hu, max_hu)?;

        self.metadata = Some(CTMetadata {
            dimensions: vol.dimensions,
            voxel_spacing_mm: vol.voxel_spacing_mm,
            voxel_spacing_m: (
                vol.voxel_spacing_mm.0 * 1e-3,
                vol.voxel_spacing_mm.1 * 1e-3,
                vol.voxel_spacing_mm.2 * 1e-3,
            ),
            affine: vol.affine,
            data_type: "Hounsfield Units (f64) via ritk-io".to_owned(),
            hu_range: (min_hu, max_hu),
        });

        self.data = Some(vol.voxels.clone());
        Ok(vol.voxels)
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
            SOUND_SPEED_WATER_SIM
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
