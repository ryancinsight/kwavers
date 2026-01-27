//! CT-based skull model loader
//!
//! Reference: Marquet et al. (2009) "Non-invasive transcranial ultrasound
//! therapy based on a 3D CT scan"

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use crate::physics::skull::HeterogeneousSkull;
use ndarray::Array3;

#[cfg(feature = "nifti")]
use nifti::{InMemNiftiObject, IntoNdArray, NiftiObject, NiftiVolume, ReaderOptions};

/// CT-based skull model
#[derive(Debug)]
pub struct CTBasedSkullModel {
    /// Hounsfield unit values from CT
    hounsfield: Array3<f64>,
    /// Voxel spacing (dx, dy, dz) in meters
    voxel_spacing: Option<(f64, f64, f64)>,
    /// Affine transformation matrix (4x4)
    affine: Option<[[f64; 4]; 4]>,
}

/// Metadata extracted from NIFTI header
#[derive(Debug, Clone)]
pub struct CTMetadata {
    /// Image dimensions (nx, ny, nz)
    pub dimensions: (usize, usize, usize),
    /// Voxel spacing in mm (dx, dy, dz)
    pub voxel_spacing_mm: (f64, f64, f64),
    /// Voxel spacing in meters (dx, dy, dz)
    pub voxel_spacing_m: (f64, f64, f64),
    /// Affine transformation matrix
    pub affine: [[f64; 4]; 4],
    /// Data type description
    pub data_type: String,
    /// Min/Max HU values in volume
    pub hu_range: (f64, f64),
}

impl CTBasedSkullModel {
    /// Load from NIFTI file (.nii or .nii.gz)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to NIFTI file containing CT scan (Hounsfield Units)
    ///
    /// # Returns
    ///
    /// CTBasedSkullModel with loaded HU values and metadata
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - File does not exist or cannot be read
    /// - File format is invalid (not NIFTI)
    /// - HU values are outside valid range (-1024 to +3071)
    /// - Dimensions are invalid (not 3D)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use kwavers::physics::skull::CTBasedSkullModel;
    /// let model = CTBasedSkullModel::from_file("patient_ct.nii.gz")?;
    /// # Ok::<(), kwavers::error::KwaversError>(())
    /// ```
    #[cfg(feature = "nifti")]
    pub fn from_file(path: &str) -> KwaversResult<Self> {
        use std::path::Path;

        // Validate file exists
        if !Path::new(path).exists() {
            return Err(KwaversError::InvalidInput(format!(
                "NIFTI file not found: {}",
                path
            )));
        }

        // Load NIFTI file
        let nifti_obj = ReaderOptions::new()
            .read_file(path)
            .map_err(|e| KwaversError::InvalidInput(format!("Failed to read NIFTI file: {}", e)))?;

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

        Ok(Self {
            hounsfield,
            voxel_spacing: Some(voxel_spacing_m),
            affine: Some(affine),
        })
    }

    /// Fallback when nifti feature is disabled
    #[cfg(not(feature = "nifti"))]
    pub fn from_file(path: &str) -> KwaversResult<Self> {
        Err(KwaversError::InvalidInput(format!(
            "NIFTI support not enabled. Rebuild with --features nifti. Path: {}",
            path
        )))
    }

    /// Create from CT data array (for testing or synthetic data)
    ///
    /// # Arguments
    ///
    /// * `ct_data` - 3D array of Hounsfield Units
    pub fn from_ct_data(ct_data: &Array3<f64>) -> KwaversResult<Self> {
        // Validate HU range
        let (min_hu, max_hu) = Self::compute_hu_range(ct_data);
        Self::validate_hu_range(min_hu, max_hu)?;

        Ok(Self {
            hounsfield: ct_data.clone(),
            voxel_spacing: None,
            affine: None,
        })
    }

    /// Create from CT data with metadata
    pub fn from_ct_data_with_metadata(
        ct_data: &Array3<f64>,
        voxel_spacing: (f64, f64, f64),
        affine: [[f64; 4]; 4],
    ) -> KwaversResult<Self> {
        // Validate HU range
        let (min_hu, max_hu) = Self::compute_hu_range(ct_data);
        Self::validate_hu_range(min_hu, max_hu)?;

        Ok(Self {
            hounsfield: ct_data.clone(),
            voxel_spacing: Some(voxel_spacing),
            affine: Some(affine),
        })
    }

    /// Extract metadata from the loaded CT scan
    pub fn metadata(&self) -> CTMetadata {
        let dims = self.hounsfield.dim();
        let (min_hu, max_hu) = Self::compute_hu_range(&self.hounsfield);

        let voxel_spacing_m = self.voxel_spacing.unwrap_or((1e-3, 1e-3, 1e-3));
        let voxel_spacing_mm = (
            voxel_spacing_m.0 * 1e3,
            voxel_spacing_m.1 * 1e3,
            voxel_spacing_m.2 * 1e3,
        );

        CTMetadata {
            dimensions: dims,
            voxel_spacing_mm,
            voxel_spacing_m,
            affine: self.affine.unwrap_or(Self::identity_affine()),
            data_type: "Hounsfield Units (f64)".to_string(),
            hu_range: (min_hu, max_hu),
        }
    }

    // Private helper methods

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
        // NIFTI can store data in various formats (i8, i16, f32, f64, etc.)
        let raw_data = volume.into_ndarray::<f64>().map_err(|e| {
            KwaversError::InvalidInput(format!("Failed to convert NIFTI data: {}", e))
        })?;

        // Check dimensions match and convert to 3D array
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

    fn compute_hu_range(hounsfield: &Array3<f64>) -> (f64, f64) {
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

    fn validate_hu_range(min_hu: f64, max_hu: f64) -> KwaversResult<()> {
        // Valid CT HU range: -1024 (air) to +3071 (dense bone)
        // Some scanners use extended range, so we allow some tolerance
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
            eprintln!(
                "Warning: No bone detected (max HU = {:.0}). Expected skull HU > 700.",
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

    /// Generate binary skull mask from CT
    pub fn generate_mask(&self, grid: &Grid) -> KwaversResult<Array3<f64>> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Threshold HU values: bone typically > 700 HU
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if self.hounsfield[[i, j, k]] > 700.0 {
                        mask[[i, j, k]] = 1.0;
                    }
                }
            }
        }

        Ok(mask)
    }

    /// Convert to heterogeneous model
    pub fn to_heterogeneous(&self, grid: &Grid) -> KwaversResult<HeterogeneousSkull> {
        // Convert HU to acoustic properties using empirical relations
        // Reference: Aubry et al. (2003)
        let mut sound_speed = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut density = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut attenuation = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let hu = self.hounsfield[[i, j, k]];

                    if hu > 700.0 {
                        // Bone
                        sound_speed[[i, j, k]] = 2800.0 + (hu - 700.0) * 0.5;
                        density[[i, j, k]] = 1700.0 + (hu - 700.0) * 0.2;
                        attenuation[[i, j, k]] = 40.0 + (hu - 700.0) * 0.05;
                    } else {
                        // Soft tissue/water
                        sound_speed[[i, j, k]] = 1500.0;
                        density[[i, j, k]] = 1000.0;
                        attenuation[[i, j, k]] = 0.002;
                    }
                }
            }
        }

        Ok(HeterogeneousSkull {
            sound_speed,
            density,
            attenuation,
        })
    }

    /// Get CT data
    pub fn ct_data(&self) -> &Array3<f64> {
        &self.hounsfield
    }

    /// Get sound speed at specific voxel
    pub fn sound_speed(&self, i: usize, j: usize, k: usize) -> f64 {
        let hu = self.hounsfield[[i, j, k]];
        if hu > 700.0 {
            // Bone: empirical relation from Aubry et al. (2003)
            2800.0 + (hu - 700.0) * 2.0
        } else {
            // Soft tissue
            1500.0
        }
    }
}
