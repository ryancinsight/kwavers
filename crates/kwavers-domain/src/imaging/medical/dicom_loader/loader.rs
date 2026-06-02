//! `DicomImageLoader` — DICOM series loading and metadata extraction.
//!
//! Delegates pixel-data decoding to the sibling `dicom_ritk` adapter
//! (the canonical SSOT wrapper over `ritk_io`). HU-conversion helpers
//! and the affine-from-IPP/IOP helper remain here for callers that work with
//! raw DICOM-derived spatial metadata directly.

use kwavers_core::error::{KwaversError, KwaversResult};
use crate::imaging::medical::{MedicalImageLoader, MedicalImageMetadata};
use ndarray::Array3;
use std::path::Path;

use super::types::{DicomMetadata, DicomModality};

/// DICOM image loader.
///
/// Loads multi-slice DICOM series and exposes the 3D volume as `Array3<f64>`
/// together with rich header metadata.
#[derive(Debug)]
pub struct DicomImageLoader {
    data: Option<Array3<f64>>,
    metadata: Option<DicomMetadata>,
}

impl DicomImageLoader {
    /// Create a new, empty DICOM image loader.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: None,
            metadata: None,
        }
    }

    /// Return loaded image data, if any.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn data(&self) -> Option<&Array3<f64>> {
        self.data.as_ref()
    }

    /// Return DICOM-specific metadata, if any.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn dicom_metadata(&self) -> Option<&DicomMetadata> {
        self.metadata.as_ref()
    }

    /// Internal series-load: delegates to the sibling `dicom_ritk` adapter.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn load_series_internal(&mut self, dir_path: &str) -> KwaversResult<Array3<f64>> {
        let volume =
            super::dicom_ritk::load_series_from_dir(Path::new(dir_path))?;

        let modality = match volume.metadata.modality.as_str() {
            "CT" => DicomModality::CT,
            "MR" | "MRI" => DicomModality::MR,
            "US" => DicomModality::US,
            "RTDOSE" | "RD" => DicomModality::RD,
            _ => DicomModality::Other,
        };

        let dims = volume.metadata.dimensions;
        let spacing_mm = volume.metadata.voxel_spacing_mm;
        let spacing_m = volume.metadata.voxel_spacing_m;
        let intensity_range = volume.metadata.intensity_range;

        self.metadata = Some(DicomMetadata {
            dimensions: dims,
            voxel_spacing_mm: spacing_mm,
            voxel_spacing_m: spacing_m,
            affine: volume.metadata.affine,
            modality,
            patient_id: String::new(),
            patient_name: String::new(),
            patient_birth_date: None,
            patient_sex: None,
            study_date: String::new(),
            study_time: String::new(),
            study_description: volume.series_info.series_description.clone(),
            series_description: volume.series_info.series_description.clone(),
            series_instance_uid: volume.series_info.series_instance_uid.clone(),
            study_instance_uid: String::new(),
            num_slices: dims.2,
            slice_thickness_mm: spacing_mm.2,
            image_position: None,
            image_orientation: None,
            intensity_range,
            window_center: None,
            window_width: None,
            rescale_intercept: None,
            rescale_slope: None,
        });

        let voxels = volume.voxels.clone();
        self.data = Some(volume.voxels);
        Ok(voxels)
    }

    pub(super) fn identity_affine() -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Convert a DICOM pixel value to Hounsfield Units (CT only).
    ///
    /// Formula: `HU = pixel_value × rescale_slope + rescale_intercept`
    #[must_use]
    pub fn to_hounsfield_units(
        pixel_value: f64,
        rescale_slope: f64,
        rescale_intercept: f64,
    ) -> f64 {
        pixel_value.mul_add(rescale_slope, rescale_intercept)
    }

    /// Compute a 4×4 affine matrix from DICOM IPP/IOP tags.
    ///
    /// # Arguments
    /// - `image_position` — Image Position (Patient): `[x0, y0, z0]`
    /// - `image_orientation` — Image Orientation (Patient): `[xx, xy, xz, yx, yy, yz]`
    /// - `pixel_spacing` — Pixel Spacing: `[dx, dy]`
    /// - `slice_thickness` — Slice Thickness
    #[must_use]
    pub fn compute_affine(
        image_position: &[f64; 3],
        image_orientation: &[f64; 6],
        pixel_spacing: &[f64; 2],
        slice_thickness: f64,
    ) -> [[f64; 4]; 4] {
        let xx = image_orientation[0];
        let xy = image_orientation[1];
        let xz = image_orientation[2];
        let yx = image_orientation[3];
        let yy = image_orientation[4];
        let yz = image_orientation[5];

        let zx = xy.mul_add(yz, -(xz * yy));
        let zy = xz.mul_add(yx, -(xx * yz));
        let zz = xx.mul_add(yy, -(xy * yx));

        [
            [
                xx * pixel_spacing[0],
                yx * pixel_spacing[1],
                zx * slice_thickness,
                image_position[0],
            ],
            [
                xy * pixel_spacing[0],
                yy * pixel_spacing[1],
                zy * slice_thickness,
                image_position[1],
            ],
            [
                xz * pixel_spacing[0],
                yz * pixel_spacing[1],
                zz * slice_thickness,
                image_position[2],
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

impl Default for DicomImageLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl MedicalImageLoader for DicomImageLoader {
    fn load(&mut self, path: &str) -> KwaversResult<Array3<f64>> {
        let file_path = Path::new(path);

        if file_path.is_dir() {
            self.load_series_internal(path)
        } else if file_path.is_file() {
            Err(KwaversError::InvalidInput(
                "Single DICOM file loading: Please provide directory path containing complete series".to_owned(),
            ))
        } else {
            Err(KwaversError::InvalidInput(format!(
                "Path does not exist: {}",
                path
            )))
        }
    }

    fn metadata(&self) -> MedicalImageMetadata {
        if let Some(dicom_meta) = &self.metadata {
            MedicalImageMetadata {
                dimensions: dicom_meta.dimensions,
                voxel_spacing_m: dicom_meta.voxel_spacing_m,
                voxel_spacing_mm: dicom_meta.voxel_spacing_mm,
                affine: dicom_meta.affine,
                data_type: format!("DICOM ({})", dicom_meta.modality),
                intensity_range: dicom_meta.intensity_range,
                modality: dicom_meta.modality.to_string(),
            }
        } else {
            MedicalImageMetadata {
                dimensions: (0, 0, 0),
                voxel_spacing_m: (1e-3, 1e-3, 1e-3),
                voxel_spacing_mm: (1.0, 1.0, 1.0),
                affine: Self::identity_affine(),
                data_type: "DICOM (Unknown)".to_owned(),
                intensity_range: (0.0, 0.0),
                modality: "Unknown".to_owned(),
            }
        }
    }

    fn name(&self) -> &str {
        "DICOM"
    }

    fn modality(&self) -> &str {
        if let Some(meta) = &self.metadata {
            match meta.modality {
                DicomModality::CT => "CT",
                DicomModality::MR => "MR",
                DicomModality::US => "US",
                DicomModality::RD => "RD",
                DicomModality::Other => "Other",
            }
        } else {
            "Unknown"
        }
    }
}
