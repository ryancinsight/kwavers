//! ritk-io → kwavers DICOM adapter (SSOT bridge).
//!
//! This module is the canonical seam between the ritk-io DICOM reader (which
//! returns a Burn `Image<B, 3>`) and the kwavers `Array3<f64>` consumer.
//! Both `domain::imaging::medical::dicom_loader::DicomImageLoader` and the
//! therapy orchestrator's CT-loading branch route through these helpers.
//!
//! # Why this exists
//!
//! `ritk_io::scan_dicom_directory` and `ritk_io::load_dicom_series::<B>` own
//! the DICOM transfer-syntax decoding, slice-spatial sorting, and
//! Image-Position/Image-Orientation handling. The kwavers domain layer,
//! however, operates on `Array3<f64>` plus a `MedicalImageMetadata` record
//! (lengths in metres). This module performs the two conversions:
//!
//! 1. ritk-io's `[depth, rows, cols]` `f32` tensor → `Array3<f64>` shape
//!    `(cols, rows, depth)` (kwavers' `(x, y, z)` ordering convention).
//! 2. ritk-io's `Spacing<3>`/`Point<3>` (millimetres) → kwavers'
//!    `MedicalImageMetadata` (metres + axis dimensions).
//!
//! # Single source of truth
//!
//! The `dicom` crate is owned by ritk-io and pulled transitively. New DICOM
//! decoding code MUST go through these helpers — never call the `dicom`
//! crate directly from kwavers source.

use std::path::Path;

use ndarray::Array3;
use ritk_io::{load_dicom_series, scan_dicom_directory, DicomSeriesInfo};

use kwavers_core::error::{KwaversError, KwaversResult};
use crate::medical::ritk_bridge::{image_to_volume, AdapterBackend};
use crate::medical::MedicalImageMetadata;

/// Result of loading a DICOM series via the ritk-io bridge.
#[derive(Debug, Clone)]
pub struct DicomSeriesVolume {
    /// 3-D voxel volume in kwavers `(x, y, z)` ordering, metres-spaced.
    pub voxels: Array3<f64>,
    /// Metadata in kwavers' canonical units (metres for spacing).
    pub metadata: MedicalImageMetadata,
    /// Original ritk-io series info, retained so callers can read e.g.
    /// `series_instance_uid` or `modality` without re-scanning.
    pub series_info: DicomSeriesInfo,
}

/// Scan a directory and select the unique DICOM series, or return an error
/// describing the available series UIDs when more than one is present.
///
/// # Errors
///
/// Returns `KwaversError::InvalidInput` when the directory contains zero or
/// multiple series (for the multi-series case, the error message lists every
/// `(uid, modality, file_count)` so the caller can re-invoke
/// [`load_series_with_uid`] with an explicit UID).
/// # Panics
/// - Panics if `len == 1`.
///
pub fn select_unique_series<P: AsRef<Path>>(path: P) -> KwaversResult<DicomSeriesInfo> {
    let series = scan_dicom_directory(path.as_ref()).map_err(|e| {
        KwaversError::InternalError(format!(
            "ritk-io failed to scan DICOM directory '{}': {e}",
            path.as_ref().display()
        ))
    })?;

    match series.len() {
        0 => Err(KwaversError::InvalidInput(format!(
            "no DICOM series found in '{}'",
            path.as_ref().display()
        ))),
        1 => Ok(series.into_iter().next().expect("len == 1")),
        n => {
            let mut report = format!("{n} DICOM series found in directory; specify one by UID:\n");
            for item in &series {
                report.push_str(&format!(
                    "  uid={} modality={} description={} files={}\n",
                    item.series_instance_uid,
                    item.modality,
                    item.series_description,
                    item.file_paths.len()
                ));
            }
            Err(KwaversError::InvalidInput(report))
        }
    }
}

/// Load a DICOM series from a directory, picking the unique series when
/// exactly one is present.
///
/// # Errors
///
/// Returns an error if zero or multiple series are present (use
/// [`load_series_with_uid`] for the multi-series case), if ritk-io fails to
/// decode the series, or if the resulting tensor is not contiguous `f32`.
pub fn load_series_from_dir<P: AsRef<Path>>(path: P) -> KwaversResult<DicomSeriesVolume> {
    let info = select_unique_series(path.as_ref())?;
    load_series(&info)
}

/// Load a DICOM series from a directory, selecting the series by its
/// `SeriesInstanceUID`.
///
/// # Errors
///
/// Returns an error if scanning fails, if no series matches the UID, or if
/// ritk-io fails to decode the series.
pub fn load_series_with_uid<P: AsRef<Path>>(
    path: P,
    series_uid: &str,
) -> KwaversResult<DicomSeriesVolume> {
    let series = scan_dicom_directory(path.as_ref()).map_err(|e| {
        KwaversError::InternalError(format!(
            "ritk-io failed to scan DICOM directory '{}': {e}",
            path.as_ref().display()
        ))
    })?;

    let info = series
        .into_iter()
        .find(|s| s.series_instance_uid == series_uid)
        .ok_or_else(|| {
            KwaversError::InvalidInput(format!(
                "DICOM series UID '{series_uid}' not found in '{}'",
                path.as_ref().display()
            ))
        })?;

    load_series(&info)
}

/// Convert a previously discovered `DicomSeriesInfo` into a
/// kwavers-native [`DicomSeriesVolume`].
///
/// # Errors
///
/// Returns an error if ritk-io fails to decode the series or if the
/// returned tensor is not contiguous `f32`.
pub fn load_series(info: &DicomSeriesInfo) -> KwaversResult<DicomSeriesVolume> {
    let device = Default::default();
    let image = load_dicom_series::<AdapterBackend>(info, &device).map_err(|e| {
        KwaversError::InternalError(format!(
            "ritk-io failed to decode DICOM series '{}': {e}",
            info.series_instance_uid
        ))
    })?;

    let vol = image_to_volume(&image)?;

    let metadata = MedicalImageMetadata {
        dimensions: vol.dimensions,
        voxel_spacing_m: (
            vol.voxel_spacing_mm.0 * 1e-3,
            vol.voxel_spacing_mm.1 * 1e-3,
            vol.voxel_spacing_mm.2 * 1e-3,
        ),
        voxel_spacing_mm: vol.voxel_spacing_mm,
        affine: vol.affine,
        data_type: format!("{} via ritk-io (f32 → f64)", info.modality),
        intensity_range: vol.intensity_range,
        modality: info.modality.clone(),
    };

    Ok(DicomSeriesVolume {
        voxels: vol.voxels,
        metadata,
        series_info: info.clone(),
    })
}
