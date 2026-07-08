//! Canonical ritk-io → kwavers volume bridge.
//!
//! Medical image formats are decoded by ritk-io — ritk is the single source of
//! truth for medical image I/O. This module owns the conversion from RITK image
//! metadata and row-major `[depth, rows, cols]` host data into kwavers'
//! canonical `(x, y, z)` `Array3<f64>` plus voxel spacing and affine.
//!
//! NIfTI and DICOM route through the native Coeus-backed RITK image path.

use coeus_core::SequentialBackend;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;
use ritk_image::native::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

/// A ritk-decoded volume in kwavers' canonical conventions.
#[derive(Debug, Clone)]
pub(crate) struct RitkVolume {
    /// Voxels in kwavers `(x, y, z)` ordering as `f64`.
    pub voxels: Array3<f64>,
    /// `(nx, ny, nz)`.
    pub dimensions: (usize, usize, usize),
    /// Voxel spacing in millimetres `(dx, dy, dz)`.
    pub voxel_spacing_mm: (f64, f64, f64),
    /// 4×4 affine: `direction · diag(spacing_mm)` linear part, `origin_mm`
    /// translation column.
    pub affine: [[f64; 4]; 4],
    /// `(min, max)` intensity over the volume.
    pub intensity_range: (f64, f64),
}

/// Convert a native Coeus-backed RITK image into a kwavers [`RitkVolume`].
///
/// # Errors
///
/// Returns [`KwaversError::InternalError`] if the native RITK image is not
/// contiguous host-addressable data or if its length does not match the shape.
pub(crate) fn native_image_to_volume(
    image: &NativeImage<f32, SequentialBackend, 3>,
) -> KwaversResult<RitkVolume> {
    let [depth, rows, cols] = image.shape();
    let values = image.data_slice().map_err(|err| {
        KwaversError::InternalError(format!(
            "native ritk image data is not contiguous f32: {err}"
        ))
    })?;
    image_components_to_volume(
        [depth, rows, cols],
        image.spacing(),
        image.origin(),
        image.direction(),
        values,
        "native ritk image",
    )
}

fn image_components_to_volume(
    [depth, rows, cols]: [usize; 3],
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    values: &[f32],
    source_label: &str,
) -> KwaversResult<RitkVolume> {
    let spacing_mm = spacing.to_array();
    let origin_mm = origin.to_array();

    let expected_len = depth * rows * cols;
    if values.len() != expected_len {
        return Err(KwaversError::InternalError(format!(
            "{source_label} length {} does not match {depth}×{rows}×{cols} = {expected_len}",
            values.len()
        )));
    }

    // Repack ritk-io's `[depth, rows, cols]` (z, y, x) layout into kwavers'
    // `(x, y, z)` indexing while tracking the intensity range.
    let mut voxels = Array3::<f64>::zeros([cols, rows, depth]);
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for z in 0..depth {
        for y in 0..rows {
            for x in 0..cols {
                let v = f64::from(values[z * rows * cols + y * cols + x]);
                voxels[[x, y, z]] = v;
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
        }
    }
    if !min_val.is_finite() || !max_val.is_finite() {
        min_val = 0.0;
        max_val = 0.0;
    }

    // Affine: direction × diag(spacing_mm) for the linear part, origin_mm for
    // the translation column (millimetres, per NIfTI/DICOM convention).
    let mut affine = [[0.0_f64; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            affine[i][j] = direction.0[(i, j)] * spacing_mm[j];
        }
        affine[i][3] = origin_mm[i];
    }
    affine[3] = [0.0, 0.0, 0.0, 1.0];

    Ok(RitkVolume {
        voxels,
        dimensions: (cols, rows, depth),
        voxel_spacing_mm: (spacing_mm[0], spacing_mm[1], spacing_mm[2]),
        affine,
        intensity_range: (min_val, max_val),
    })
}
