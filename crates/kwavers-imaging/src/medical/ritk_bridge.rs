//! Canonical ritk-io → kwavers volume bridge.
//!
//! All medical image formats (DICOM, NIfTI, and the other formats ritk-io
//! supports) are decoded by ritk-io — ritk is the single source of truth for
//! medical image I/O. This module owns the *one* conversion from a ritk-core
//! `Image<NdArray, 3>` (a burn-tensor-backed image with ITK-style
//! origin/spacing/direction metadata) into kwavers' canonical
//! `(x, y, z)`-ordered `Array3<f64>` plus voxel spacing and affine. Both the
//! DICOM and CT/NIfTI loaders route through [`image_to_volume`].

use burn::backend::NdArray;
use burn::tensor::TensorData;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use ritk_core::Image;

/// CPU backend used by the host-side adapters. ritk-io readers are generic over
/// `Backend`; `NdArray` avoids GPU staging round-trips for pure-CPU consumers,
/// and the `f32 → f64` conversion happens during the host-side copy.
pub(crate) type AdapterBackend = NdArray;

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

/// Convert a ritk-core image into a kwavers [`RitkVolume`].
///
/// # Errors
///
/// Returns [`KwaversError::InternalError`] if ritk-io returns a non-rank-3
/// spacing/origin, a non-contiguous-`f32` tensor, or a tensor whose length does
/// not match the reported shape.
pub(crate) fn image_to_volume(image: &Image<AdapterBackend, 3>) -> KwaversResult<RitkVolume> {
    let [depth, rows, cols] = image.shape();

    let spacing_mm = image.spacing().to_vec();
    if spacing_mm.len() != 3 {
        return Err(KwaversError::InternalError(format!(
            "ritk-io returned spacing rank {} (expected 3)",
            spacing_mm.len()
        )));
    }
    let origin_mm = image.origin().to_vec();
    if origin_mm.len() != 3 {
        return Err(KwaversError::InternalError(format!(
            "ritk-io returned origin rank {} (expected 3)",
            origin_mm.len()
        )));
    }

    let tensor_data: TensorData = image.data().clone().into_data();
    let values = tensor_data.as_slice::<f32>().map_err(|err| {
        KwaversError::InternalError(format!(
            "ritk-io tensor data is not contiguous f32: {err:?}"
        ))
    })?;
    let expected_len = depth * rows * cols;
    if values.len() != expected_len {
        return Err(KwaversError::InternalError(format!(
            "ritk-io tensor length {} does not match {depth}×{rows}×{cols} = {expected_len}",
            values.len()
        )));
    }

    // Repack ritk-io's `[depth, rows, cols]` (z, y, x) layout into kwavers'
    // `(x, y, z)` indexing while tracking the intensity range.
    let mut voxels = Array3::<f64>::zeros((cols, rows, depth));
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
    let direction = image.direction();
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
