//! Shared CT loading and voxel-space geometry for seismic examples.

use anyhow::Context as _;
use coeus_core::MoiraiBackend;
use leto::Array3;
use ritk_io::format::nifti::native::NiftiReader as NativeNiftiReader;
use ritk_io::format::png::native::PngSeriesReader as NativePngSeriesReader;
use ritk_io::ImageReader;
use ritk_io::{load_native_dicom_series, scan_dicom_directory};
use std::path::Path;

/// Raw CT volume in voxel space.
///
/// `hu` has shape `(cols, rows, depth)` = `(x, y, z)` in the patient frame.
/// `spacing_mm` is `[dx, dy, dz]`, the physical millimetres per voxel.
pub(crate) struct CtVolume {
    hu: Array3<f64>,
    spacing_mm: [f64; 3],
}

impl CtVolume {
    /// Returns the Hounsfield-unit volume in `[x, y, z]` order.
    pub(crate) const fn hu(&self) -> &Array3<f64> {
        &self.hu
    }

    /// Returns the voxel spacing in millimetres as `[dx, dy, dz]`.
    pub(crate) const fn spacing_mm(&self) -> [f64; 3] {
        self.spacing_mm
    }
}

/// Load a CT volume from a NIfTI file, DICOM directory, or PNG series.
pub(crate) fn load_ct_volume(path: &Path) -> anyhow::Result<CtVolume> {
    let backend = MoiraiBackend;

    if path.is_dir() {
        let has_png = std::fs::read_dir(path)
            .with_context(|| format!("failed to read dir '{}'", path.display()))?
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
            });

        if has_png {
            println!("  PNG series      : {}", path.display());
            let image = ImageReader::read(&NativePngSeriesReader::new(backend), path)
                .map_err(|error| anyhow::anyhow!("PNG series load failed: {error:#}"))?;
            let [depth, rows, cols] = image.shape();
            let values = image
                .data_slice()
                .map_err(|error| anyhow::anyhow!("PNG tensor data is not f32: {error:?}"))?;
            anyhow::ensure!(
                values.len() == depth * rows * cols,
                "PNG data length mismatch: got {}, expected {}",
                values.len(),
                depth * rows * cols
            );
            anyhow::ensure!(
                values
                    .iter()
                    .all(|&value| value.is_finite() && (0.0..=255.0).contains(&value)),
                "PNG samples must be finite display values in [0, 255]"
            );
            let peak = values.iter().copied().fold(0.0_f32, f32::max);
            anyhow::ensure!(
                peak > 1.0,
                "PNG samples look normalized (peak {peak}); expected 8-bit display values"
            );

            // Bone window W=2000, C=400 maps display pixels back to HU.
            const PNG_WINDOW: f64 = 2000.0;
            const PNG_CENTER: f64 = 400.0;
            // PNG series have no physical spacing metadata; this is the
            // acquisition spacing assumed by the example's HU reconstruction.
            const PNG_SPACING_MM: [f64; 3] = [0.5, 0.5, 4.0];
            let hu_low = PNG_CENTER - PNG_WINDOW / 2.0;
            let hu_per_pixel = PNG_WINDOW / 255.0;
            let mut hu = Array3::<f64>::zeros((cols, rows, depth));
            for z in 0..depth {
                for y in 0..rows {
                    for x in 0..cols {
                        let pixel = f64::from(values[z * rows * cols + y * cols + x]);
                        hu[[x, y, z]] = hu_low + pixel * hu_per_pixel;
                    }
                }
            }
            clamp_hounsfield_units(&mut hu);
            return Ok(CtVolume {
                hu,
                spacing_mm: PNG_SPACING_MM,
            });
        }
    }

    let image = if path.is_dir() {
        let series = scan_dicom_directory(path)
            .with_context(|| format!("failed to scan DICOM dir '{}'", path.display()))?;
        if series.is_empty() {
            anyhow::bail!("no DICOM series found in '{}'", path.display());
        }
        let selected = super::dicom::select_series(series);
        println!(
            "  DICOM series    : '{}' ({} files)",
            selected.series_description,
            selected.file_paths.len()
        );
        load_native_dicom_series(&selected, &backend).map_err(|error| {
            anyhow::anyhow!(
                "DICOM load failed for series '{}': {error:#}",
                selected.series_instance_uid()
            )
        })?
    } else {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".nii") && !name.ends_with(".nii.gz") {
            anyhow::bail!(
                "unrecognised format for '{}'; expected .nii/.nii.gz or a DICOM dir",
                path.display()
            );
        }
        println!("  NIfTI file      : {}", path.display());
        ImageReader::read(&NativeNiftiReader::new(backend), path)
            .with_context(|| format!("NIfTI read failed for '{}'", path.display()))?
    };

    let [depth, rows, cols] = image.shape();
    let spacing = image.spacing().into_vector().to_array();
    let values = image
        .data_slice()
        .map_err(|error| anyhow::anyhow!("data tensor is not f32: {error:?}"))?;
    anyhow::ensure!(
        values.len() == depth * rows * cols,
        "data length mismatch: got {}, expected {}",
        values.len(),
        depth * rows * cols
    );

    let mut hu = Array3::<f64>::zeros((cols, rows, depth));
    for z in 0..depth {
        for y in 0..rows {
            for x in 0..cols {
                hu[[x, y, z]] = f64::from(values[z * rows * cols + y * cols + x]);
            }
        }
    }
    clamp_hounsfield_units(&mut hu);
    Ok(CtVolume {
        hu,
        spacing_mm: [spacing[0], spacing[1], spacing[2]],
    })
}

fn clamp_hounsfield_units(hu: &mut Array3<f64>) {
    for value in hu.iter_mut() {
        *value = (*value).clamp(-1024.0, 3071.0);
    }
}

/// Find the axial slice with the largest number of bone voxels.
pub(crate) fn skull_equator_z(hu: &Array3<f64>) -> usize {
    let [_, _, nz] = hu.shape();
    (0..nz)
        .max_by_key(|&z| {
            hu.index_axis::<2>(2, z)
                .expect("invariant: equator slice index is in bounds")
                .iter()
                .filter(|&&value| value > 300.0)
                .count()
        })
        .unwrap_or(nz / 2)
}

/// Find the centroid of bone voxels on an axial CT slice.
pub(crate) fn skull_centroid_2d(hu: &Array3<f64>, z: usize) -> (f64, f64) {
    let slice = hu
        .index_axis::<2>(2, z)
        .expect("invariant: centroid slice index is in bounds");
    let [nx, ny] = slice.shape();
    let (mut sum_x, mut sum_y, mut count) = (0.0, 0.0, 0.0);
    for ([x, y], &value) in slice.indexed_iter() {
        if value > 300.0 {
            sum_x += x as f64;
            sum_y += y as f64;
            count += 1.0;
        }
    }
    if count > 0.0 {
        (sum_x / count, sum_y / count)
    } else {
        (nx as f64 / 2.0, ny as f64 / 2.0)
    }
}

/// Measure the outer skull radius around an axial-slice centroid.
pub(crate) fn skull_outer_radius_ct(hu: &Array3<f64>, z: usize, cx: f64, cy: f64) -> f64 {
    let [nx, ny, _] = hu.shape();
    let radius = hu
        .index_axis::<2>(2, z)
        .expect("invariant: radius slice index is in bounds")
        .indexed_iter()
        .filter(|(_, &value)| value > 300.0)
        .map(|([x, y], _)| {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0_f64, f64::max);
    if radius < 1.0 {
        (nx.min(ny) / 4) as f64
    } else {
        radius
    }
}

#[cfg(test)]
mod tests {
    use super::{
        clamp_hounsfield_units, skull_centroid_2d, skull_equator_z, skull_outer_radius_ct,
    };
    use leto::Array3;

    #[test]
    fn axial_geometry_uses_the_bone_dominant_slice() {
        let mut hu = Array3::<f64>::zeros((5, 5, 3));
        for x in 1..4 {
            for y in 1..4 {
                hu[[x, y, 1]] = 700.0;
            }
        }

        assert_eq!(skull_equator_z(&hu), 1);
        assert_eq!(skull_centroid_2d(&hu, 1), (2.0, 2.0));
        assert_eq!(skull_outer_radius_ct(&hu, 1, 2.0, 2.0), 2.0_f64.sqrt());
    }

    #[test]
    fn empty_geometry_returns_center_and_safe_radius() {
        let hu = Array3::<f64>::zeros((4, 6, 2));

        assert_eq!(skull_equator_z(&hu), 1);
        assert_eq!(skull_centroid_2d(&hu, 1), (2.0, 3.0));
        assert_eq!(skull_outer_radius_ct(&hu, 1, 2.0, 3.0), 1.0);
    }

    #[test]
    fn hounsfield_units_are_clamped_to_the_provider_range() {
        let mut hu = Array3::from_shape_vec((3, 1, 1), vec![-2000.0, 120.0, 4000.0])
            .expect("test shape matches values");

        clamp_hounsfield_units(&mut hu);

        assert_eq!(
            hu.as_slice().expect("standard layout"),
            &[-1024.0, 120.0, 3071.0]
        );
    }
}
