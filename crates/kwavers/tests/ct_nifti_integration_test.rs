//! Integration tests for NIFTI file I/O via RITK
//!
//! These tests exercise the RITK-backed NIfTI reader/writer stack.
//! Run with: cargo test

use coeus_core::SequentialBackend;
use leto::Array3;
use ritk_io::domain::{ImageReader, ImageWriter};
use ritk_io::format::nifti::native::{NiftiReader, NiftiWriter};
use ritk_spatial::{Direction, Point, Spacing};
use std::fs;
use std::path::{Path, PathBuf};

/// Helper to create synthetic NIFTI file for testing via RITK writer
fn create_synthetic_nifti(
    path: &str,
    dims: (u16, u16, u16),
    hu_values: f64,
) -> std::io::Result<()> {
    let (nx, ny, nz) = dims;
    let voxels: Vec<f32> = (0..nx as usize * ny as usize * nz as usize)
        .map(|_| hu_values as f32)
        .collect();

    let origin = Point::default();
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    let image = ritk_image::native::Image::from_flat_on(
        voxels,
        [nx as usize, ny as usize, nz as usize],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .map_err(|error| std::io::Error::other(error.to_string()))?;

    NiftiWriter::new(SequentialBackend)
        .write(path, &image)
        .map_err(|e| std::io::Error::other(e.to_string()))
}

/// Helper to create a realistic skull phantom NIFTI via RITK writer
fn create_skull_phantom_nifti(path: &str) -> std::io::Result<()> {
    let dims = (64, 64, 64);
    let mut volume = Array3::zeros(dims);

    let center = (32.0, 32.0, 32.0);
    let outer_radius = 28.0;
    let inner_radius = 24.0;

    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                let r = ((i as f64 - center.0).powi(2)
                    + (j as f64 - center.1).powi(2)
                    + (k as f64 - center.2).powi(2))
                .sqrt();

                let hu = if r > outer_radius {
                    -1000.0 // Air
                } else if r > inner_radius {
                    // Skull with gradient (cortical to trabecular)
                    let t = (r - inner_radius) / (outer_radius - inner_radius);
                    1200.0 + t * 600.0 // 1200-1800 HU
                } else {
                    40.0 // Brain tissue
                };

                volume[[i, j, k]] = hu;
            }
        }
    }

    let voxels: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let origin = Point::new([-16.0, -16.0, -16.0]); // center offset for 0.5mm spacing
    let spacing = Spacing::new([0.5, 0.5, 0.5]);
    let direction = Direction::identity();

    let image = ritk_image::native::Image::from_flat_on(
        voxels,
        [dims.0, dims.1, dims.2],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .map_err(|error| std::io::Error::other(error.to_string()))?;

    NiftiWriter::new(SequentialBackend)
        .write(path, &image)
        .map_err(|e| std::io::Error::other(e.to_string()))
}

fn get_test_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ct_scans")
}

fn remove_test_file(path: &Path) {
    fs::remove_file(path).expect("remove NIfTI integration fixture");
}

#[test]
fn test_load_synthetic_nifti_soft_tissue() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("synthetic_soft_tissue.nii");
    create_synthetic_nifti(nifti_path.to_str().unwrap(), (32, 32, 32), 50.0).unwrap();

    let reader = NiftiReader::new(SequentialBackend);
    let image = reader.read(nifti_path.to_str().unwrap()).unwrap();

    let shape = image.shape();
    assert_eq!(shape, [32, 32, 32]);

    let spacing = image.spacing();
    assert_eq!(spacing[0], 1.0);
    assert_eq!(spacing[1], 1.0);
    assert_eq!(spacing[2], 1.0);

    let data = image.data_slice().expect("contiguous");
    let min = data.iter().copied().fold(f32::MAX, f32::min);
    let max = data.iter().copied().fold(f32::MIN, f32::max);
    assert!(
        (min - 50.0).abs() < 1e-6,
        "Expected HU ~50.0, got min {min}"
    );
    assert!(
        (max - 50.0).abs() < 1e-6,
        "Expected HU ~50.0, got max {max}"
    );

    remove_test_file(&nifti_path);
}

#[test]
fn test_load_synthetic_nifti_bone() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("synthetic_bone.nii");
    create_synthetic_nifti(nifti_path.to_str().unwrap(), (16, 16, 16), 1500.0).unwrap();

    let reader = NiftiReader::new(SequentialBackend);
    let image = reader.read(nifti_path.to_str().unwrap()).unwrap();

    let shape = image.shape();
    assert_eq!(shape, [16, 16, 16]);

    let data = image.data_slice().expect("contiguous");
    let max = data.iter().copied().fold(f32::MIN, f32::max);
    assert!(max >= 1500.0, "Should detect bone HU values, max={max}");

    remove_test_file(&nifti_path);
}

#[test]
fn test_load_skull_phantom() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("skull_phantom.nii");
    create_skull_phantom_nifti(nifti_path.to_str().unwrap()).unwrap();

    let reader = NiftiReader::new(SequentialBackend);
    let image = reader.read(nifti_path.to_str().unwrap()).unwrap();

    let shape = image.shape();
    assert_eq!(shape, [64, 64, 64]);

    let spacing = image.spacing();
    assert_eq!(spacing[0], 0.5);
    assert_eq!(spacing[1], 0.5);
    assert_eq!(spacing[2], 0.5);

    let data = image.data_slice().expect("contiguous");
    let min = data.iter().copied().fold(f32::MAX, f32::min);
    let max = data.iter().copied().fold(f32::MIN, f32::max);
    assert!(min < 0.0, "Should have air regions, min={min}");
    assert!(max > 1000.0, "Should have bone regions, max={max}");

    remove_test_file(&nifti_path);
}

#[test]
fn test_nifti_affine_transformation() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("affine_test.nii");

    // Create NIFTI with non-identity affine
    let dims = (10, 10, 10);
    let volume = Array3::from_elem(dims, 100.0f64);
    let voxels: Vec<f32> = volume.iter().map(|&v| v as f32).collect();

    let origin = Point::new([-10.0, -10.0, -10.0]); // 2mm spacing, 10mm offset
    let spacing = Spacing::new([2.0, 2.0, 2.0]);
    let direction = Direction::identity();

    let image = ritk_image::native::Image::from_flat_on(
        voxels,
        [dims.0, dims.1, dims.2],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .unwrap();

    NiftiWriter::new(SequentialBackend)
        .write(nifti_path.to_str().unwrap(), &image)
        .unwrap();

    let reader = NiftiReader::new(SequentialBackend);
    let loaded = reader.read(nifti_path.to_str().unwrap()).unwrap();

    let loaded_spacing = loaded.spacing();
    assert_eq!(loaded_spacing[0], 2.0);
    assert_eq!(loaded_spacing[1], 2.0);
    assert_eq!(loaded_spacing[2], 2.0);

    let loaded_origin = loaded.origin();
    assert!((loaded_origin[0] - (-10.0)).abs() < 1e-6);
    assert!((loaded_origin[1] - (-10.0)).abs() < 1e-6);
    assert!((loaded_origin[2] - (-10.0)).abs() < 1e-6);

    remove_test_file(&nifti_path);
}

#[test]
fn test_compressed_nifti_gz() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("compressed.nii.gz");
    create_synthetic_nifti(nifti_path.to_str().unwrap(), (20, 20, 20), 800.0).unwrap();

    let reader = NiftiReader::new(SequentialBackend);
    let image = reader.read(nifti_path.to_str().unwrap()).unwrap();

    let shape = image.shape();
    assert_eq!(shape, [20, 20, 20]);

    let data = image.data_slice().expect("contiguous");
    let min = data.iter().copied().fold(f32::MAX, f32::min);
    let max = data.iter().copied().fold(f32::MIN, f32::max);
    assert!(
        (min - 800.0).abs() < 1e-6,
        "Expected HU ~800.0, got min {min}"
    );
    assert!(
        (max - 800.0).abs() < 1e-6,
        "Expected HU ~800.0, got max {max}"
    );

    remove_test_file(&nifti_path);
}

#[test]
fn test_invalid_file_format() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let bad_path = test_dir.join("not_a_nifti.txt");
    fs::write(&bad_path, b"This is not a NIFTI file").unwrap();

    let reader = NiftiReader::new(SequentialBackend);
    let error = reader
        .read(bad_path.to_str().unwrap())
        .expect_err("non-NIfTI input must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::Other);

    remove_test_file(&bad_path);
}

#[test]
fn test_roundtrip_accuracy() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    // Create known HU distribution
    let mut original_ct = Array3::zeros((32, 32, 32));
    for i in 0..32 {
        for j in 0..32 {
            for k in 0..32 {
                let value = (i + j + k) as f64 * 10.0; // Gradient 0-930
                original_ct[[i, j, k]] = value;
            }
        }
    }

    let nifti_path = test_dir.join("roundtrip.nii");

    // Write to NIFTI via RITK writer
    let voxels: Vec<f32> = original_ct.iter().map(|&v| v as f32).collect();
    let origin = Point::default();
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    let image = ritk_image::native::Image::from_flat_on(
        voxels,
        [32, 32, 32],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .unwrap();

    NiftiWriter::new(SequentialBackend)
        .write(nifti_path.to_str().unwrap(), &image)
        .unwrap();

    // Read back via RITK reader
    let reader = NiftiReader::new(SequentialBackend);
    let loaded = reader.read(nifti_path.to_str().unwrap()).unwrap();
    let loaded_data = loaded.data_slice().expect("contiguous");

    // Check all values match (within floating point tolerance)
    for i in 0..32 {
        for j in 0..32 {
            for k in 0..32 {
                let idx = i * 32 * 32 + j * 32 + k;
                assert_eq!(
                    loaded_data[idx] as f64,
                    original_ct[[i, j, k]],
                    "roundtrip value differs at [{i}, {j}, {k}]"
                );
            }
        }
    }

    remove_test_file(&nifti_path);
}
