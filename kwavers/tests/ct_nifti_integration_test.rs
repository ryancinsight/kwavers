//! Integration tests for NIFTI file I/O
//!
//! These tests require the 'nifti' feature to be enabled.
//! Run with: cargo test --features nifti

#![cfg(feature = "nifti")]

use kwavers::core::error::KwaversError;
use kwavers::domain::grid::Grid;
use kwavers::physics::skull::CTBasedSkullModel;
use ndarray::Array3;
use nifti::{writer::WriterOptions, InMemNiftiObject, NiftiHeader};
use std::fs;
use std::path::PathBuf;

/// Helper to create synthetic NIFTI file for testing
fn create_synthetic_nifti(
    path: &str,
    dims: (u16, u16, u16),
    hu_values: f64,
) -> std::io::Result<()> {
    // Create a simple 3D volume
    let (nx, ny, nz) = dims;
    let volume = Array3::from_elem((nx as usize, ny as usize, nz as usize), hu_values);

    // Create NIFTI header
    let mut header = NiftiHeader::default();
    header.dim[0] = 3; // 3D
    header.dim[1] = nx;
    header.dim[2] = ny;
    header.dim[3] = nz;
    header.pixdim[1] = 1.0; // 1mm isotropic
    header.pixdim[2] = 1.0;
    header.pixdim[3] = 1.0;
    header.datatype = 16; // f32
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;

    // Set sform matrix (identity with 1mm spacing)
    header.sform_code = 1;
    header.srow_x = [1.0, 0.0, 0.0, 0.0];
    header.srow_y = [0.0, 1.0, 0.0, 0.0];
    header.srow_z = [0.0, 0.0, 1.0, 0.0];

    // Write NIFTI file
    let nifti = InMemNiftiObject::from_header_and_data(header, volume);
    WriterOptions::new(path).write_nifti(&nifti)?;

    Ok(())
}

/// Helper to create realistic skull phantom NIFTI
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

    // Create header
    let mut header = NiftiHeader::default();
    header.dim[0] = 3;
    header.dim[1] = dims.0 as u16;
    header.dim[2] = dims.1 as u16;
    header.dim[3] = dims.2 as u16;
    header.pixdim[1] = 0.5; // 0.5mm isotropic (high resolution)
    header.pixdim[2] = 0.5;
    header.pixdim[3] = 0.5;
    header.datatype = 16; // f32
    header.sform_code = 1;
    header.srow_x = [0.5, 0.0, 0.0, 0.0];
    header.srow_y = [0.0, 0.5, 0.0, 0.0];
    header.srow_z = [0.0, 0.0, 0.5, 0.0];

    let nifti = NiftiObject::from_header_and_data(header, volume);
    WriterOptions::new(path).write_nifti(&nifti)?;

    Ok(())
}

fn get_test_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ct_scans")
}

#[test]
fn test_load_synthetic_nifti_soft_tissue() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("synthetic_soft_tissue.nii");
    create_synthetic_nifti(nifti_path.to_str().unwrap(), (32, 32, 32), 50.0).unwrap();

    let model = CTBasedSkullModel::from_file(nifti_path.to_str().unwrap()).unwrap();
    let metadata = model.metadata();

    assert_eq!(metadata.dimensions, (32, 32, 32));
    assert_eq!(metadata.hu_range, (50.0, 50.0));
    assert_eq!(metadata.voxel_spacing_mm, (1.0, 1.0, 1.0));

    // Cleanup
    fs::remove_file(nifti_path).ok();
}

#[test]
fn test_load_synthetic_nifti_bone() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("synthetic_bone.nii");
    create_synthetic_nifti(nifti_path.to_str().unwrap(), (16, 16, 16), 1500.0).unwrap();

    let model = CTBasedSkullModel::from_file(nifti_path.to_str().unwrap()).unwrap();
    let metadata = model.metadata();

    assert_eq!(metadata.dimensions, (16, 16, 16));
    assert!(
        metadata.hu_range.1 >= 1500.0,
        "Should detect bone HU values"
    );

    // Verify sound speed calculation
    let c = model.sound_speed(8, 8, 8);
    assert!(c > 2800.0, "Bone should have high sound speed");

    // Cleanup
    fs::remove_file(nifti_path).ok();
}

#[test]
fn test_load_skull_phantom() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("skull_phantom.nii");
    create_skull_phantom_nifti(nifti_path.to_str().unwrap()).unwrap();

    let model = CTBasedSkullModel::from_file(nifti_path.to_str().unwrap()).unwrap();
    let metadata = model.metadata();

    assert_eq!(metadata.dimensions, (64, 64, 64));
    assert_eq!(metadata.voxel_spacing_mm, (0.5, 0.5, 0.5));
    assert!(metadata.hu_range.0 < 0.0, "Should have air regions");
    assert!(metadata.hu_range.1 > 1000.0, "Should have bone regions");

    // Generate mask and verify skull detection
    let grid = Grid::new(64, 64, 64, 0.5e-3, 0.5e-3, 0.5e-3).unwrap();
    let mask = model.generate_mask(&grid).unwrap();

    // Center should be brain (no mask)
    assert_eq!(mask[[32, 32, 32]], 0.0, "Brain center should not be masked");

    // Count skull voxels
    let skull_count = mask.iter().filter(|&&v| v > 0.5).count();
    assert!(skull_count > 1000, "Should detect substantial skull region");
    assert!(
        skull_count < 64 * 64 * 64 / 2,
        "Skull should not be majority"
    );

    // Generate heterogeneous model
    let het = model.to_heterogeneous(&grid).unwrap();

    // Verify acoustic properties vary appropriately
    let c_center = het.sound_speed[[32, 32, 32]]; // Brain
    let c_skull = het.sound_speed[[50, 32, 32]]; // Skull region

    assert_eq!(c_center, 1500.0, "Brain should have water-like sound speed");
    assert!(c_skull > 2800.0, "Skull should have bone-like sound speed");

    // Cleanup
    fs::remove_file(nifti_path).ok();
}

#[test]
fn test_nifti_affine_transformation() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("affine_test.nii");

    // Create NIFTI with non-identity affine
    let dims = (10, 10, 10);
    let volume = Array3::from_elem(dims, 100.0);

    let mut header = NiftiHeader::default();
    header.dim[0] = 3;
    header.dim[1] = dims.0 as u16;
    header.dim[2] = dims.1 as u16;
    header.dim[3] = dims.2 as u16;
    header.pixdim[1] = 2.0; // 2mm spacing
    header.pixdim[2] = 2.0;
    header.pixdim[3] = 2.0;
    header.datatype = 16;
    header.sform_code = 1;
    header.srow_x = [2.0, 0.0, 0.0, 10.0]; // 2mm spacing, 10mm offset
    header.srow_y = [0.0, 2.0, 0.0, 20.0];
    header.srow_z = [0.0, 0.0, 2.0, 30.0];

    let nifti = NiftiObject::from_header_and_data(header, volume);
    WriterOptions::new(nifti_path.to_str().unwrap())
        .write_nifti(&nifti)
        .unwrap();

    let model = CTBasedSkullModel::from_file(nifti_path.to_str().unwrap()).unwrap();
    let metadata = model.metadata();

    // Check voxel spacing extracted correctly
    assert_eq!(metadata.voxel_spacing_mm, (2.0, 2.0, 2.0));

    // Check affine matrix includes offset
    assert_eq!(metadata.affine[0][3], 10.0);
    assert_eq!(metadata.affine[1][3], 20.0);
    assert_eq!(metadata.affine[2][3], 30.0);

    // Cleanup
    fs::remove_file(nifti_path).ok();
}

#[test]
fn test_compressed_nifti_gz() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("compressed.nii.gz");
    create_synthetic_nifti(nifti_path.to_str().unwrap(), (20, 20, 20), 800.0).unwrap();

    // Should handle .nii.gz files automatically
    let model = CTBasedSkullModel::from_file(nifti_path.to_str().unwrap()).unwrap();
    let metadata = model.metadata();

    assert_eq!(metadata.dimensions, (20, 20, 20));
    assert_eq!(metadata.hu_range, (800.0, 800.0));

    // Cleanup
    fs::remove_file(nifti_path).ok();
}

#[test]
fn test_invalid_file_format() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    // Create a non-NIFTI file
    let bad_path = test_dir.join("not_a_nifti.txt");
    fs::write(&bad_path, b"This is not a NIFTI file").unwrap();

    let result = CTBasedSkullModel::from_file(bad_path.to_str().unwrap());
    assert!(result.is_err(), "Should fail to load non-NIFTI file");

    if let Err(KwaversError::InvalidInput(msg)) = result {
        assert!(
            msg.contains("Failed to read"),
            "Error should mention read failure"
        );
    }

    // Cleanup
    fs::remove_file(bad_path).ok();
}

#[test]
fn test_nifti_dimension_validation() {
    let test_dir = get_test_dir();
    fs::create_dir_all(&test_dir).unwrap();

    let nifti_path = test_dir.join("4d_volume.nii");

    // Create a 4D volume (should be rejected as we expect 3D)
    let volume = Array3::from_elem((10, 10, 10), 100.0);
    let mut header = NiftiHeader::default();
    header.dim[0] = 4; // 4D (WRONG)
    header.dim[1] = 10;
    header.dim[2] = 10;
    header.dim[3] = 10;
    header.dim[4] = 5; // time points

    let nifti = NiftiObject::from_header_and_data(header, volume);
    WriterOptions::new(nifti_path.to_str().unwrap())
        .write_nifti(&nifti)
        .unwrap();

    let result = CTBasedSkullModel::from_file(nifti_path.to_str().unwrap());
    assert!(result.is_err(), "Should reject 4D volumes");

    if let Err(KwaversError::Validation(_)) = result {
        // Expected validation error
    } else {
        panic!("Expected ValidationError for non-3D volume");
    }

    // Cleanup
    fs::remove_file(nifti_path).ok();
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

    // Write to NIFTI
    let mut header = NiftiHeader::default();
    header.dim[0] = 3;
    header.dim[1] = 32;
    header.dim[2] = 32;
    header.dim[3] = 32;
    header.pixdim[1] = 1.0;
    header.pixdim[2] = 1.0;
    header.pixdim[3] = 1.0;
    header.datatype = 16;

    let nifti = NiftiObject::from_header_and_data(header, original_ct.clone());
    WriterOptions::new(nifti_path.to_str().unwrap())
        .write_nifti(&nifti)
        .unwrap();

    // Read back
    let model = CTBasedSkullModel::from_file(nifti_path.to_str().unwrap()).unwrap();
    let loaded_ct = model.ct_data();

    // Check all values match (within floating point tolerance)
    for i in 0..32 {
        for j in 0..32 {
            for k in 0..32 {
                let diff = (original_ct[[i, j, k]] - loaded_ct[[i, j, k]]).abs();
                assert!(diff < 1e-6, "Roundtrip values should match exactly");
            }
        }
    }

    // Cleanup
    fs::remove_file(nifti_path).ok();
}

// Cleanup after all tests
#[test]
#[ignore] // Run manually to clean up test data
fn cleanup_test_data() {
    let test_dir = get_test_dir();
    if test_dir.exists() {
        fs::remove_dir_all(test_dir).ok();
    }
}
