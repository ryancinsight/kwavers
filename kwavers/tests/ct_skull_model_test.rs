//! Comprehensive tests for CT-based skull modeling
//!
//! Tests cover:
//! - Positive cases: valid CT data loading and processing
//! - Negative cases: invalid inputs, missing files, corrupted data
//! - Integration: skull model generation and acoustic property conversion

use kwavers::core::error::KwaversError;
use kwavers::domain::grid::Grid;
use kwavers::physics::skull::CTBasedSkullModel;
use ndarray::Array3;

#[test]
fn test_from_ct_data_valid_synthetic() {
    // Create synthetic CT data with valid HU values
    let mut ct_data = Array3::zeros((64, 64, 64));

    // Create a spherical skull structure
    let center = (32.0, 32.0, 32.0);
    let inner_radius = 25.0;
    let outer_radius = 30.0;

    for i in 0..64 {
        for j in 0..64 {
            for k in 0..64 {
                let r = ((i as f64 - center.0).powi(2)
                    + (j as f64 - center.1).powi(2)
                    + (k as f64 - center.2).powi(2))
                .sqrt();

                if r >= inner_radius && r <= outer_radius {
                    // Skull bone: 1500 HU (cortical bone)
                    ct_data[[i, j, k]] = 1500.0;
                } else if r < inner_radius {
                    // Brain: 40 HU (soft tissue)
                    ct_data[[i, j, k]] = 40.0;
                } else {
                    // Air/background: -1000 HU
                    ct_data[[i, j, k]] = -1000.0;
                }
            }
        }
    }

    let model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();
    let metadata = model.metadata();

    assert_eq!(metadata.dimensions, (64, 64, 64));
    assert!(metadata.hu_range.0 >= -1000.0);
    assert!(metadata.hu_range.1 >= 1500.0);
}

#[test]
fn test_from_ct_data_with_metadata() {
    // Create minimal valid CT data
    let ct_data = Array3::from_elem((32, 32, 32), 100.0); // Soft tissue

    let voxel_spacing = (0.5e-3, 0.5e-3, 0.5e-3); // 0.5 mm isotropic
    let affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let model =
        CTBasedSkullModel::from_ct_data_with_metadata(&ct_data, voxel_spacing, affine).unwrap();

    let metadata = model.metadata();
    assert_eq!(metadata.voxel_spacing_m, voxel_spacing);
    assert_eq!(metadata.voxel_spacing_mm, (0.5, 0.5, 0.5));
}

#[test]
fn test_hu_range_validation_too_low() {
    // Invalid HU values (below air)
    let ct_data = Array3::from_elem((16, 16, 16), -3000.0);

    let result = CTBasedSkullModel::from_ct_data(&ct_data);
    assert!(result.is_err());

    if let Err(KwaversError::Validation(_)) = result {
        // Expected validation error
    } else {
        panic!("Expected ValidationError for HU values too low");
    }
}

#[test]
fn test_hu_range_validation_too_high() {
    // Invalid HU values (above dense bone)
    let ct_data = Array3::from_elem((16, 16, 16), 5000.0);

    let result = CTBasedSkullModel::from_ct_data(&ct_data);
    assert!(result.is_err());

    if let Err(KwaversError::Validation(_)) = result {
        // Expected validation error
    } else {
        panic!("Expected ValidationError for HU values too high");
    }
}

#[test]
fn test_generate_mask_skull_detection() {
    // Create CT data with clear skull structure
    let mut ct_data = Array3::zeros((32, 32, 32));

    // Add skull voxels (HU > 700)
    for i in 10..20 {
        for j in 10..20 {
            for k in 10..20 {
                ct_data[[i, j, k]] = 1200.0; // Bone
            }
        }
    }

    let model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let mask = model.generate_mask(&grid).unwrap();

    // Check skull voxels are detected
    for i in 10..20 {
        for j in 10..20 {
            for k in 10..20 {
                assert_eq!(mask[[i, j, k]], 1.0, "Skull voxel should be masked");
            }
        }
    }

    // Check non-skull voxels are not detected
    assert_eq!(mask[[0, 0, 0]], 0.0, "Non-skull voxel should not be masked");
}

#[test]
fn test_to_heterogeneous_acoustic_properties() {
    // Create CT data with bone and soft tissue
    let mut ct_data = Array3::zeros((32, 32, 32));

    // Skull region: 1500 HU
    ct_data[[15, 15, 15]] = 1500.0;

    // Soft tissue region: 50 HU
    ct_data[[5, 5, 5]] = 50.0;

    let model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let het = model.to_heterogeneous(&grid).unwrap();

    // Check bone acoustic properties
    // c_skull(HU) = 2800 + (HU - 700) × 0.5
    let expected_c_bone = 2800.0 + (1500.0 - 700.0) * 0.5;
    assert!((het.sound_speed[[15, 15, 15]] - expected_c_bone).abs() < 1.0);

    // Check soft tissue properties (should be water-like)
    assert_eq!(het.sound_speed[[5, 5, 5]], 1500.0);
    assert_eq!(het.density[[5, 5, 5]], 1000.0);
}

#[test]
fn test_sound_speed_at_voxel() {
    let mut ct_data = Array3::zeros((16, 16, 16));
    ct_data[[8, 8, 8]] = 1000.0; // Bone
    ct_data[[4, 4, 4]] = 50.0; // Soft tissue

    let model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();

    // Bone voxel
    let c_bone = model.sound_speed(8, 8, 8);
    let expected = 2800.0 + (1000.0 - 700.0) * 2.0;
    assert_eq!(c_bone, expected);

    // Soft tissue voxel
    let c_tissue = model.sound_speed(4, 4, 4);
    assert_eq!(c_tissue, 1500.0);
}

#[test]
fn test_ct_data_accessor() {
    let ct_data = Array3::from_elem((8, 8, 8), 100.0);
    let model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();

    let retrieved = model.ct_data();
    assert_eq!(retrieved.shape(), &[8, 8, 8]);
    assert_eq!(retrieved[[0, 0, 0]], 100.0);
}

#[test]
fn test_metadata_extraction() {
    let ct_data = Array3::from_elem((10, 20, 30), 500.0);
    let voxel_spacing = (1e-3, 2e-3, 3e-3);
    let affine = [
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 10.0],
        [0.0, 0.0, 1.0, 15.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let model =
        CTBasedSkullModel::from_ct_data_with_metadata(&ct_data, voxel_spacing, affine).unwrap();

    let metadata = model.metadata();

    assert_eq!(metadata.dimensions, (10, 20, 30));
    assert_eq!(metadata.voxel_spacing_m, voxel_spacing);
    assert_eq!(metadata.voxel_spacing_mm, (1.0, 2.0, 3.0));
    assert_eq!(metadata.affine[0][3], 5.0);
    assert_eq!(metadata.hu_range, (500.0, 500.0));
}

// Negative test: File not found (when nifti feature is enabled)
#[test]
#[cfg(feature = "nifti")]
fn test_from_file_not_found() {
    let result = CTBasedSkullModel::from_file("/nonexistent/path/to/ct.nii.gz");
    assert!(result.is_err());

    if let Err(KwaversError::InvalidInput(msg)) = result {
        assert!(msg.contains("not found"));
    } else {
        panic!("Expected InvalidInput error for missing file");
    }
}

// Negative test: Feature not enabled
#[test]
#[cfg(not(feature = "nifti"))]
fn test_from_file_feature_disabled() {
    let result = CTBasedSkullModel::from_file("dummy.nii");
    assert!(result.is_err());

    if let Err(KwaversError::InvalidInput(msg)) = result {
        assert!(msg.contains("not enabled"));
    } else {
        panic!("Expected InvalidInput error when nifti feature disabled");
    }
}

#[test]
fn test_heterogeneous_skull_integration() {
    // Create realistic skull phantom
    let mut ct_data = Array3::zeros((48, 48, 48));

    let center = (24.0, 24.0, 24.0);
    let skull_radius = 18.0;
    let brain_radius = 15.0;

    for i in 0..48 {
        for j in 0..48 {
            for k in 0..48 {
                let r = ((i as f64 - center.0).powi(2)
                    + (j as f64 - center.1).powi(2)
                    + (k as f64 - center.2).powi(2))
                .sqrt();

                if r > skull_radius {
                    ct_data[[i, j, k]] = -1000.0; // Air
                } else if r > brain_radius {
                    // Gradient in skull (cortical to trabecular)
                    let t = (r - brain_radius) / (skull_radius - brain_radius);
                    ct_data[[i, j, k]] = 800.0 + t * 1000.0; // 800-1800 HU
                } else {
                    ct_data[[i, j, k]] = 40.0; // Brain tissue
                }
            }
        }
    }

    let model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();
    let grid = Grid::new(48, 48, 48, 0.5e-3, 0.5e-3, 0.5e-3).unwrap();

    // Generate heterogeneous skull
    let het_skull = model.to_heterogeneous(&grid).unwrap();

    // Validate acoustic impedance varies across skull
    // Center is at (24, 24, 24), skull_radius = 18, brain_radius = 15
    // Position (41, 24, 24): r = sqrt((41-24)^2) = 17 (in skull, near outer radius 18)
    // Position (39, 24, 24): r = sqrt((39-24)^2) = 15 (at brain-skull boundary)
    let z_outer = het_skull.impedance_at(41, 24, 24); // r = 17 (in skull, near outer)
    let z_inner = het_skull.impedance_at(40, 24, 24); // r = 16 (in skull, near inner)
    let z_brain = het_skull.impedance_at(24, 24, 24); // r = 0 (brain center)

    // Debug values
    let c_outer = het_skull.sound_speed[[41, 24, 24]];
    let c_inner = het_skull.sound_speed[[40, 24, 24]];
    let c_brain = het_skull.sound_speed[[24, 24, 24]];

    // Skull should have higher impedance than brain
    // Brain: ρ=1000, c=1500 → Z = 1.5e6
    // Skull (800 HU): ρ=1720, c=2850 → Z ≈ 4.9e6 (factor of 3.3x)
    assert_eq!(c_brain, 1500.0, "Brain should have water-like sound speed");
    assert!(
        c_outer > 2800.0,
        "Outer skull should have bone-like sound speed: got {}",
        c_outer
    );
    assert!(
        c_inner > 2800.0,
        "Inner skull should have bone-like sound speed: got {}",
        c_inner
    );

    assert!(
        z_outer > z_brain * 2.5,
        "Outer skull impedance ({:.2e}) should be much higher than brain ({:.2e})",
        z_outer,
        z_brain
    );
    assert!(
        z_inner > z_brain * 2.5,
        "Inner skull impedance ({:.2e}) should be much higher than brain ({:.2e})",
        z_inner,
        z_brain
    );

    // Impedance should vary across skull thickness due to HU gradient (800 → 1800 HU)
    assert!(
        (z_outer - z_inner).abs() > 5e4,
        "Impedance should vary across skull thickness: outer={:.2e}, inner={:.2e}",
        z_outer,
        z_inner
    );
}

#[test]
fn test_empty_array_handling() {
    // Edge case: 1x1x1 volume
    let ct_data = Array3::from_elem((1, 1, 1), 1000.0);
    let model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();

    let metadata = model.metadata();
    assert_eq!(metadata.dimensions, (1, 1, 1));
    assert_eq!(metadata.hu_range, (1000.0, 1000.0));
}

#[test]
fn test_large_volume_hu_range() {
    // Larger volume with varied HU values
    let mut ct_data = Array3::zeros((100, 100, 50));

    // Fill with realistic distribution
    for i in 0..100 {
        for j in 0..100 {
            for k in 0..50 {
                let hu = if (i + j + k) % 3 == 0 {
                    1500.0 // Bone
                } else if (i + j + k) % 3 == 1 {
                    50.0 // Soft tissue
                } else {
                    -100.0 // Fat/air
                };
                ct_data[[i, j, k]] = hu;
            }
        }
    }

    let model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();
    let metadata = model.metadata();

    assert_eq!(metadata.dimensions, (100, 100, 50));
    assert!(metadata.hu_range.0 <= -100.0);
    assert!(metadata.hu_range.1 >= 1500.0);
}

#[test]
fn test_cortical_trabecular_distinction() {
    // Test different bone densities
    let mut ct_data = Array3::zeros((20, 20, 20));

    ct_data[[5, 5, 5]] = 1800.0; // Dense cortical bone
    ct_data[[10, 10, 10]] = 900.0; // Trabecular bone
    ct_data[[15, 15, 15]] = 700.0; // Bone threshold

    let model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();
    let grid = Grid::new(20, 20, 20, 1e-3, 1e-3, 1e-3).unwrap();
    let het = model.to_heterogeneous(&grid).unwrap();

    // Cortical should have highest sound speed
    let c_cortical = het.sound_speed[[5, 5, 5]];
    let c_trabecular = het.sound_speed[[10, 10, 10]];
    let c_threshold = het.sound_speed[[15, 15, 15]];

    assert!(
        c_cortical > c_trabecular,
        "Cortical bone should have higher sound speed"
    );
    assert!(
        c_trabecular > c_threshold,
        "Trabecular bone should have higher sound speed than threshold"
    );

    // Density should follow same pattern
    let rho_cortical = het.density[[5, 5, 5]];
    let rho_trabecular = het.density[[10, 10, 10]];

    assert!(
        rho_cortical > rho_trabecular,
        "Cortical bone should have higher density"
    );
}

#[test]
fn test_mask_generation_boundary_cases() {
    // Test mask generation with exact threshold values
    let mut ct_data = Array3::zeros((16, 16, 16));

    ct_data[[8, 8, 8]] = 699.9; // Just below threshold
    ct_data[[8, 8, 9]] = 700.0; // At threshold
    ct_data[[8, 8, 10]] = 700.1; // Just above threshold

    let model = CTBasedSkullModel::from_ct_data(&ct_data).unwrap();
    let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
    let mask = model.generate_mask(&grid).unwrap();

    assert_eq!(mask[[8, 8, 8]], 0.0, "Below threshold should not be masked");
    assert_eq!(
        mask[[8, 8, 9]],
        0.0,
        "At threshold should not be masked (> not >=)"
    );
    assert_eq!(mask[[8, 8, 10]], 1.0, "Above threshold should be masked");
}
