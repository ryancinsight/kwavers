use super::loader::CTImageLoader;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::domain::imaging::medical::MedicalImageLoader;

#[test]
fn test_ct_loader_creation() {
    let loader = CTImageLoader::new();
    assert!(loader.data().is_none());
    assert!(loader.ct_metadata().is_none());
}

#[test]
fn test_ct_loader_metadata_default() {
    let loader = CTImageLoader::new();
    let metadata = loader.metadata();
    assert_eq!(metadata.dimensions, (0, 0, 0));
    assert_eq!(metadata.modality, "CT");
}

#[test]
fn test_ct_loader_name() {
    let loader = CTImageLoader::new();
    assert_eq!(loader.name(), "CT (NIFTI)");
    assert_eq!(loader.modality(), "CT");
}

#[test]
fn test_hu_to_sound_speed_bone() {
    let c_bone = CTImageLoader::hu_to_sound_speed(1500.0);
    assert!(c_bone > 3000.0);

    let c_very_dense = CTImageLoader::hu_to_sound_speed(2500.0);
    assert!(c_very_dense > c_bone);
}

#[test]
fn test_hu_to_sound_speed_soft_tissue() {
    let c_water = CTImageLoader::hu_to_sound_speed(0.0);
    assert!((c_water - SOUND_SPEED_WATER_SIM).abs() < 1e-6);

    let c_tissue = CTImageLoader::hu_to_sound_speed(100.0);
    assert_eq!(c_tissue, SOUND_SPEED_WATER_SIM);
}

#[test]
fn test_hu_to_density_bone() {
    let rho_bone = CTImageLoader::hu_to_density(1000.0);
    assert!(rho_bone > 1500.0);

    let rho_very_dense = CTImageLoader::hu_to_density(2000.0);
    assert!(rho_very_dense > rho_bone);
}

#[test]
fn test_hu_to_density_soft_tissue() {
    let rho_water = CTImageLoader::hu_to_density(0.0);
    assert_eq!(rho_water, 1000.0);

    let rho_tissue = CTImageLoader::hu_to_density(50.0);
    assert_eq!(rho_tissue, 1000.0);
}

#[test]
fn test_identity_affine_via_metadata() {
    // Unloaded loader returns identity affine via metadata()
    let loader = CTImageLoader::new();
    let affine = loader.metadata().affine;
    assert_eq!(affine[0][0], 1.0);
    assert_eq!(affine[1][1], 1.0);
    assert_eq!(affine[2][2], 1.0);
    assert_eq!(affine[3][3], 1.0);
    assert_eq!(affine[0][1], 0.0);
    assert_eq!(affine[3][0], 0.0);
}
