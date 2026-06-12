use super::loader::CTImageLoader;
use crate::medical::MedicalImageLoader;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

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
    // Schneider 1996: c(1500) = 1500 + 0.76·1500 = 2640 m/s — within the
    // 1996–3114 m/s skull range measured by Webb (2018). The previous binary
    // ramp gave an unphysical 4400 m/s; that assertion was analytically wrong.
    let c_bone = CTImageLoader::hu_to_sound_speed(1500.0);
    assert!((2500.0..=3100.0).contains(&c_bone), "skull speed {c_bone} out of band");
    let c_very_dense = CTImageLoader::hu_to_sound_speed(2500.0);
    assert!(c_very_dense > c_bone, "speed must increase with HU");
}

#[test]
fn test_hu_to_sound_speed_soft_tissue_is_resolved() {
    // Water anchor exact; muscle (HU = 100) is DISTINCT from and faster than
    // water (1576 vs 1500 m/s). The old model collapsed all soft tissue to
    // water — the bug this change fixes — so `c(100) == c_water` was incorrect.
    let c_water = CTImageLoader::hu_to_sound_speed(0.0);
    assert!((c_water - SOUND_SPEED_WATER_SIM).abs() < 1e-6);
    let c_muscle = CTImageLoader::hu_to_sound_speed(100.0);
    assert!((c_muscle - 1576.0).abs() < 1e-6, "muscle speed {c_muscle} != 1576");
    assert!(c_muscle > c_water);
}

#[test]
fn test_hu_to_density_bone() {
    // Schneider: ρ(1000) = 1000 + 0.96·1000 = 1960 kg/m³ (cortical band).
    let rho_bone = CTImageLoader::hu_to_density(1000.0);
    assert!((1900.0..=2000.0).contains(&rho_bone), "bone density {rho_bone}");
    let rho_very_dense = CTImageLoader::hu_to_density(2000.0);
    assert!(rho_very_dense > rho_bone);
}

#[test]
fn test_hu_to_density_soft_tissue_is_resolved() {
    // Water anchor exact; muscle (HU = 50) is DISTINCT from water (1048 vs
    // 1000 kg/m³). The old model forced ρ(50) == ρ_water, erasing tissue
    // contrast — an analytically incorrect assertion now corrected.
    let rho_water = CTImageLoader::hu_to_density(0.0);
    assert_eq!(rho_water, DENSITY_WATER_NOMINAL);
    let rho_muscle = CTImageLoader::hu_to_density(50.0);
    assert!((rho_muscle - 1048.0).abs() < 1e-6, "muscle density {rho_muscle} != 1048");
    assert!(rho_muscle > rho_water);
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
