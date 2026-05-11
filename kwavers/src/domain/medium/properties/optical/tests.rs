use super::OpticalPropertyData;

#[test]
fn test_optical_total_attenuation() {
    let props = OpticalPropertyData::new(10.0, 100.0, 0.9, 1.4).unwrap();
    assert_eq!(props.total_attenuation(), 110.0);
}

#[test]
fn test_optical_reduced_scattering() {
    let props = OpticalPropertyData::new(10.0, 100.0, 0.9, 1.4).unwrap();
    assert!((props.reduced_scattering() - 10.0).abs() < 1e-12);
}

#[test]
fn test_optical_albedo() {
    let props = OpticalPropertyData::new(10.0, 90.0, 0.8, 1.4).unwrap();
    assert!((props.albedo() - 0.9).abs() < 1e-10);
}

#[test]
fn test_optical_mean_free_path() {
    let props = OpticalPropertyData::new(1.0, 99.0, 0.9, 1.4).unwrap();
    assert_eq!(props.mean_free_path(), 0.01);
}

#[test]
fn test_optical_fresnel_reflectance() {
    let water = OpticalPropertyData::water();
    let reflectance = water.fresnel_reflectance_normal();
    assert!((reflectance - 0.02).abs() < 0.001);
}

#[test]
fn test_optical_validation() {
    assert!(OpticalPropertyData::new(-1.0, 100.0, 0.9, 1.4).is_err());
    assert!(OpticalPropertyData::new(10.0, 100.0, 1.5, 1.4).is_err());
    assert!(OpticalPropertyData::new(10.0, 100.0, 0.9, 0.5).is_err());
    let op = OpticalPropertyData::new(10.0, 100.0, 0.9, 1.4).unwrap();
    assert!(op.absorption_coefficient > 0.0);
}

#[test]
fn test_optical_presets() {
    let water = OpticalPropertyData::water();
    let tissue = OpticalPropertyData::soft_tissue();
    let blood = OpticalPropertyData::blood_oxygenated();

    assert!(water.absorption_coefficient < tissue.absorption_coefficient);
    assert!(blood.scattering_coefficient > water.scattering_coefficient);
    assert!(tissue.anisotropy > 0.8);
}

#[test]
fn test_optical_penetration_depth() {
    let props = OpticalPropertyData::soft_tissue();
    let depth = props.penetration_depth();

    assert!(depth > 0.0);
    assert!(depth < 1.0);
}
