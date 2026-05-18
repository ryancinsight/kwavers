use super::properties::AcousticSkullProperties;
use super::simulation::TranscranialSimulation;
use crate::domain::grid::Grid;

#[test]
fn test_skull_properties_default() {
    let props = AcousticSkullProperties::default();
    assert!(props.sound_speed > 2800.0 && props.sound_speed < 3500.0);
    assert!(props.density > 1800.0 && props.density < 2100.0);
}

#[test]
fn test_bone_types() {
    let cortical = AcousticSkullProperties::from_bone_type("cortical").unwrap();
    let trabecular = AcousticSkullProperties::from_bone_type("trabecular").unwrap();

    assert!(cortical.sound_speed > trabecular.sound_speed);
    assert!(cortical.density > trabecular.density);
}

#[test]
fn test_acoustic_impedance() {
    let props = AcousticSkullProperties::default();
    let z = props.acoustic_impedance();

    // Skull impedance should be much higher than water (1.5 MRayl)
    assert!(z > 5.0e6);
    assert!(z < 8.0e6);
}

#[test]
fn test_transmission_coefficient() {
    let props = AcousticSkullProperties::default();
    let water_z = 1.5e6;
    let t = props.transmission_coefficient(water_z);

    // Should have significant loss (<50% transmission)
    assert!(t > 0.0 && t < 0.5);
}

#[test]
fn test_frequency_dependent_attenuation() {
    let props = AcousticSkullProperties::default();

    let atten_500k = props.attenuation_at_frequency(500e3);
    let atten_1m = props.attenuation_at_frequency(1e6);

    // Attenuation should increase with frequency
    assert!(atten_1m > atten_500k);
}

#[test]
fn test_transcranial_simulation_creation() {
    let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001).unwrap();
    let props = AcousticSkullProperties::default();

    let _sim = TranscranialSimulation::new(&grid, props).unwrap();
}

#[test]
fn test_analytical_sphere_geometry() {
    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
    let props = AcousticSkullProperties::default();

    let mut sim = TranscranialSimulation::new(&grid, props).unwrap();
    let result = sim.set_analytical_geometry("sphere", &[20.0]);

    result.unwrap();
    assert!(!sim.skull_mask.as_ref().unwrap().is_empty());
}

#[test]
fn test_insertion_loss_estimation() {
    let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001).unwrap();
    let props = AcousticSkullProperties::default();

    let sim = TranscranialSimulation::new(&grid, props).unwrap();
    let loss = sim.estimate_insertion_loss(650e3).unwrap();

    // Should have significant insertion loss
    assert!(loss > 0.1 && loss < 0.5);
}
