use super::properties::AcousticSkullProperties;
use super::simulation::TranscranialSimulation;
use aequitas::systems::si::quantities::{
    AcousticImpedance, Frequency, Length, MassDensity, ReciprocalLength, Velocity,
};
use aequitas::systems::si::units::Megahertz;
use kwavers_core::constants::acoustic_parameters::{BONE_DENSITY, SOUND_SPEED_SKULL_CORTICAL};
use kwavers_core::constants::fundamental::ACOUSTIC_IMPEDANCE_WATER_NOMINAL;
use kwavers_core::error::KwaversError;
use kwavers_grid::Grid;

#[test]
fn test_skull_properties_default() {
    let props = AcousticSkullProperties::default();
    assert_eq!(props.sound_speed().into_base(), SOUND_SPEED_SKULL_CORTICAL);
    assert_eq!(props.density().into_base(), BONE_DENSITY);
    assert_eq!(props.attenuation_at_one_megahertz().into_base(), 60.0);
    assert_eq!(props.thickness().into_base(), 0.007);
}

#[test]
fn test_bone_types() {
    let cortical = AcousticSkullProperties::from_bone_type("cortical").unwrap();
    let trabecular = AcousticSkullProperties::from_bone_type("trabecular").unwrap();

    assert!(cortical.sound_speed() > trabecular.sound_speed());
    assert!(cortical.density() > trabecular.density());
    assert!(cortical.shear_speed().is_some());
    assert!(AcousticSkullProperties::suture().shear_speed().is_none());
}

#[test]
fn test_acoustic_impedance() {
    let props = AcousticSkullProperties::default();
    let z = props.acoustic_impedance();
    assert_eq!(z.into_base(), BONE_DENSITY * SOUND_SPEED_SKULL_CORTICAL);
}

#[test]
fn test_transmission_coefficient() {
    let props = AcousticSkullProperties::default();
    let water_z = 1.5e6;
    let skull_z = BONE_DENSITY * SOUND_SPEED_SKULL_CORTICAL;
    let expected = 4.0 * water_z * skull_z / (water_z + skull_z).powi(2);
    let observed = props
        .transmission_coefficient(AcousticImpedance::from_base(water_z))
        .expect("positive impedance")
        .into_base();
    let bound = 16.0 * f64::EPSILON * expected;
    assert!((observed - expected).abs() <= bound);
}

#[test]
fn test_frequency_dependent_attenuation() {
    let props = AcousticSkullProperties::default();

    let atten_500k = props
        .attenuation_at_frequency(Frequency::from_unit::<Megahertz>(0.5))
        .expect("finite frequency");
    let atten_1m = props
        .attenuation_at_frequency(Frequency::from_unit::<Megahertz>(1.0))
        .expect("finite frequency");

    assert_eq!(atten_500k.into_base(), 30.0);
    assert_eq!(atten_1m.into_base(), 60.0);
}

#[test]
fn skull_properties_reject_invalid_physical_values() {
    let invalid = [
        AcousticSkullProperties::new(
            Velocity::from_base(0.0),
            MassDensity::from_base(BONE_DENSITY),
            ReciprocalLength::from_base(60.0),
            Length::from_base(0.007),
            None,
        ),
        AcousticSkullProperties::new(
            Velocity::from_base(SOUND_SPEED_SKULL_CORTICAL),
            MassDensity::from_base(f64::NAN),
            ReciprocalLength::from_base(60.0),
            Length::from_base(0.007),
            None,
        ),
        AcousticSkullProperties::new(
            Velocity::from_base(SOUND_SPEED_SKULL_CORTICAL),
            MassDensity::from_base(BONE_DENSITY),
            ReciprocalLength::from_base(-1.0),
            Length::from_base(0.007),
            None,
        ),
    ];

    for result in invalid {
        assert!(matches!(result, Err(KwaversError::InvalidInput(_))));
    }
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
    let mask = sim
        .skull_mask
        .as_ref()
        .expect("successful geometry construction stores the mask");
    assert_eq!(mask.shape(), [64, 64, 64]);
    let skull_voxels = mask.iter().filter(|&&value| value > 0.5).count();
    // The 7-cell-thick lattice shell contains integer offsets with
    // 13² <= x² + y² + z² <= 20².
    assert_eq!(skull_voxels, 24_308);
}

#[test]
fn test_insertion_loss_estimation() {
    let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001).unwrap();
    let props = AcousticSkullProperties::default();
    let frequency_megahertz = 0.65;
    let attenuation_np = props.attenuation_at_one_megahertz().into_base()
        * frequency_megahertz
        * props.thickness().into_base();
    let skull_impedance = props.acoustic_impedance().into_base();
    let impedance_sum = ACOUSTIC_IMPEDANCE_WATER_NOMINAL + skull_impedance;
    let interface_transmission =
        4.0 * ACOUSTIC_IMPEDANCE_WATER_NOMINAL * skull_impedance / impedance_sum.powi(2);
    let expected = (-attenuation_np).exp() * interface_transmission;

    let sim = TranscranialSimulation::new(&grid, props).unwrap();
    let loss = sim
        .estimate_insertion_loss(Frequency::from_base(650e3))
        .unwrap();

    // The closed form uses fewer than 16 rounded arithmetic operations. The
    // factor of four covers platform libm rounding in exp and operation order.
    let roundoff_bound = 64.0 * f64::EPSILON * expected.abs();
    assert!((loss - expected).abs() <= roundoff_bound);
}
