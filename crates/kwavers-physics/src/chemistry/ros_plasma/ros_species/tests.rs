use super::*;
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;

#[test]
fn test_ros_properties() {
    assert_eq!(ROSSpecies::HydroxylRadical.name(), "•OH");
    assert!(ROSSpecies::HydroxylRadical.reduction_potential() > 2.0);
    assert!(ROSSpecies::AtomicHydrogen.diffusion_coefficient() > 5e-9);
}

#[test]
fn test_ros_concentrations() {
    let mut ros = ROSConcentrations::new(10, 10, 10);

    // Set some concentration
    if let Some(oh) = ros.get_mut(ROSSpecies::HydroxylRadical) {
        oh[[5, 5, 5]] = 1e-6; // 1 μM
    }

    // Update total
    ros.update_total();
    assert!(ros.total_ros[[5, 5, 5]] > 0.0);

    // Test decay
    ros.apply_decay(1e-9); // 1 ns
    if let Some(oh) = ros.get(ROSSpecies::HydroxylRadical) {
        assert!(oh[[5, 5, 5]] < 1e-6); // Should have decayed
    }
}

#[test]
fn ros_decay_matches_species_lifetime_exponential() {
    let mut ros = ROSConcentrations::new(2, 1, 1);
    let initial = 2.5e-6;
    let species = ROSSpecies::HydrogenPeroxide;
    ros.get_mut(species)
        .expect("invariant: H2O2 field exists")
        .fill(initial);

    let dt = 0.25 * species.lifetime_water();
    ros.apply_decay(dt);

    let expected = initial * (-dt / species.lifetime_water()).exp();
    let field = ros.get(species).expect("invariant: H2O2 field exists");
    for &actual in field {
        assert!(
            (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-12,
            "decayed concentration {actual} should equal {expected}"
        );
    }
}

#[test]
fn test_ros_generation() {
    let rates = calculate_ros_generation(3000.0, ATMOSPHERIC_PRESSURE, 0.5);
    assert!(rates.contains_key(&ROSSpecies::HydroxylRadical));
    assert!(rates.contains_key(&ROSSpecies::AtomicOxygen));
}
