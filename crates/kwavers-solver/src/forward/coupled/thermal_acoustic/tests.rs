use super::*;

#[test]
fn test_coupler_creation() {
    let c = ThermalAcousticCoupler::new_default();
    assert_eq!(c.step_count(), 0);
    assert_eq!(c.total_time(), 0.0);
}

#[test]
fn test_config_validation_negative_dt() {
    let mut config = ThermalAcousticConfig::default();
    config.dt = -0.001;
    let result = ThermalAcousticCoupler::new(config);
    assert!(result.is_err());
}

#[test]
fn test_config_validation_cfl_acoustic() {
    let mut config = ThermalAcousticConfig::default();
    config.dt = 0.01; // Way too large
    let result = ThermalAcousticCoupler::new(config);
    assert!(result.is_err());
}

#[test]
fn test_single_step() {
    let mut coupler = ThermalAcousticCoupler::new_default();
    coupler.step().unwrap();
    assert_eq!(coupler.step_count(), 1);
    assert!(coupler.total_time() > 0.0);
}

#[test]
fn test_multiple_steps() {
    let mut coupler = ThermalAcousticCoupler::new_default();
    for _ in 0..10 {
        coupler.step().unwrap();
    }
    assert_eq!(coupler.step_count(), 10);
}

#[test]
fn test_temperature_bounded() {
    let mut coupler = ThermalAcousticCoupler::new_default();
    for _ in 0..5 {
        if coupler.step().is_err() {
            break;
        }
    }
    for &t in coupler.temperature().iter() {
        assert!(
            (0.0..=100.0).contains(&t),
            "Temperature out of bounds: {}",
            t
        );
    }
}

#[test]
fn test_acoustic_heating_nonnegative() {
    let mut coupler = ThermalAcousticCoupler::new_default();
    let _ = coupler.step();

    for &q in coupler.acoustic_heating().iter() {
        assert!(q >= 0.0, "Acoustic heating should be non-negative: {}", q);
    }
}
