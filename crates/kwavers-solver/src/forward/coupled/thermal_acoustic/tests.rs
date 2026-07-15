use super::*;

fn assert_close(actual: f64, expected: f64) {
    let bound = 8.0 * f64::EPSILON * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= bound,
        "actual {actual} differs from expected {expected} by more than {bound}"
    );
}

#[test]
fn test_coupler_creation() {
    let c = ThermalAcousticCoupler::new_default();
    assert_eq!(c.step_count(), 0);
    assert_eq!(c.total_time(), 0.0);
}

#[test]
fn test_config_validation_negative_dt() {
    let config = ThermalAcousticConfig {
        dt: -0.001,
        ..Default::default()
    };
    let result = ThermalAcousticCoupler::new(config);
    assert!(result.is_err());
}

#[test]
fn test_config_validation_cfl_acoustic() {
    let config = ThermalAcousticConfig {
        dt: 0.01, // Way too large
        ..Default::default()
    };
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

#[test]
fn material_properties_follow_temperature_law() {
    let config = ThermalAcousticConfig {
        nx: 3,
        ny: 3,
        nz: 3,
        ..ThermalAcousticConfig::default()
    };
    let mut coupler = ThermalAcousticCoupler::new(config).unwrap();
    coupler.temperature[[1, 1, 1]] = config.t_ref + 2.0;

    coupler.update_material_properties();

    assert_close(
        coupler.sound_speed[[1, 1, 1]],
        config.dc_d_t.mul_add(2.0, config.c_ref),
    );
    assert_close(
        coupler.density[[1, 1, 1]],
        config.drho_d_t.mul_add(2.0, config.rho_ref),
    );
}

#[test]
fn acoustic_heating_uses_pressure_density_and_sound_speed() {
    let config = ThermalAcousticConfig {
        nx: 3,
        ny: 3,
        nz: 3,
        alpha_ac: 0.25,
        ..ThermalAcousticConfig::default()
    };
    let mut coupler = ThermalAcousticCoupler::new(config).unwrap();
    coupler.pressure_prev[[1, 1, 1]] = 2.0;
    coupler.density[[1, 1, 1]] = 4.0;
    coupler.sound_speed[[1, 1, 1]] = 5.0;

    coupler.compute_acoustic_heating();

    assert_close(coupler.acoustic_heating[[1, 1, 1]], 0.1);
    assert_eq!(coupler.acoustic_heating[[0, 0, 0]], 0.0);
}
