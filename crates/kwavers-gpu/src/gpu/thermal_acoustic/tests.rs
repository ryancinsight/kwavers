use super::config::GpuThermalAcousticConfig;

#[test]
fn test_config_validation() {
    let config = GpuThermalAcousticConfig::default();
    config.validate().unwrap();
}

#[test]
fn test_config_cfl_acoustic_violation() {
    let mut config = GpuThermalAcousticConfig::default();
    config.dt = 1.0;
    let err = config.validate().unwrap_err();
    assert!(format!("{err:?}").contains("acoustic"));
}

#[test]
fn test_config_cfl_thermal_violation() {
    let mut config = GpuThermalAcousticConfig::default();
    config.alpha_thermal = 30.0;
    let err = config.validate().unwrap_err();
    assert!(format!("{err:?}").contains("thermal"));
}

#[test]
fn test_config_invalid_grid() {
    let mut config = GpuThermalAcousticConfig::default();
    config.nx = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_default_config_valid() {
    let config = GpuThermalAcousticConfig::default();
    config.validate().unwrap();

    let max_c = config.c_ref + config.dc_dT * 10.0;
    let cfl_ac = max_c * config.dt / config.dx.min(config.dy).min(config.dz);
    assert!(cfl_ac < 0.3);

    let cfl_th = config.alpha_thermal * config.dt / (config.dx * config.dx);
    assert!(cfl_th < 0.25);
}
