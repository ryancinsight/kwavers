use super::config::GpuThermalAcousticConfig;
use super::{
    GpuThermalAcousticBuffers, GpuThermalAcousticSolver, ThermalAcousticBufferProvider,
    ThermalAcousticSolverProvider, WgpuThermalAcousticBuffers, WgpuThermalAcousticSolverProvider,
};

#[test]
fn thermal_acoustic_buffers_are_generic_over_provider_trait() {
    fn assert_provider<P>()
    where
        P: ThermalAcousticBufferProvider,
    {
        let _ = core::mem::size_of::<GpuThermalAcousticBuffers<P>>();
    }

    assert_provider::<WgpuThermalAcousticBuffers>();
}

#[test]
fn wgpu_thermal_acoustic_buffers_declare_native_scalar() {
    fn assert_scalar<P>()
    where
        P: ThermalAcousticBufferProvider<Scalar = f32>,
    {
        let _ = core::mem::size_of::<GpuThermalAcousticBuffers<P>>();
    }

    assert_scalar::<WgpuThermalAcousticBuffers>();
}

#[test]
fn thermal_acoustic_solver_is_generic_over_provider_trait() {
    fn assert_provider<P>()
    where
        P: ThermalAcousticSolverProvider,
    {
        let _ = core::mem::size_of::<GpuThermalAcousticSolver<P>>();
        let _ = core::mem::size_of::<<P as crate::backend::provider::GpuProviderBackend>::Device>();
    }

    assert_provider::<WgpuThermalAcousticSolverProvider>();
}

#[test]
fn test_config_validation() {
    let config = GpuThermalAcousticConfig::default();
    config.validate().unwrap();
}

#[test]
fn test_config_cfl_acoustic_violation() {
    let config = GpuThermalAcousticConfig {
        dt: 1.0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(format!("{err:?}").contains("acoustic"));
}

#[test]
fn test_config_cfl_thermal_violation() {
    let config = GpuThermalAcousticConfig {
        alpha_thermal: 30.0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(format!("{err:?}").contains("thermal"));
}

#[test]
fn test_config_invalid_grid() {
    let config = GpuThermalAcousticConfig {
        nx: 0,
        ..Default::default()
    };
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
