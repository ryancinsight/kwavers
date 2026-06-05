use super::*;

#[test]
fn test_config_defaults() {
    let config = PinnBeamformingConfig::default();
    assert_eq!(config.inference.batch_size, 32);
    assert!(!config.inference.use_fp16);
    assert!(!config.uncertainty.bayesian_enabled);
}

#[test]
fn test_device_config() {
    let cpu = DeviceConfig::Cpu;
    assert!(matches!(cpu, DeviceConfig::Cpu));

    let gpu = DeviceConfig::Gpu { device_id: 0 };
    assert!(matches!(gpu, DeviceConfig::Gpu { .. }));
}

#[test]
fn test_provider_registry() {
    let registry = PinnProviderRegistry::new();
    assert_eq!(registry.list_providers().len(), 0);
}
