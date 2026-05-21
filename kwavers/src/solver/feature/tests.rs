use super::*;

#[test]
fn test_feature_set_operations() {
    let mut features = SolverFeatureSet::new();

    // Test enabling features
    features.enable(SolverFeature::Reconstruction);
    features.enable(SolverFeature::GpuAcceleration);

    assert!(features.is_enabled(SolverFeature::Reconstruction));
    assert!(features.is_enabled(SolverFeature::GpuAcceleration));
    assert!(!features.is_enabled(SolverFeature::PhotoacousticTimeReversal));

    // Test disabling features
    features.disable(SolverFeature::Reconstruction);
    assert!(!features.is_enabled(SolverFeature::Reconstruction));
    assert!(features.is_enabled(SolverFeature::GpuAcceleration));
}

#[test]
fn test_preset_configurations() {
    let accuracy_features = SolverFeatureSet::accuracy_optimized();
    assert!(accuracy_features.is_enabled(SolverFeature::HighPrecision));
    assert!(accuracy_features.is_enabled(SolverFeature::AdaptiveMeshRefinement));

    let performance_features = SolverFeatureSet::performance_optimized();
    assert!(performance_features.is_enabled(SolverFeature::GpuAcceleration));
    assert!(performance_features.is_enabled(SolverFeature::MultiThreaded));
}

#[test]
fn test_feature_manager() {
    let mut manager = FeatureManager::new();

    // Test enabling available feature
    manager
        .enable_feature(SolverFeature::Reconstruction)
        .unwrap();
    assert!(manager.is_enabled(SolverFeature::Reconstruction));

    // Test enabling unavailable feature (should be available by default)
    manager
        .enable_feature(SolverFeature::PhotoacousticTimeReversal)
        .unwrap();

    // Test with limited available features
    let limited_features = SolverFeatureSet::RECONSTRUCTION | SolverFeatureSet::GPU_ACCELERATION;
    let mut limited_manager = FeatureManager::with_available_features(limited_features);

    limited_manager
        .enable_feature(SolverFeature::Reconstruction)
        .unwrap();
    assert!(limited_manager
        .enable_feature(SolverFeature::PhotoacousticTimeReversal)
        .is_err());
}

#[test]
fn test_feature_display() {
    let mut features = SolverFeatureSet::new();
    features.enable(SolverFeature::Reconstruction);
    features.enable(SolverFeature::GpuAcceleration);

    let display = format!("{}", features);
    assert!(display.contains("Reconstruction"));
    assert!(display.contains("GPU Acceleration"));
}
