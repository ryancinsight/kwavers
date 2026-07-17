//! `ParamFieldPINNConfig` validation tests.

use super::super::config::ParamFieldPINNConfig;

#[test]
fn test_default_config_validates() {
    let cfg = ParamFieldPINNConfig::default();
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_invalid_config_rejected() {
    let bad = ParamFieldPINNConfig {
        hidden_layers: Vec::new(),
        ..ParamFieldPINNConfig::default()
    };
    assert!(bad.validate().is_err());

    let bad = ParamFieldPINNConfig {
        hidden_layers: vec![64, 0, 64],
        ..ParamFieldPINNConfig::default()
    };
    assert!(bad.validate().is_err());

    let bad = ParamFieldPINNConfig {
        learning_rate: 0.0,
        ..ParamFieldPINNConfig::default()
    };
    assert!(bad.validate().is_err());

    let bad = ParamFieldPINNConfig {
        batch_size: 0,
        ..ParamFieldPINNConfig::default()
    };
    assert!(bad.validate().is_err());
}
