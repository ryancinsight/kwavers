//! `ParamFieldPINNConfig` validation tests.

use super::super::config::ParamFieldPINNConfig;

#[test]
fn test_default_config_validates() {
    let cfg = ParamFieldPINNConfig::default();
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_invalid_config_rejected() {
    let mut bad = ParamFieldPINNConfig::default();
    bad.hidden_layers.clear();
    assert!(bad.validate().is_err());

    let mut bad = ParamFieldPINNConfig::default();
    bad.hidden_layers = vec![64, 0, 64];
    assert!(bad.validate().is_err());

    let mut bad = ParamFieldPINNConfig::default();
    bad.learning_rate = 0.0;
    assert!(bad.validate().is_err());

    let mut bad = ParamFieldPINNConfig::default();
    bad.batch_size = 0;
    assert!(bad.validate().is_err());
}
