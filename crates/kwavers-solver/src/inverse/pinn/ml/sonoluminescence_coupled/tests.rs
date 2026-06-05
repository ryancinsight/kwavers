use super::config::{SonoluminescenceCouplingConfig, SonoluminescenceCouplingType};
use super::domain::SonoluminescenceCoupledDomain;
use crate::inverse::pinn::ml::physics::SimulationPhysicsDomain;

#[test]
fn test_sonoluminescence_coupled_domain_creation() {
    let config = SonoluminescenceCouplingConfig::default();
    let domain: SonoluminescenceCoupledDomain<
        burn::backend::Autodiff<burn::backend::NdArray<f32>>,
    > = SonoluminescenceCoupledDomain::new(config, SonoluminescenceCouplingType::DynamicEmission);

    assert_eq!(domain.domain_name(), "sonoluminescence_coupled");
    assert!(domain.supports_coupling());
    assert!(!domain.coupling_interfaces().is_empty());
}

#[test]
fn test_spectral_coupling_interfaces() {
    let config = SonoluminescenceCouplingConfig {
        spectral_resolution: true,
        ..Default::default()
    };
    let domain: SonoluminescenceCoupledDomain<
        burn::backend::Autodiff<burn::backend::NdArray<f32>>,
    > = SonoluminescenceCoupledDomain::new(config, SonoluminescenceCouplingType::SpectralCoupling);

    let interfaces = domain.coupling_interfaces();
    assert!(interfaces.len() >= 2);

    let has_em_sl = interfaces
        .iter()
        .any(|i| i.name == "electromagnetic_sonoluminescence");
    let has_spectral = interfaces.iter().any(|i| i.name == "spectral_propagation");

    assert!(has_em_sl);
    assert!(has_spectral);
}

#[test]
fn test_coupling_efficiency_parameter() {
    let config = SonoluminescenceCouplingConfig {
        coupling_efficiency: 0.005,
        ..Default::default()
    };
    let domain: SonoluminescenceCoupledDomain<
        burn::backend::Autodiff<burn::backend::NdArray<f32>>,
    > = SonoluminescenceCoupledDomain::new(config, SonoluminescenceCouplingType::DynamicEmission);

    let weights = domain.loss_weights();
    assert_eq!(weights.physics_weights["light_source_weight"], 0.005);
}
