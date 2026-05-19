//! Domain-level integration tests for [`CavitationCoupledDomain`].

#[cfg(test)]
mod tests {
    use super::super::config::{CavitationCouplingConfig, CavitationCouplingType};
    use super::super::domain::CavitationCoupledDomain;
    use crate::solver::inverse::pinn::ml::physics::SimulationPhysicsDomain;

    type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

    #[test]
    fn test_cavitation_coupled_domain_creation() {
        let config = CavitationCouplingConfig::default();
        let domain: CavitationCoupledDomain<B> =
            CavitationCoupledDomain::new(config, CavitationCouplingType::Weak, vec![1e-2, 1e-2]);

        assert_eq!(domain.domain_name(), "cavitation_coupled");
        assert!(domain.supports_coupling());
        assert!(!domain.coupling_interfaces().is_empty());
    }

    #[test]
    fn test_coupling_interfaces() {
        let config = CavitationCouplingConfig {
            multi_bubble_effects: true,
            ..Default::default()
        };
        let domain: CavitationCoupledDomain<B> = CavitationCoupledDomain::new(
            config,
            CavitationCouplingType::MultiBubble,
            vec![1e-2, 1e-2],
        );

        let interfaces = domain.coupling_interfaces();
        // Must have acoustic-bubble and multi-bubble Bjerknes interfaces.
        assert!(interfaces.len() >= 2);

        assert!(interfaces
            .iter()
            .any(|i| i.name == "acoustic_bubble_coupling"));
        assert!(interfaces
            .iter()
            .any(|i| i.name == "multi_bubble_interactions"));
    }

    #[test]
    fn test_loss_weights_by_coupling_type() {
        let config = CavitationCouplingConfig::default();

        let weak: CavitationCoupledDomain<B> = CavitationCoupledDomain::new(
            config.clone(),
            CavitationCouplingType::Weak,
            vec![1e-2, 1e-2],
        );
        let strong: CavitationCoupledDomain<B> = CavitationCoupledDomain::new(
            config.clone(),
            CavitationCouplingType::Strong,
            vec![1e-2, 1e-2],
        );

        let ww = weak.loss_weights();
        let sw = strong.loss_weights();

        // Weak coupling has higher boundary weight than strong (less coupled BCs dominate).
        assert!(
            ww.boundary_weight >= sw.boundary_weight,
            "weak bc_weight={} must be ≥ strong bc_weight={}",
            ww.boundary_weight,
            sw.boundary_weight
        );
    }
}
