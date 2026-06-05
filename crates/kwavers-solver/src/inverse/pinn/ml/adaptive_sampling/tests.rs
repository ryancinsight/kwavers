use super::*;
use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

#[derive(Debug)]
struct MockPhysicsDomain;

impl<B: AutodiffBackend> crate::inverse::pinn::ml::physics::SimulationPhysicsDomain<B>
    for MockPhysicsDomain
{
    fn domain_name(&self) -> &'static str {
        "mock"
    }
    fn pde_residual(
        &self,
        _model: &crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        _y: &Tensor<B, 2>,
        _t: &Tensor<B, 2>,
        _params: &PinnDomainPhysicsParameters,
    ) -> Tensor<B, 2> {
        x.clone() * 0.1
    }
    fn boundary_conditions(
        &self,
    ) -> Vec<crate::inverse::pinn::ml::physics::PinnBoundaryConditionSpec> {
        vec![]
    }
    fn initial_conditions(&self) -> Vec<crate::inverse::pinn::ml::physics::InitialConditionSpec> {
        vec![]
    }
    fn loss_weights(&self) -> crate::inverse::pinn::ml::physics::PhysicsLossWeights {
        crate::inverse::pinn::ml::physics::PhysicsLossWeights::default()
    }
    fn validation_metrics(
        &self,
    ) -> Vec<crate::inverse::pinn::ml::physics::PhysicsValidationMetric> {
        vec![]
    }
}

#[test]
fn test_adaptive_sampler_creation() {
    type TestBackend = Autodiff<NdArray<f32>>;

    let domain: Box<dyn crate::inverse::pinn::ml::physics::SimulationPhysicsDomain<TestBackend>> =
        Box::new(MockPhysicsDomain);
    let strategy = AdaptiveRefinementConfig::default();

    let sampler = AdaptiveCollocationSampler::<TestBackend>::new(100, domain, strategy);

    let sampler = sampler.unwrap();
    assert_eq!(sampler.total_points, 100);
    assert_eq!(sampler.active_points.shape().dims, [100, 3]);
    assert_eq!(sampler.priorities.shape().dims, [100]);
}

#[test]
fn test_sampling_strategy_defaults() {
    let strategy = AdaptiveRefinementConfig::default();

    assert_eq!(strategy.refinement_threshold, 0.8);
    assert_eq!(strategy.coarsening_threshold, 0.2);
    assert_eq!(strategy.refinement_fraction, 0.1);
    assert_eq!(strategy.uncertainty_weight, 0.3);
    assert_eq!(strategy.residual_weight, 0.7);
}

#[test]
fn test_sampling_stats_defaults() {
    let stats = SamplingStats::default();

    assert_eq!(stats.iterations, 0);
    assert_eq!(stats.points_refined, 0);
    assert_eq!(stats.points_coarsened, 0);
    assert_eq!(stats.distribution_entropy, 0.0);
    assert_eq!(stats.avg_priority, 1.0);
    assert_eq!(stats.max_priority, 1.0);
}
