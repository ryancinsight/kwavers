//! `PhysicsDomain<B>` trait implementation for `SonoluminescenceCoupledDomain`.

use std::collections::HashMap;

use burn::tensor::{backend::AutodiffBackend, Tensor};

use crate::solver::inverse::pinn::ml::physics::{
    BoundaryPosition, CouplingInterface, InitialConditionSpec, PhysicsDomain, PhysicsLossWeights,
    PhysicsValidationMetric, PinnBoundaryComponent, PinnBoundaryConditionSpec,
    PinnDomainPhysicsParameters,
};

use super::super::config::SonoluminescenceCouplingType;
use super::SonoluminescenceCoupledDomain;

impl<B: AutodiffBackend> PhysicsDomain<B> for SonoluminescenceCoupledDomain<B> {
    fn domain_name(&self) -> &'static str {
        "sonoluminescence_coupled"
    }

    fn pde_residual(
        &self,
        model: &crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PinnDomainPhysicsParameters,
    ) -> Tensor<B, 2> {
        self.electromagnetic_residual_with_sources(model, x, y, t, physics_params)
    }

    fn boundary_conditions(&self) -> Vec<PinnBoundaryConditionSpec> {
        vec![
            PinnBoundaryConditionSpec::Dirichlet {
                boundary: BoundaryPosition::Left,
                value: vec![0.0],
                component: PinnBoundaryComponent::Scalar,
            },
            PinnBoundaryConditionSpec::Dirichlet {
                boundary: BoundaryPosition::Right,
                value: vec![0.0],
                component: PinnBoundaryComponent::Scalar,
            },
            PinnBoundaryConditionSpec::Dirichlet {
                boundary: BoundaryPosition::Top,
                value: vec![0.0],
                component: PinnBoundaryComponent::Scalar,
            },
            PinnBoundaryConditionSpec::Dirichlet {
                boundary: BoundaryPosition::Bottom,
                value: vec![0.0],
                component: PinnBoundaryComponent::Scalar,
            },
        ]
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        vec![
            InitialConditionSpec::DirichletConstant {
                value: vec![0.0, 0.0],
                component: PinnBoundaryComponent::Vector(vec![0, 1]),
            },
            InitialConditionSpec::DirichletConstant {
                value: vec![0.0, 0.0],
                component: PinnBoundaryComponent::Vector(vec![0, 1]),
            },
        ]
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        let (pde_weight, bc_weight) = match self.coupling_type {
            SonoluminescenceCouplingType::StaticEmission => (1.0, 10.0),
            SonoluminescenceCouplingType::DynamicEmission => (1.0, 5.0),
            SonoluminescenceCouplingType::SpectralCoupling => (1.0, 2.0),
        };

        PhysicsLossWeights {
            pde_weight,
            boundary_weight: bc_weight,
            initial_weight: 10.0,
            physics_weights: {
                let mut weights = HashMap::new();
                weights.insert(
                    "light_source_weight".to_string(),
                    self.config.coupling_efficiency,
                );
                weights.insert(
                    "spectral_weight".to_string(),
                    if self.config.spectral_resolution {
                        1.0
                    } else {
                        0.0
                    },
                );
                weights
            },
        }
    }

    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric> {
        vec![
            PhysicsValidationMetric {
                name: "light_emission_efficiency".to_string(),
                value: 0.0,
                acceptable_range: (0.0, 1.0),
                description: "Efficiency of bubble energy conversion to light".to_string(),
            },
            PhysicsValidationMetric {
                name: "spectral_accuracy".to_string(),
                value: 0.0,
                acceptable_range: (-0.1, 0.1),
                description: "Accuracy of spectral emission calculations".to_string(),
            },
            PhysicsValidationMetric {
                name: "electromagnetic_consistency".to_string(),
                value: 0.0,
                acceptable_range: (-1e-6, 1e-6),
                description: "Maxwell's equations residual with light sources".to_string(),
            },
            PhysicsValidationMetric {
                name: "total_luminosity".to_string(),
                value: 0.0,
                acceptable_range: (0.0, f64::INFINITY),
                description: "Total light output from sonoluminescence".to_string(),
            },
        ]
    }

    fn supports_coupling(&self) -> bool {
        true
    }

    fn coupling_interfaces(&self) -> Vec<CouplingInterface> {
        self.coupling_interfaces.clone()
    }
}
