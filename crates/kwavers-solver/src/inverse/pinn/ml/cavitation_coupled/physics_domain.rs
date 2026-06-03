//! `SimulationPhysicsDomain` implementation for [`CavitationCoupledDomain`].

use super::domain::CavitationCoupledDomain;
use crate::inverse::pinn::ml::physics::{
    BoundaryPosition, PinnCouplingInterface, InitialConditionSpec, PhysicsLossWeights,
    PhysicsValidationMetric, PinnBoundaryComponent, PinnBoundaryConditionSpec,
    PinnDomainPhysicsParameters, SimulationPhysicsDomain,
};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

use super::config::CavitationCouplingType;

impl<B: AutodiffBackend> SimulationPhysicsDomain<B> for CavitationCoupledDomain<B> {
    fn domain_name(&self) -> &'static str {
        "cavitation_coupled"
    }

    fn pde_residual(
        &self,
        model: &crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PinnDomainPhysicsParameters,
    ) -> Tensor<B, 2> {
        let acoustic_field = model.forward(x.clone(), y.clone(), t.clone());

        // Build bubble-position tensor from physics-driven nucleation sites.
        // Falls back to collocation points if no bubbles have nucleated yet.
        let device = x.device();
        let bubble_positions = if !self.bubble_locations.is_empty() {
            let n = self.bubble_locations.len();
            let xs: Vec<f32> = self
                .bubble_locations
                .iter()
                .map(|(v, _, _)| *v as f32)
                .collect();
            let ys: Vec<f32> = self
                .bubble_locations
                .iter()
                .map(|(_, v, _)| *v as f32)
                .collect();
            let xt = Tensor::<B, 1>::from_floats(xs.as_slice(), &device).reshape([n, 1]);
            let yt = Tensor::<B, 1>::from_floats(ys.as_slice(), &device).reshape([n, 1]);
            Tensor::cat(vec![xt, yt], 1)
        } else {
            Tensor::cat(vec![x.clone(), y.clone()], 1)
        };

        let cav = self.cavitation_residual(&acoustic_field, &bubble_positions, physics_params);
        let scat = self.bubble_scattering_residual(&acoustic_field, &bubble_positions);
        cav + scat
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
        ]
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        vec![
            InitialConditionSpec::DirichletConstant {
                value: vec![0.0],
                component: PinnBoundaryComponent::Scalar,
            },
            InitialConditionSpec::DirichletConstant {
                value: vec![self.config.bubble_params.r0],
                component: PinnBoundaryComponent::Custom("bubble_radius".to_string()),
            },
        ]
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        let (pde_weight, bc_weight) = match self.coupling_type {
            CavitationCouplingType::Weak => (1.0, 10.0),
            CavitationCouplingType::Strong => (1.0, 5.0),
            CavitationCouplingType::MultiBubble => (1.0, 3.0),
        };

        let mut phys = HashMap::new();
        phys.insert(
            "cavitation_weight".to_string(),
            self.config.coupling_strength,
        );
        phys.insert(
            "scattering_weight".to_string(),
            if self.config.nonlinear_acoustic {
                0.5
            } else {
                0.0
            },
        );

        PhysicsLossWeights {
            pde_weight,
            boundary_weight: bc_weight,
            initial_weight: 10.0,
            physics_weights: phys,
        }
    }

    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric> {
        vec![
            PhysicsValidationMetric {
                name: "cavitation_efficiency".to_string(),
                value: 0.0,
                acceptable_range: (0.0, 1.0),
                description: "Acoustic-to-cavitation energy conversion efficiency".to_string(),
            },
            PhysicsValidationMetric {
                name: "bubble_stability".to_string(),
                value: 0.0,
                acceptable_range: (0.0, f64::INFINITY),
                description: "Bubble oscillation stability metric".to_string(),
            },
            PhysicsValidationMetric {
                name: "nonlinear_acoustic_error".to_string(),
                value: 0.0,
                acceptable_range: (-0.1, 0.1),
                description: "Nonlinear acoustic effects error".to_string(),
            },
        ]
    }

    fn supports_coupling(&self) -> bool {
        true
    }

    fn coupling_interfaces(&self) -> Vec<PinnCouplingInterface> {
        self.coupling_interfaces.clone()
    }
}
