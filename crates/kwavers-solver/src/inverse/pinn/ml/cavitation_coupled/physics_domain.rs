//! `SimulationPhysicsDomain` implementation for [`CavitationCoupledDomain`].

use super::domain::CavitationCoupledDomain;
use crate::inverse::pinn::ml::physics::{
    BoundaryPosition, InitialConditionSpec, PhysicsLossWeights, PhysicsValidationMetric,
    PinnBoundaryComponent, PinnBoundaryConditionSpec, PinnCouplingInterface,
    PinnDomainPhysicsParameters, SimulationPhysicsDomain,
};
use coeus_autograd::Var;
use std::collections::HashMap;

use super::config::CavitationCouplingType;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> SimulationPhysicsDomain<B>
    for CavitationCoupledDomain<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    fn domain_name(&self) -> &'static str {
        "cavitation_coupled"
    }

    fn pde_residual(
        &self,
        model: &crate::inverse::pinn::ml::PinnWave2D<B>,
        x: &Var<f32, B>,
        y: &Var<f32, B>,
        t: &Var<f32, B>,
        physics_params: &PinnDomainPhysicsParameters,
    ) -> Var<f32, B> {
        let acoustic_field = model.forward(x, y, t);

        // Build bubble-position tensor from physics-driven nucleation sites.
        // Falls back to collocation points if no bubbles have nucleated yet.
        let backend = B::default();
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
            let xt = Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &xs, &backend),
                false,
            );
            let yt = Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &ys, &backend),
                false,
            );
            coeus_autograd::cat(&[&xt, &yt], 1)
        } else {
            coeus_autograd::cat(&[x, y], 1)
        };

        let cav = self.cavitation_residual(&acoustic_field, &bubble_positions, physics_params);
        let scat = self.bubble_scattering_residual(&acoustic_field, &bubble_positions);
        coeus_autograd::add(&cav, &scat)
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
