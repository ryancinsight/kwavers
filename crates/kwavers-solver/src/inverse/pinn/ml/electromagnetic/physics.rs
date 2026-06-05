use super::domain::ElectromagneticDomain;
use super::residuals::{
    electrostatic_residual, magnetostatic_residual, quasi_static_residual,
    wave_propagation_residual,
};
use super::types::{EMProblemType, ElectromagneticBoundarySpec};
use crate::inverse::pinn::ml::physics::{
    BoundaryPosition, InitialConditionSpec, PhysicsLossWeights, PhysicsValidationMetric,
    PinnBoundaryComponent, PinnBoundaryConditionSpec, PinnCouplingInterface,
    PinnDomainPhysicsParameters, SimulationPhysicsDomain,
};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use std::collections::HashMap;

impl<B: AutodiffBackend> SimulationPhysicsDomain<B> for ElectromagneticDomain<B> {
    fn domain_name(&self) -> &'static str {
        "electromagnetic"
    }

    fn pde_residual(
        &self,
        model: &crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PinnDomainPhysicsParameters,
    ) -> Tensor<B, 2> {
        // Get material properties from parameters or defaults
        let eps = physics_params
            .domain_params
            .get("permittivity")
            .copied()
            .unwrap_or(self.permittivity);
        let mu = physics_params
            .domain_params
            .get("permeability")
            .copied()
            .unwrap_or(self.permeability);
        let sigma = physics_params
            .domain_params
            .get("conductivity")
            .copied()
            .unwrap_or(self.conductivity);

        match self.problem_type {
            EMProblemType::Electrostatic => {
                electrostatic_residual(model, x, y, eps, physics_params)
            }
            EMProblemType::Magnetostatic => magnetostatic_residual(model, x, y, mu, physics_params),
            EMProblemType::QuasiStatic => {
                quasi_static_residual(model, x, y, t, eps, mu, sigma, physics_params)
            }
            EMProblemType::WavePropagation => {
                wave_propagation_residual(model, x, y, t, eps, mu, sigma, physics_params)
            }
        }
    }

    fn boundary_conditions(&self) -> Vec<PinnBoundaryConditionSpec> {
        if self.boundary_specs.is_empty() {
            return match self.problem_type {
                EMProblemType::Electrostatic => vec![
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
                        boundary: BoundaryPosition::Bottom,
                        value: vec![0.0],
                        component: PinnBoundaryComponent::Scalar,
                    },
                    PinnBoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Top,
                        value: vec![0.0],
                        component: PinnBoundaryComponent::Scalar,
                    },
                ],
                EMProblemType::Magnetostatic => vec![
                    PinnBoundaryConditionSpec::Neumann {
                        boundary: BoundaryPosition::Left,
                        flux: vec![0.0, 0.0],
                        component: PinnBoundaryComponent::Vector(vec![0, 1]),
                    },
                    PinnBoundaryConditionSpec::Neumann {
                        boundary: BoundaryPosition::Right,
                        flux: vec![0.0, 0.0],
                        component: PinnBoundaryComponent::Vector(vec![0, 1]),
                    },
                    PinnBoundaryConditionSpec::Neumann {
                        boundary: BoundaryPosition::Bottom,
                        flux: vec![0.0, 0.0],
                        component: PinnBoundaryComponent::Vector(vec![0, 1]),
                    },
                    PinnBoundaryConditionSpec::Neumann {
                        boundary: BoundaryPosition::Top,
                        flux: vec![0.0, 0.0],
                        component: PinnBoundaryComponent::Vector(vec![0, 1]),
                    },
                ],
                EMProblemType::QuasiStatic => vec![
                    PinnBoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Left,
                        value: vec![0.0, 0.0],
                        component: PinnBoundaryComponent::Vector(vec![0, 1]),
                    },
                    PinnBoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Right,
                        value: vec![0.0, 0.0],
                        component: PinnBoundaryComponent::Vector(vec![0, 1]),
                    },
                    PinnBoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Bottom,
                        value: vec![0.0, 0.0],
                        component: PinnBoundaryComponent::Vector(vec![0, 1]),
                    },
                    PinnBoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Top,
                        value: vec![0.0, 0.0],
                        component: PinnBoundaryComponent::Vector(vec![0, 1]),
                    },
                ],
                EMProblemType::WavePropagation => vec![
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
                        boundary: BoundaryPosition::Bottom,
                        value: vec![0.0],
                        component: PinnBoundaryComponent::Scalar,
                    },
                    PinnBoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Top,
                        value: vec![0.0],
                        component: PinnBoundaryComponent::Scalar,
                    },
                ],
            };
        }

        self.boundary_specs
            .iter()
            .map(|spec| match spec {
                ElectromagneticBoundarySpec::PerfectElectricConductor { position } => {
                    PinnBoundaryConditionSpec::Dirichlet {
                        boundary: position.clone(),
                        value: vec![0.0],
                        component: PinnBoundaryComponent::Scalar,
                    }
                }
                ElectromagneticBoundarySpec::PerfectMagneticConductor { position } => {
                    PinnBoundaryConditionSpec::Neumann {
                        boundary: position.clone(),
                        flux: vec![0.0, 0.0],
                        component: PinnBoundaryComponent::Vector(vec![0, 1]),
                    }
                }
                ElectromagneticBoundarySpec::ImpedanceBoundary {
                    position,
                    impedance,
                } => PinnBoundaryConditionSpec::Robin {
                    boundary: position.clone(),
                    alpha: 1.0 / impedance,
                    beta: 0.0,
                    component: PinnBoundaryComponent::Scalar,
                },
                ElectromagneticBoundarySpec::Port {
                    position,
                    port_impedance,
                    ..
                } => PinnBoundaryConditionSpec::Robin {
                    boundary: position.clone(),
                    alpha: 1.0 / port_impedance,
                    beta: 0.0,
                    component: PinnBoundaryComponent::Scalar,
                },
            })
            .collect()
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        match self.problem_type {
            EMProblemType::Electrostatic => {
                vec![InitialConditionSpec::DirichletConstant {
                    value: vec![0.0],
                    component: PinnBoundaryComponent::Scalar,
                }]
            }
            EMProblemType::Magnetostatic => {
                vec![InitialConditionSpec::DirichletConstant {
                    value: vec![0.0, 0.0],
                    component: PinnBoundaryComponent::Vector(vec![0, 1]),
                }]
            }
            EMProblemType::QuasiStatic => {
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
            EMProblemType::WavePropagation => {
                vec![
                    InitialConditionSpec::DirichletConstant {
                        value: vec![0.0],
                        component: PinnBoundaryComponent::Scalar,
                    },
                    InitialConditionSpec::DirichletConstant {
                        value: vec![0.0],
                        component: PinnBoundaryComponent::Scalar,
                    },
                ]
            }
        }
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        let (pde_weight, bc_weight) = match self.problem_type {
            EMProblemType::Electrostatic => (1.0, 10.0),
            EMProblemType::Magnetostatic => (1.0, 10.0),
            EMProblemType::QuasiStatic => (1.0, 5.0),
            EMProblemType::WavePropagation => (1.0, 2.0),
        };

        PhysicsLossWeights {
            pde_weight,
            boundary_weight: bc_weight,
            initial_weight: 10.0,
            physics_weights: {
                let mut weights = HashMap::new();
                weights.insert("maxwell_weight".to_string(), 1.0);
                weights.insert("constitutive_weight".to_string(), 1.0);
                weights
            },
        }
    }

    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric> {
        let metrics = match self.problem_type {
            EMProblemType::Electrostatic => vec![
                PhysicsValidationMetric {
                    name: "gauss_law_residual".to_string(),
                    value: 0.0,
                    acceptable_range: (-1e-6, 1e-6),
                    description: "Gauss's law residual ∇·D = ρ".to_string(),
                },
                PhysicsValidationMetric {
                    name: "energy_functional".to_string(),
                    value: 0.0,
                    acceptable_range: (0.0, f64::INFINITY),
                    description: "Electrostatic energy functional".to_string(),
                },
            ],
            EMProblemType::Magnetostatic => vec![
                PhysicsValidationMetric {
                    name: "ampere_law_residual".to_string(),
                    value: 0.0,
                    acceptable_range: (-1e-6, 1e-6),
                    description: "Ampere's law residual ∇×H = J".to_string(),
                },
                PhysicsValidationMetric {
                    name: "magnetic_energy".to_string(),
                    value: 0.0,
                    acceptable_range: (0.0, f64::INFINITY),
                    description: "Magnetic energy functional".to_string(),
                },
            ],
            EMProblemType::QuasiStatic => vec![
                PhysicsValidationMetric {
                    name: "faraday_residual".to_string(),
                    value: 0.0,
                    acceptable_range: (-1e-6, 1e-6),
                    description: "Faraday's law residual ∇×E = -∂B/∂t".to_string(),
                },
                PhysicsValidationMetric {
                    name: "ampere_maxwell_residual".to_string(),
                    value: 0.0,
                    acceptable_range: (-1e-6, 1e-6),
                    description: "Ampere-Maxwell law residual ∇×H = J + ∂D/∂t".to_string(),
                },
                PhysicsValidationMetric {
                    name: "divergence_free_b".to_string(),
                    value: 0.0,
                    acceptable_range: (-1e-6, 1e-6),
                    description: "Magnetic Gauss law residual ∇·B = 0".to_string(),
                },
            ],
            EMProblemType::WavePropagation => vec![
                PhysicsValidationMetric {
                    name: "faraday_residual".to_string(),
                    value: 0.0,
                    acceptable_range: (-1e-6, 1e-6),
                    description: "Faraday's law residual ∇×E = -∂B/∂t".to_string(),
                },
                PhysicsValidationMetric {
                    name: "ampere_maxwell_residual".to_string(),
                    value: 0.0,
                    acceptable_range: (-1e-6, 1e-6),
                    description: "Ampere-Maxwell law residual ∇×H = J + ∂D/∂t + σE".to_string(),
                },
                PhysicsValidationMetric {
                    name: "divergence_free_b".to_string(),
                    value: 0.0,
                    acceptable_range: (-1e-6, 1e-6),
                    description: "Magnetic Gauss law residual ∇·B = 0".to_string(),
                },
                PhysicsValidationMetric {
                    name: "gauss_law_electric".to_string(),
                    value: 0.0,
                    acceptable_range: (-1e-6, 1e-6),
                    description: "Electric Gauss law residual ∇·D = ρ".to_string(),
                },
                PhysicsValidationMetric {
                    name: "wave_speed".to_string(),
                    value: 0.0,
                    acceptable_range: (2.99792458e8 * 0.99, 2.99792458e8 * 1.01),
                    description: "Wave propagation speed validation".to_string(),
                },
            ],
        };

        let common_metrics = vec![
            PhysicsValidationMetric {
                name: "boundary_error".to_string(),
                value: 0.0,
                acceptable_range: (-1e-6, 1e-6),
                description: "Boundary condition satisfaction error".to_string(),
            },
            PhysicsValidationMetric {
                name: "field_continuity".to_string(),
                value: 0.0,
                acceptable_range: (-1e-6, 1e-6),
                description: "Field continuity at material interfaces".to_string(),
            },
        ];

        metrics.into_iter().chain(common_metrics).collect()
    }

    fn supports_coupling(&self) -> bool {
        false
    }

    fn coupling_interfaces(&self) -> Vec<PinnCouplingInterface> {
        Vec::new()
    }
}
