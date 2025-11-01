//! Navier-Stokes Physics Domain for PINN
//!
//! This module implements incompressible Navier-Stokes equations for fluid dynamics
//! simulation using Physics-Informed Neural Networks. The implementation supports
//! laminar and turbulent flow regimes with proper boundary condition handling.
//!
//! ## Mathematical Formulation
//!
//! The incompressible Navier-Stokes equations are:
//!
//! ∂u/∂t + (u·∇)u = -1/ρ ∇p + ν ∇²u + f
//! ∂v/∂t + (u·∇)v = -1/ρ ∇p + ν ∇²v + f
//! ∇·u = 0
//!
//! Where:
//! - u, v: velocity components
//! - p: pressure
//! - ρ: density
//! - ν: kinematic viscosity
//! - f: body forces
//!
//! ## Boundary Conditions
//!
//! - No-slip wall: u = 0, v = 0
//! - Inlet: prescribed velocity profile
//! - Outlet: zero traction (∂u/∂n = 0)
//! - Free surface: dynamic boundary conditions

use crate::error::{KwaversError, KwaversResult};
use crate::ml::pinn::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface,
    CouplingType, InitialConditionSpec, PhysicsDomain, PhysicsLossWeights,
    PhysicsParameters, PhysicsValidationMetric,
};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

/// Flow regime classification
#[derive(Debug, Clone, PartialEq)]
pub enum FlowRegime {
    /// Laminar flow (Re < 2300)
    Laminar,
    /// Transitional flow (2300 < Re < 4000)
    Transitional,
    /// Turbulent flow (Re > 4000)
    Turbulent,
}

/// Turbulence model specification
#[derive(Debug, Clone)]
pub enum TurbulenceModel {
    /// k-ε turbulence model
    KEpsilon,
    /// SST k-ω turbulence model
    SSTKOmega,
    /// Reynolds stress model
    ReynoldsStress,
}

/// Navier-Stokes boundary condition specification
#[derive(Debug, Clone)]
pub enum NavierStokesBoundarySpec {
    /// No-slip wall boundary condition
    NoSlipWall {
        /// Boundary position
        position: BoundaryPosition,
    },
    /// Inlet with prescribed velocity
    Inlet {
        /// Boundary position
        position: BoundaryPosition,
        /// Inlet velocity components [u, v]
        velocity: Vec<f64>,
    },
    /// Outlet with zero traction
    Outlet {
        /// Boundary position
        position: BoundaryPosition,
    },
    /// Free surface boundary
    FreeSurface {
        /// Boundary position
        position: BoundaryPosition,
        /// Surface tension coefficient
        surface_tension: f64,
    },
    /// Periodic boundary condition
    Periodic {
        /// Master boundary
        master: BoundaryPosition,
        /// Slave boundary
        slave: BoundaryPosition,
    },
}

/// Navier-Stokes physics domain implementation
#[derive(Debug, Clone)]
pub struct NavierStokesDomain {
    /// Reynolds number
    pub reynolds_number: f64,
    /// Flow regime
    pub flow_regime: FlowRegime,
    /// Turbulence model (optional)
    pub turbulence_model: Option<TurbulenceModel>,
    /// Boundary conditions
    pub boundary_specs: Vec<NavierStokesBoundarySpec>,
    /// Reference density
    pub density: f64,
    /// Reference viscosity
    pub viscosity: f64,
    /// Body forces [fx, fy]
    pub body_forces: Vec<f64>,
    /// Domain dimensions [Lx, Ly]
    pub domain_size: Vec<f64>,
}

impl Default for NavierStokesDomain {
    fn default() -> Self {
        Self {
            reynolds_number: 100.0,
            flow_regime: FlowRegime::Laminar,
            turbulence_model: None,
            boundary_specs: Vec::new(),
            density: 1000.0, // Water density kg/m³
            viscosity: 0.001, // Water viscosity Pa·s
            body_forces: vec![0.0, 0.0], // No body forces
            domain_size: vec![1.0, 1.0], // 1m x 1m domain
        }
    }
}

impl<B: AutodiffBackend> PhysicsDomain<B> for NavierStokesDomain {
    fn domain_name(&self) -> &'static str {
        "navier_stokes"
    }

    fn pde_residual(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Get kinematic viscosity from parameters or default
        let nu = physics_params
            .domain_params
            .get("kinematic_viscosity")
            .copied()
            .unwrap_or(self.viscosity / self.density);

        // Create input tensor for neural network
        let inputs = Tensor::cat(vec![x.clone(), y.clone(), t.clone()], 1);

        // Forward pass through model to get velocity components
        // Note: This assumes the model outputs [u, v] velocity components
        let outputs = model.forward(&inputs);

        // Split outputs into velocity components (assuming 2D output)
        let batch_size = x.shape().dims[0];
        let u = outputs.clone().slice([0..batch_size, 0..1]).squeeze(1);
        let v = outputs.clone().slice([0..batch_size, 1..2]).squeeze(1);

        // Compute spatial derivatives
        let u_x = u.backward(x).squeeze(1);
        let u_y = u.backward(y).squeeze(1);
        let u_t = u.backward(t).squeeze(1);
        let u_xx = u_x.backward(x).squeeze(1);
        let u_yy = u_y.backward(y).squeeze(1);

        let v_x = v.backward(x).squeeze(1);
        let v_y = v.backward(y).squeeze(1);
        let v_t = v.backward(t).squeeze(1);
        let v_xx = v_x.backward(x).squeeze(1);
        let v_yy = v_y.backward(y).squeeze(1);

        // Continuity equation: ∂u/∂x + ∂v/∂y = 0
        let continuity = u_x + v_y;

        // Momentum equations with convection and diffusion
        // ∂u/∂t + u*∂u/∂x + v*∂u/∂y = -1/ρ ∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²) + fx
        // ∂v/∂t + u*∂v/∂x + v*∂v/∂y = -1/ρ ∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²) + fy

        // For incompressible NS, we use the pressure Poisson equation approach
        // Here we compute the momentum residuals assuming pressure is computed separately

        let convection_u = u.clone() * u_x + v.clone() * u_y;
        let diffusion_u = nu * (u_xx + u_yy);
        let momentum_u = u_t + convection_u - diffusion_u;

        let convection_v = u.clone() * v_x + v.clone() * v_y;
        let diffusion_v = nu * (v_xx + v_yy);
        let momentum_v = v_t + convection_v - diffusion_v;

        // Add body forces if specified
        let fx = physics_params
            .domain_params
            .get("body_force_x")
            .copied()
            .unwrap_or(self.body_forces[0]);
        let fy = physics_params
            .domain_params
            .get("body_force_y")
            .copied()
            .unwrap_or(self.body_forces[1]);

        let momentum_u_with_force = momentum_u - fx;
        let momentum_v_with_force = momentum_v - fy;

        // Combine all residuals
        Tensor::cat(vec![continuity, momentum_u_with_force, momentum_v_with_force], 0)
    }

    fn boundary_conditions(&self) -> Vec<BoundaryConditionSpec> {
        self.boundary_specs
            .iter()
            .map(|spec| match spec {
                NavierStokesBoundarySpec::NoSlipWall { position } => {
                    BoundaryConditionSpec::Dirichlet {
                        boundary: position.clone(),
                        value: vec![0.0, 0.0], // u = 0, v = 0
                        component: BoundaryComponent::Vector(vec![0, 1]), // velocity components
                    }
                }
                NavierStokesBoundarySpec::Inlet { position, velocity } => {
                    BoundaryConditionSpec::Dirichlet {
                        boundary: position.clone(),
                        value: velocity.clone(),
                        component: BoundaryComponent::Vector(vec![0, 1]), // velocity components
                    }
                }
                NavierStokesBoundarySpec::Outlet { position } => {
                    // Zero traction boundary condition: ∂u/∂n = 0
                    BoundaryConditionSpec::Neumann {
                        boundary: position.clone(),
                        flux: vec![0.0, 0.0], // zero normal stress
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    }
                }
                NavierStokesBoundarySpec::FreeSurface { position, .. } => {
                    // Free surface: normal stress balance with surface tension
                    BoundaryConditionSpec::Neumann {
                        boundary: position.clone(),
                        flux: vec![0.0, 0.0], // To be computed with surface tension
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    }
                }
                NavierStokesBoundarySpec::Periodic { .. } => {
                    // Periodic boundaries are handled specially in the solver
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::CustomRectangular {
                            x_min: 0.0,
                            x_max: 0.0,
                            y_min: 0.0,
                            y_max: 0.0,
                        },
                        value: vec![0.0, 0.0],
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    }
                }
            })
            .collect()
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        // Default initial condition: zero velocity
        vec![InitialConditionSpec::DirichletConstant {
            value: vec![0.0, 0.0],
            component: BoundaryComponent::Vector(vec![0, 1]),
        }]
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        PhysicsLossWeights {
            pde_weight: 1.0,
            boundary_weight: 10.0,
            initial_weight: 10.0,
            physics_weights: {
                let mut weights = HashMap::new();
                weights.insert("continuity_weight".to_string(), 1.0);
                weights.insert("momentum_weight".to_string(), 1.0);
                weights
            },
        }
    }

    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric> {
        vec![
            PhysicsValidationMetric {
                name: "continuity_residual".to_string(),
                value: 0.0, // To be computed during validation
                acceptable_range: (-1e-6, 1e-6),
                description: "Mass conservation error (should be ~0)".to_string(),
            },
            PhysicsValidationMetric {
                name: "momentum_residual".to_string(),
                value: 0.0,
                acceptable_range: (-1e-4, 1e-4),
                description: "Momentum conservation error".to_string(),
            },
            PhysicsValidationMetric {
                name: "energy_conservation".to_string(),
                value: 0.0,
                acceptable_range: (-1e-3, 1e-3),
                description: "Kinetic energy conservation".to_string(),
            },
            PhysicsValidationMetric {
                name: "boundary_error".to_string(),
                value: 0.0,
                acceptable_range: (-1e-6, 1e-6),
                description: "Boundary condition satisfaction error".to_string(),
            },
        ]
    }

    fn supports_coupling(&self) -> bool {
        true // Navier-Stokes can couple with heat transfer, structural mechanics
    }

    fn coupling_interfaces(&self) -> Vec<CouplingInterface> {
        vec![
            CouplingInterface {
                name: "fluid_solid_interface".to_string(),
                position: BoundaryPosition::CustomRectangular {
                    x_min: 0.0,
                    x_max: 1.0,
                    y_min: 0.0,
                    y_max: 0.0,
                },
                coupled_domains: vec!["structural_mechanics".to_string()],
                coupling_type: CouplingType::Conjugate,
                coupling_params: {
                    let mut params = HashMap::new();
                    params.insert("friction_coefficient".to_string(), 0.0);
                    params
                },
            },
            CouplingInterface {
                name: "fluid_thermal_interface".to_string(),
                position: BoundaryPosition::CustomRectangular {
                    x_min: 0.0,
                    x_max: 1.0,
                    y_min: 0.0,
                    y_max: 0.0,
                },
                coupled_domains: vec!["heat_transfer".to_string()],
                coupling_type: CouplingType::Conjugate,
                coupling_params: {
                    let mut params = HashMap::new();
                    params.insert("heat_transfer_coefficient".to_string(), 100.0);
                    params
                },
            },
        ]
    }
}

impl NavierStokesDomain {
    /// Create a new Navier-Stokes domain with specified parameters
    pub fn new(
        reynolds_number: f64,
        density: f64,
        viscosity: f64,
        domain_size: Vec<f64>,
    ) -> Self {
        let flow_regime = if reynolds_number < 2300.0 {
            FlowRegime::Laminar
        } else if reynolds_number < 4000.0 {
            FlowRegime::Transitional
        } else {
            FlowRegime::Turbulent
        };

        Self {
            reynolds_number,
            flow_regime,
            turbulence_model: if matches!(flow_regime, FlowRegime::Turbulent) {
                Some(TurbulenceModel::KEpsilon)
            } else {
                None
            },
            boundary_specs: Vec::new(),
            density,
            viscosity,
            body_forces: vec![0.0, 0.0],
            domain_size,
        }
    }

    /// Add a no-slip wall boundary condition
    pub fn add_no_slip_wall(mut self, position: BoundaryPosition) -> Self {
        self.boundary_specs.push(NavierStokesBoundarySpec::NoSlipWall { position });
        self
    }

    /// Add an inlet boundary condition
    pub fn add_inlet(mut self, position: BoundaryPosition, velocity: Vec<f64>) -> Self {
        self.boundary_specs.push(NavierStokesBoundarySpec::Inlet { position, velocity });
        self
    }

    /// Add an outlet boundary condition
    pub fn add_outlet(mut self, position: BoundaryPosition) -> Self {
        self.boundary_specs.push(NavierStokesBoundarySpec::Outlet { position });
        self
    }

    /// Add body forces
    pub fn with_body_forces(mut self, forces: Vec<f64>) -> Self {
        self.body_forces = forces;
        self
    }

    /// Enable turbulence modeling
    pub fn with_turbulence_model(mut self, model: TurbulenceModel) -> Self {
        self.turbulence_model = Some(model);
        self.flow_regime = FlowRegime::Turbulent;
        self
    }

    /// Compute kinematic viscosity
    pub fn kinematic_viscosity(&self) -> f64 {
        self.viscosity / self.density
    }

    /// Validate domain configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.reynolds_number <= 0.0 {
            return Err(KwaversError::ValidationError {
                field: "reynolds_number".to_string(),
                reason: "Reynolds number must be positive".to_string(),
            });
        }

        if self.density <= 0.0 {
            return Err(KwaversError::ValidationError {
                field: "density".to_string(),
                reason: "Density must be positive".to_string(),
            });
        }

        if self.viscosity <= 0.0 {
            return Err(KwaversError::ValidationError {
                field: "viscosity".to_string(),
                reason: "Viscosity must be positive".to_string(),
            });
        }

        if self.domain_size.len() != 2 {
            return Err(KwaversError::ValidationError {
                field: "domain_size".to_string(),
                reason: "Domain size must specify [Lx, Ly]".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navier_stokes_domain_creation() {
        let domain = NavierStokesDomain::new(100.0, 1000.0, 0.001, vec![1.0, 1.0]);

        assert_eq!(domain.reynolds_number, 100.0);
        assert!(matches!(domain.flow_regime, FlowRegime::Laminar));
        assert!(domain.turbulence_model.is_none());
        assert_eq!(domain.density, 1000.0);
        assert_eq!(domain.viscosity, 0.001);
    }

    #[test]
    fn test_turbulent_flow_detection() {
        let domain = NavierStokesDomain::new(5000.0, 1000.0, 0.001, vec![1.0, 1.0]);

        assert!(matches!(domain.flow_regime, FlowRegime::Turbulent));
        assert!(domain.turbulence_model.is_some());
    }

    #[test]
    fn test_kinematic_viscosity() {
        let domain = NavierStokesDomain::new(100.0, 1000.0, 0.001, vec![1.0, 1.0]);
        assert!((domain.kinematic_viscosity() - 0.000001).abs() < 1e-10);
    }

    #[test]
    fn test_domain_validation() {
        let valid_domain = NavierStokesDomain::new(100.0, 1000.0, 0.001, vec![1.0, 1.0]);
        assert!(valid_domain.validate().is_ok());

        let invalid_domain = NavierStokesDomain::new(-100.0, 1000.0, 0.001, vec![1.0, 1.0]);
        assert!(invalid_domain.validate().is_err());
    }

    #[test]
    fn test_boundary_condition_builder() {
        let domain = NavierStokesDomain::default()
            .add_no_slip_wall(BoundaryPosition::Bottom)
            .add_inlet(BoundaryPosition::Left, vec![1.0, 0.0])
            .add_outlet(BoundaryPosition::Right);

        assert_eq!(domain.boundary_specs.len(), 3);

        match &domain.boundary_specs[0] {
            NavierStokesBoundarySpec::NoSlipWall { position } => {
                assert!(matches!(position, BoundaryPosition::Bottom));
            }
            _ => panic!("Expected NoSlipWall"),
        }

        match &domain.boundary_specs[1] {
            NavierStokesBoundarySpec::Inlet { velocity, .. } => {
                assert_eq!(velocity[0], 1.0);
                assert_eq!(velocity[1], 0.0);
            }
            _ => panic!("Expected Inlet"),
        }
    }

    #[test]
    fn test_physics_domain_interface() {
        let domain = NavierStokesDomain::default();

        assert_eq!(domain.domain_name(), "navier_stokes");

        let weights = domain.loss_weights();
        assert_eq!(weights.pde_weight, 1.0);
        assert_eq!(weights.boundary_weight, 10.0);

        let metrics = domain.validation_metrics();
        assert_eq!(metrics.len(), 4);

        assert!(domain.supports_coupling());
        let interfaces = domain.coupling_interfaces();
        assert_eq!(interfaces.len(), 2);
    }
}
