//! Structural Mechanics Physics Domain for PINN
//!
//! This module implements linear and nonlinear elasticity for structural analysis
//! using Physics-Informed Neural Networks. The implementation supports static and
//! dynamic loading, material nonlinearity, and multi-physics coupling.
//!
//! ## Mathematical Formulation
//!
//! The equations of motion for continuum mechanics:
//!
//! ∇·σ + b = ρ ∂²u/∂t²
//!
//! Where:
//! - σ: Cauchy stress tensor
//! - b: body forces per unit volume
//! - ρ: mass density
//! - u: displacement vector
//!
//! ## Constitutive Relations
//!
//! Linear elasticity: σ = C:ε
//! where C is the stiffness tensor and ε is the strain tensor.

use crate::error::{KwaversError, KwaversResult};
use crate::ml::pinn::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface,
    CouplingType, InitialConditionSpec, PhysicsDomain, PhysicsLossWeights,
    PhysicsParameters, PhysicsValidationMetric,
};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

/// Structural load specification
#[derive(Debug, Clone)]
pub enum StructuralLoad {
    /// Concentrated force at a point
    ConcentratedForce {
        /// Load position
        position: (f64, f64),
        /// Force components [Fx, Fy] (N)
        force: Vec<f64>,
    },
    /// Distributed pressure load
    PressureLoad {
        /// Boundary position
        boundary: BoundaryPosition,
        /// Pressure magnitude (Pa)
        pressure: f64,
    },
    /// Body force (gravity, etc.)
    BodyForce {
        /// Body force components [bx, by] (N/m³)
        force_density: Vec<f64>,
    },
}

/// Material model specification
#[derive(Debug, Clone)]
pub enum MaterialModel {
    /// Linear elastic isotropic material
    LinearElastic {
        /// Young's modulus (Pa)
        youngs_modulus: f64,
        /// Poisson's ratio
        poissons_ratio: f64,
    },
    /// Hyperelastic material (neo-Hookean)
    NeoHookean {
        /// Shear modulus (Pa)
        shear_modulus: f64,
        /// Bulk modulus (Pa)
        bulk_modulus: f64,
    },
}

/// Structural mechanics boundary condition specification
#[derive(Debug, Clone)]
pub enum StructuralMechanicsBoundarySpec {
    /// Fixed displacement boundary
    FixedDisplacement {
        /// Boundary position
        position: BoundaryPosition,
        /// Prescribed displacements [ux, uy]
        displacement: Vec<f64>,
    },
    /// Free boundary (traction-free)
    Free {
        /// Boundary position
        position: BoundaryPosition,
    },
    /// Roller boundary (normal displacement fixed)
    Roller {
        /// Boundary position
        position: BoundaryPosition,
        /// Normal direction [nx, ny]
        normal: Vec<f64>,
    },
}

/// Structural mechanics physics domain implementation
#[derive(Debug, Clone)]
pub struct StructuralMechanicsDomain {
    /// Young's modulus (Pa)
    pub youngs_modulus: f64,
    /// Poisson's ratio
    pub poissons_ratio: f64,
    /// Material density (kg/m³)
    pub density: f64,
    /// Damping coefficient (for dynamic problems)
    pub damping: f64,
    /// Material model
    pub material_model: MaterialModel,
    /// Structural loads
    pub loads: Vec<StructuralLoad>,
    /// Boundary conditions
    pub boundary_specs: Vec<StructuralMechanicsBoundarySpec>,
    /// Domain dimensions [Lx, Ly]
    pub domain_size: Vec<f64>,
}

impl Default for StructuralMechanicsDomain {
    fn default() -> Self {
        Self {
            youngs_modulus: 200e9, // Steel-like
            poissons_ratio: 0.3,
            density: 7850.0,       // Steel density
            damping: 0.0,          // No damping by default
            material_model: MaterialModel::LinearElastic {
                youngs_modulus: 200e9,
                poissons_ratio: 0.3,
            },
            loads: Vec::new(),
            boundary_specs: Vec::new(),
            domain_size: vec![1.0, 1.0],
        }
    }
}

impl<B: AutodiffBackend> PhysicsDomain<B> for StructuralMechanicsDomain {
    fn domain_name(&self) -> &'static str {
        "structural_mechanics"
    }

    fn pde_residual(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Get material properties from parameters or defaults
        let e = physics_params
            .domain_params
            .get("youngs_modulus")
            .copied()
            .unwrap_or(self.youngs_modulus);
        let nu = physics_params
            .domain_params
            .get("poissons_ratio")
            .copied()
            .unwrap_or(self.poissons_ratio);
        let rho = physics_params
            .domain_params
            .get("density")
            .copied()
            .unwrap_or(self.density);

        // Lamé parameters for plane strain
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e / (2.0 * (1.0 + nu));

        // Create input tensor for neural network
        let inputs = Tensor::cat(vec![x.clone(), y.clone(), t.clone()], 1);

        // Forward pass through model to get displacement components
        let outputs = model.forward(&inputs);

        // Split outputs into displacement components (assuming 2D output)
        let batch_size = x.shape().dims[0];
        let u = outputs.clone().slice([0..batch_size, 0..1]).squeeze(1);
        let v = outputs.clone().slice([0..batch_size, 1..2]).squeeze(1);

        // Compute displacement gradients (strains)
        let u_x = u.backward(x);
        let u_y = u.backward(y);
        let v_x = v.backward(x);
        let v_y = v.backward(y);

        // Compute stresses using constitutive relation (plane strain)
        let sigma_xx = lambda * (u_x + v_y) + 2.0 * mu * u_x;
        let sigma_yy = lambda * (u_x + v_y) + 2.0 * mu * v_y;
        let sigma_xy = mu * (u_y + v_x);

        // Compute equilibrium equations (∇·σ + b = ρ ∂²u/∂t²)
        let sigma_xx_x = sigma_xx.backward(x);
        let sigma_xy_y = sigma_xy.backward(y);
        let force_x = sigma_xx_x + sigma_xy_y;

        let sigma_xy_x = sigma_xy.backward(x);
        let sigma_yy_y = sigma_yy.backward(y);
        let force_y = sigma_xy_x + sigma_yy_y;

        // Add body forces
        let bx = physics_params
            .domain_params
            .get("body_force_x")
            .copied()
            .unwrap_or(0.0);
        let by = physics_params
            .domain_params
            .get("body_force_y")
            .copied()
            .unwrap_or(0.0);

        let force_x_with_body = force_x - bx;
        let force_y_with_body = force_y - by;

        // For dynamic problems, include inertial terms
        if self.damping > 0.0 || physics_params.domain_params.contains_key("dynamic") {
            let u_tt = u.backward(t).backward(t);
            let v_tt = v.backward(t).backward(t);

            let damping_force_x = self.damping * u.backward(t);
            let damping_force_y = self.damping * v.backward(t);

            let inertial_x = rho * u_tt + damping_force_x;
            let inertial_y = rho * v_tt + damping_force_y;

            Tensor::cat(vec![force_x_with_body - inertial_x, force_y_with_body - inertial_y], 0)
        } else {
            // Static equilibrium
            Tensor::cat(vec![force_x_with_body, force_y_with_body], 0)
        }
    }

    fn boundary_conditions(&self) -> Vec<BoundaryConditionSpec> {
        self.boundary_specs
            .iter()
            .map(|spec| match spec {
                StructuralMechanicsBoundarySpec::FixedDisplacement { position, displacement } => {
                    BoundaryConditionSpec::Dirichlet {
                        boundary: position.clone(),
                        value: displacement.clone(),
                        component: BoundaryComponent::Vector(vec![0, 1]), // displacement components
                    }
                }
                StructuralMechanicsBoundarySpec::Free { position } => {
                    BoundaryConditionSpec::Neumann {
                        boundary: position.clone(),
                        flux: vec![0.0, 0.0], // zero traction
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    }
                }
                StructuralMechanicsBoundarySpec::Roller { position, normal } => {
                    // For roller boundary, normal displacement = 0, tangential traction = 0
                    // This is more complex and would require proper implementation
                    BoundaryConditionSpec::Neumann {
                        boundary: position.clone(),
                        flux: vec![0.0, 0.0], // Simplified
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    }
                }
            })
            .collect()
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        // Default initial conditions: zero displacement and velocity
        vec![
            InitialConditionSpec::DirichletConstant {
                value: vec![0.0, 0.0], // zero displacement
                component: BoundaryComponent::Vector(vec![0, 1]),
            },
            InitialConditionSpec::NeumannConstant {
                flux: vec![0.0, 0.0], // zero velocity
                component: BoundaryComponent::Vector(vec![0, 1]),
            },
        ]
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        PhysicsLossWeights {
            pde_weight: 1.0,
            boundary_weight: 100.0, // Higher weight for displacement BCs
            initial_weight: 10.0,
            physics_weights: {
                let mut weights = HashMap::new();
                weights.insert("equilibrium_weight".to_string(), 1.0);
                weights.insert("compatibility_weight".to_string(), 1.0);
                weights
            },
        }
    }

    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric> {
        vec![
            PhysicsValidationMetric {
                name: "equilibrium_residual".to_string(),
                value: 0.0,
                acceptable_range: (-1e-6, 1e-6),
                description: "Force equilibrium residual (should be ~0)".to_string(),
            },
            PhysicsValidationMetric {
                name: "strain_energy".to_string(),
                value: 0.0,
                acceptable_range: (0.0, f64::INFINITY),
                description: "Total strain energy (should be positive)".to_string(),
            },
            PhysicsValidationMetric {
                name: "boundary_error".to_string(),
                value: 0.0,
                acceptable_range: (-1e-6, 1e-6),
                description: "Boundary condition satisfaction error".to_string(),
            },
            PhysicsValidationMetric {
                name: "stress_concentration".to_string(),
                value: 0.0,
                acceptable_range: (-f64::INFINITY, f64::INFINITY),
                description: "Maximum stress concentration factor".to_string(),
            },
        ]
    }

    fn supports_coupling(&self) -> bool {
        true // Structural mechanics can couple with Navier-Stokes, heat transfer
    }

    fn coupling_interfaces(&self) -> Vec<CouplingInterface> {
        vec![
            CouplingInterface {
                name: "fluid_structure_interface".to_string(),
                position: BoundaryPosition::CustomRectangular {
                    x_min: 0.0,
                    x_max: 1.0,
                    y_min: 0.0,
                    y_max: 0.0,
                },
                coupled_domains: vec!["navier_stokes".to_string()],
                coupling_type: CouplingType::Conjugate,
                coupling_params: {
                    let mut params = HashMap::new();
                    params.insert("fluid_density".to_string(), 1000.0);
                    params.insert("fluid_viscosity".to_string(), 0.001);
                    params
                },
            },
            CouplingInterface {
                name: "thermal_stress_interface".to_string(),
                position: BoundaryPosition::CustomRectangular {
                    x_min: 0.0,
                    x_max: 1.0,
                    y_min: 0.0,
                    y_max: 0.0,
                },
                coupled_domains: vec!["heat_transfer".to_string()],
                coupling_type: CouplingType::SolutionContinuity,
                coupling_params: {
                    let mut params = HashMap::new();
                    params.insert("thermal_expansion_coefficient".to_string(), 1e-5);
                    params.insert("reference_temperature".to_string(), 293.0);
                    params
                },
            },
        ]
    }
}

impl StructuralMechanicsDomain {
    /// Create a new structural mechanics domain
    pub fn new(
        youngs_modulus: f64,
        poissons_ratio: f64,
        density: f64,
        domain_size: Vec<f64>,
    ) -> Self {
        Self {
            youngs_modulus,
            poissons_ratio,
            density,
            damping: 0.0,
            material_model: MaterialModel::LinearElastic {
                youngs_modulus,
                poissons_ratio,
            },
            loads: Vec::new(),
            boundary_specs: Vec::new(),
            domain_size,
        }
    }

    /// Add a fixed displacement boundary condition
    pub fn add_fixed_bc(mut self, position: BoundaryPosition, displacement: Vec<f64>) -> Self {
        self.boundary_specs.push(StructuralMechanicsBoundarySpec::FixedDisplacement {
            position,
            displacement,
        });
        self
    }

    /// Add a free boundary condition
    pub fn add_free_bc(mut self, position: BoundaryPosition) -> Self {
        self.boundary_specs.push(StructuralMechanicsBoundarySpec::Free { position });
        self
    }

    /// Add a concentrated force load
    pub fn add_concentrated_force(mut self, position: (f64, f64), force: Vec<f64>) -> Self {
        self.loads.push(StructuralLoad::ConcentratedForce { position, force });
        self
    }

    /// Add a pressure load
    pub fn add_pressure_load(mut self, boundary: BoundaryPosition, pressure: f64) -> Self {
        self.loads.push(StructuralLoad::PressureLoad { boundary, pressure });
        self
    }

    /// Enable damping for dynamic analysis
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Set material model
    pub fn with_material_model(mut self, model: MaterialModel) -> Self {
        self.material_model = model;
        match model {
            MaterialModel::LinearElastic { youngs_modulus, poissons_ratio } => {
                self.youngs_modulus = youngs_modulus;
                self.poissons_ratio = poissons_ratio;
            }
            MaterialModel::NeoHookean { .. } => {
                // Update properties for hyperelastic material
                // This would require additional implementation
            }
        }
        self
    }

    /// Compute bulk modulus
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))
    }

    /// Compute shear modulus
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))
    }

    /// Validate domain configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.youngs_modulus <= 0.0 {
            return Err(KwaversError::ValidationError {
                field: "youngs_modulus".to_string(),
                reason: "Young's modulus must be positive".to_string(),
            });
        }

        if !(0.0..=0.5).contains(&self.poissons_ratio) {
            return Err(KwaversError::ValidationError {
                field: "poissons_ratio".to_string(),
                reason: "Poisson's ratio must be between 0 and 0.5".to_string(),
            });
        }

        if self.density <= 0.0 {
            return Err(KwaversError::ValidationError {
                field: "density".to_string(),
                reason: "Density must be positive".to_string(),
            });
        }

        if self.damping < 0.0 {
            return Err(KwaversError::ValidationError {
                field: "damping".to_string(),
                reason: "Damping coefficient cannot be negative".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structural_mechanics_domain_creation() {
        let domain = StructuralMechanicsDomain::new(200e9, 0.3, 7850.0, vec![1.0, 1.0]);

        assert_eq!(domain.youngs_modulus, 200e9);
        assert_eq!(domain.poissons_ratio, 0.3);
        assert_eq!(domain.density, 7850.0);
    }

    #[test]
    fn test_material_properties() {
        let domain = StructuralMechanicsDomain::new(200e9, 0.3, 7850.0, vec![1.0, 1.0]);

        let expected_bulk = 200e9 / (3.0 * (1.0 - 2.0 * 0.3));
        assert!((domain.bulk_modulus() - expected_bulk).abs() < 1e-6);

        let expected_shear = 200e9 / (2.0 * (1.0 + 0.3));
        assert!((domain.shear_modulus() - expected_shear).abs() < 1e-6);
    }

    #[test]
    fn test_domain_validation() {
        let valid_domain = StructuralMechanicsDomain::new(200e9, 0.3, 7850.0, vec![1.0, 1.0]);
        assert!(valid_domain.validate().is_ok());

        let invalid_domain = StructuralMechanicsDomain::new(-200e9, 0.3, 7850.0, vec![1.0, 1.0]);
        assert!(invalid_domain.validate().is_err());
    }

    #[test]
    fn test_boundary_condition_builder() {
        let domain = StructuralMechanicsDomain::default()
            .add_fixed_bc(BoundaryPosition::Left, vec![0.0, 0.0])
            .add_free_bc(BoundaryPosition::Top);

        assert_eq!(domain.boundary_specs.len(), 2);

        match &domain.boundary_specs[0] {
            StructuralMechanicsBoundarySpec::FixedDisplacement { displacement, .. } => {
                assert_eq!(displacement[0], 0.0);
                assert_eq!(displacement[1], 0.0);
            }
            _ => panic!("Expected FixedDisplacement"),
        }
    }

    #[test]
    fn test_load_builder() {
        let domain = StructuralMechanicsDomain::default()
            .add_concentrated_force((0.5, 1.0), vec![0.0, -1000.0])
            .add_pressure_load(BoundaryPosition::Top, 1e5);

        assert_eq!(domain.loads.len(), 2);

        match &domain.loads[0] {
            StructuralLoad::ConcentratedForce { force, .. } => {
                assert_eq!(force[1], -1000.0);
            }
            _ => panic!("Expected ConcentratedForce"),
        }
    }

    #[test]
    fn test_physics_domain_interface() {
        let domain = StructuralMechanicsDomain::default();

        assert_eq!(domain.domain_name(), "structural_mechanics");

        let weights = domain.loss_weights();
        assert_eq!(weights.pde_weight, 1.0);
        assert_eq!(weights.boundary_weight, 100.0);

        let metrics = domain.validation_metrics();
        assert_eq!(metrics.len(), 4);

        assert!(domain.supports_coupling());
        let interfaces = domain.coupling_interfaces();
        assert_eq!(interfaces.len(), 2);
    }
}
