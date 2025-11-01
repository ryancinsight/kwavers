//! Electromagnetic Physics Domain for PINN
//!
//! This module implements Maxwell's equations for electromagnetic field simulation
//! using Physics-Informed Neural Networks. The implementation supports electrostatic,
//! magnetostatic, and quasi-static electromagnetic phenomena.
//!
//! ## Mathematical Formulation
//!
//! Maxwell's equations:
//!
//! ∇·D = ρ_free    (Gauss's law)
//! ∇·B = 0         (Gauss's law for magnetism)
//! ∇×E = -∂B/∂t   (Faraday's law)
//! ∇×H = J + ∂D/∂t (Ampere's law with Maxwell's addition)
//!
//! Where:
//! - E: electric field intensity (V/m)
//! - D: electric displacement field (C/m²)
//! - H: magnetic field intensity (A/m)
//! - B: magnetic flux density (T)
//! - J: current density (A/m²)
//! - ρ: charge density (C/m³)

use crate::error::{KwaversError, KwaversResult};
use crate::ml::pinn::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface,
    CouplingType, InitialConditionSpec, PhysicsDomain, PhysicsLossWeights,
    PhysicsParameters, PhysicsValidationMetric,
};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

/// Electromagnetic problem type
#[derive(Debug, Clone, PartialEq)]
pub enum EMProblemType {
    /// Electrostatics (time-independent E field)
    Electrostatic,
    /// Magnetostatics (time-independent B field)
    Magnetostatic,
    /// Quasi-static electromagnetics (low frequency approximation)
    QuasiStatic,
}

/// Current source specification
#[derive(Debug, Clone)]
pub struct CurrentSource {
    /// Source position
    pub position: (f64, f64),
    /// Current density components [Jx, Jy] (A/m²)
    pub current_density: Vec<f64>,
    /// Source radius/size
    pub radius: f64,
}

/// Electromagnetic boundary condition specification
#[derive(Debug, Clone)]
pub enum ElectromagneticBoundarySpec {
    /// Perfect electric conductor (PEC): E_tangential = 0
    PerfectElectricConductor {
        /// Boundary position
        position: BoundaryPosition,
    },
    /// Perfect magnetic conductor (PMC): H_tangential = 0
    PerfectMagneticConductor {
        /// Boundary position
        position: BoundaryPosition,
    },
    /// Impedance boundary: Z E_tangential = -η H_normal
    ImpedanceBoundary {
        /// Boundary position
        position: BoundaryPosition,
        /// Surface impedance (Ω)
        impedance: f64,
    },
    /// Port boundary for waveguide analysis
    Port {
        /// Boundary position
        position: BoundaryPosition,
        /// Port impedance (Ω)
        port_impedance: f64,
        /// Incident mode specification
        mode: usize,
    },
}

/// Electromagnetic physics domain implementation
#[derive(Debug, Clone)]
pub struct ElectromagneticDomain {
    /// Problem type
    pub problem_type: EMProblemType,
    /// Electric permittivity (F/m)
    pub permittivity: f64,
    /// Magnetic permeability (H/m)
    pub permeability: f64,
    /// Electrical conductivity (S/m)
    pub conductivity: f64,
    /// Speed of light in medium (m/s)
    pub c: f64,
    /// Current sources
    pub current_sources: Vec<CurrentSource>,
    /// Boundary conditions
    pub boundary_specs: Vec<ElectromagneticBoundarySpec>,
    /// Domain dimensions [Lx, Ly]
    pub domain_size: Vec<f64>,
}

impl Default for ElectromagneticDomain {
    fn default() -> Self {
        let permittivity = 8.854e-12; // Vacuum permittivity
        let permeability = 4e-7 * std::f64::consts::PI; // Vacuum permeability
        let c = 1.0 / (permittivity * permeability).sqrt();

        Self {
            problem_type: EMProblemType::Electrostatic,
            permittivity,
            permeability,
            conductivity: 0.0, // Perfect dielectric
            c,
            current_sources: Vec::new(),
            boundary_specs: Vec::new(),
            domain_size: vec![1.0, 1.0],
        }
    }
}

impl<B: AutodiffBackend> PhysicsDomain<B> for ElectromagneticDomain {
    fn domain_name(&self) -> &'static str {
        "electromagnetic"
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

        // Create input tensor for neural network
        let inputs = Tensor::cat(vec![x.clone(), y.clone(), t.clone()], 1);

        // Forward pass through model to get field components
        let outputs = model.forward(&inputs);

        match self.problem_type {
            EMProblemType::Electrostatic => {
                // Electrostatic case: solve ∇·(ε∇φ) = -ρ
                // where φ is electric potential, E = -∇φ
                self.electrostatic_residual(&outputs, x, y, eps, physics_params)
            }
            EMProblemType::Magnetostatic => {
                // Magnetostatic case: solve ∇×(ν∇×A) = J
                // where A is magnetic vector potential, B = ∇×A
                self.magnetostatic_residual(&outputs, x, y, mu, physics_params)
            }
            EMProblemType::QuasiStatic => {
                // Quasi-static case: solve full Maxwell's equations with low frequency approximation
                self.quasi_static_residual(&outputs, x, y, t, eps, mu, sigma, physics_params)
            }
        }
    }

    fn boundary_conditions(&self) -> Vec<BoundaryConditionSpec> {
        self.boundary_specs
            .iter()
            .map(|spec| match spec {
                ElectromagneticBoundarySpec::PerfectElectricConductor { position } => {
                    // E_tangential = 0 (Dirichlet for potential in electrostatics)
                    BoundaryConditionSpec::Dirichlet {
                        boundary: position.clone(),
                        value: vec![0.0], // Zero potential
                        component: BoundaryComponent::Scalar,
                    }
                }
                ElectromagneticBoundarySpec::PerfectMagneticConductor { position } => {
                    // H_tangential = 0 (Neumann for vector potential)
                    BoundaryConditionSpec::Neumann {
                        boundary: position.clone(),
                        flux: vec![0.0, 0.0], // Zero tangential H field
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    }
                }
                ElectromagneticBoundarySpec::ImpedanceBoundary { position, impedance } => {
                    // Impedance boundary condition
                    BoundaryConditionSpec::Robin {
                        boundary: position.clone(),
                        alpha: 1.0 / impedance,
                        beta: 0.0,
                        component: BoundaryComponent::Scalar,
                    }
                }
                ElectromagneticBoundarySpec::Port { position, port_impedance, .. } => {
                    // Port boundary with characteristic impedance
                    BoundaryConditionSpec::Robin {
                        boundary: position.clone(),
                        alpha: 1.0 / port_impedance,
                        beta: 0.0,
                        component: BoundaryComponent::Scalar,
                    }
                }
            })
            .collect()
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        match self.problem_type {
            EMProblemType::Electrostatic => {
                // Zero initial potential
                vec![InitialConditionSpec::DirichletConstant {
                    value: vec![0.0],
                    component: BoundaryComponent::Scalar,
                }]
            }
            EMProblemType::Magnetostatic => {
                // Zero initial vector potential
                vec![InitialConditionSpec::DirichletConstant {
                    value: vec![0.0, 0.0],
                    component: BoundaryComponent::Vector(vec![0, 1]),
                }]
            }
            EMProblemType::QuasiStatic => {
                // Zero initial fields
                vec![
                    InitialConditionSpec::DirichletConstant {
                        value: vec![0.0, 0.0], // E field
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    },
                    InitialConditionSpec::DirichletConstant {
                        value: vec![0.0, 0.0], // H field
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    },
                ]
            }
        }
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        let (pde_weight, bc_weight) = match self.problem_type {
            EMProblemType::Electrostatic => (1.0, 10.0),
            EMProblemType::Magnetostatic => (1.0, 10.0),
            EMProblemType::QuasiStatic => (1.0, 5.0), // Lower BC weight for time-dependent
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
        };

        // Common metrics
        let mut common_metrics = vec![
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
        false // EM coupling not implemented yet
    }

    fn coupling_interfaces(&self) -> Vec<CouplingInterface> {
        Vec::new() // No coupling interfaces defined yet
    }
}

impl ElectromagneticDomain {
    /// Create a new electromagnetic domain
    pub fn new(
        problem_type: EMProblemType,
        permittivity: f64,
        permeability: f64,
        conductivity: f64,
        domain_size: Vec<f64>,
    ) -> Self {
        let c = 1.0 / (permittivity * permeability).sqrt();

        Self {
            problem_type,
            permittivity,
            permeability,
            conductivity,
            c,
            current_sources: Vec::new(),
            boundary_specs: Vec::new(),
            domain_size,
        }
    }

    /// Add a current source
    pub fn add_current_source(mut self, position: (f64, f64), current_density: Vec<f64>, radius: f64) -> Self {
        self.current_sources.push(CurrentSource {
            position,
            current_density,
            radius,
        });
        self
    }

    /// Add a perfect electric conductor boundary
    pub fn add_pec_boundary(mut self, position: BoundaryPosition) -> Self {
        self.boundary_specs.push(ElectromagneticBoundarySpec::PerfectElectricConductor { position });
        self
    }

    /// Add a perfect magnetic conductor boundary
    pub fn add_pmc_boundary(mut self, position: BoundaryPosition) -> Self {
        self.boundary_specs.push(ElectromagneticBoundarySpec::PerfectMagneticConductor { position });
        self
    }

    /// Set problem type
    pub fn with_problem_type(mut self, problem_type: EMProblemType) -> Self {
        self.problem_type = problem_type;
        self
    }

    /// Compute electrostatic residual: ∇·(ε∇φ) = -ρ
    fn electrostatic_residual(
        &self,
        outputs: &Tensor<B, 2>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        eps: f64,
        _physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Assume output is electric potential φ
        let phi = outputs.clone();

        // Compute electric field E = -∇φ
        let e_x = -phi.backward(x);
        let e_y = -phi.backward(y);

        // Compute ∇·(εE) = ε(∂Ex/∂x + ∂Ey/∂y)
        let div_d = eps * (e_x.backward(x) + e_y.backward(y));

        // For electrostatics, ∇·D = ρ_free
        // Assume ρ = 0 for now (can be extended with charge sources)
        div_d // Should equal -ρ, so residual is ∇·D + ρ
    }

    /// Compute magnetostatic residual: ∇×(ν∇×A) = μ₀J
    fn magnetostatic_residual(
        &self,
        outputs: &Tensor<B, 2>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        mu: f64,
        _physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Assume outputs are Az (z-component of vector potential A)
        let az = outputs.clone();

        // Compute magnetic field B = ∇×A = (∂Ay/∂x - ∂Ax/∂y, ∂Az/∂y - ∂Ay/∂z, ∂Ax/∂z - ∂Az/∂x)
        // For 2D, assume Ax = Ay = 0, so Bx = -∂Az/∂y, By = ∂Az/∂x
        let b_x = -az.backward(y);
        let b_y = az.backward(x);

        // Compute ∇×B = ∂By/∂x - ∂Bx/∂y
        let div_b = b_y.backward(x) - b_x.backward(y);

        // For magnetostatics, ∇·B = 0 should be satisfied
        div_b
    }

    /// Compute quasi-static electromagnetic residual
    fn quasi_static_residual(
        &self,
        outputs: &Tensor<B, 2>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        eps: f64,
        mu: f64,
        sigma: f64,
        _physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // For quasi-static, solve for E and H fields
        // Assume outputs are [Ex, Ey, Hx, Hy] or similar
        // This is a simplified implementation

        let batch_size = x.shape().dims[0];

        // Extract field components (simplified assumption)
        let ex = outputs.clone().slice([0..batch_size, 0..1]).squeeze(1);
        let ey = outputs.clone().slice([0..batch_size, 1..2]).squeeze(1);
        let hx = outputs.clone().slice([0..batch_size, 2..3]).squeeze(1);
        let hy = outputs.clone().slice([0..batch_size, 3..4]).squeeze(1);

        // Faraday's law: ∇×E = -μ∂H/∂t
        let curl_e_z = ey.backward(x) - ex.backward(y);
        let d_hx_dt = hx.backward(t);
        let d_hy_dt = hy.backward(t);
        let faraday_residual = curl_e_z + mu * (d_hy_dt.backward(x) - d_hx_dt.backward(y));

        // Simplified Ampere's law ( neglecting displacement current for low frequency)
        let curl_h_z = hy.backward(x) - hx.backward(y);
        let ampere_residual = curl_h_z; // Should equal J, assumed 0 here

        Tensor::cat(vec![faraday_residual, ampere_residual], 0)
    }

    /// Validate domain configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.permittivity <= 0.0 {
            return Err(KwaversError::ValidationError {
                field: "permittivity".to_string(),
                reason: "Permittivity must be positive".to_string(),
            });
        }

        if self.permeability <= 0.0 {
            return Err(KwaversError::ValidationError {
                field: "permeability".to_string(),
                reason: "Permeability must be positive".to_string(),
            });
        }

        if self.conductivity < 0.0 {
            return Err(KwaversError::ValidationError {
                field: "conductivity".to_string(),
                reason: "Conductivity cannot be negative".to_string(),
            });
        }

        if self.c <= 0.0 || !self.c.is_finite() {
            return Err(KwaversError::ValidationError {
                field: "speed_of_light".to_string(),
                reason: "Speed of light must be positive and finite".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_electromagnetic_domain_creation() {
        let domain = ElectromagneticDomain::new(
            EMProblemType::Electrostatic,
            8.854e-12,
            4e-7 * std::f64::consts::PI,
            0.0,
            vec![1.0, 1.0],
        );

        assert_eq!(domain.problem_type, EMProblemType::Electrostatic);
        assert!((domain.permittivity - 8.854e-12).abs() < 1e-15);
    }

    #[test]
    fn test_domain_validation() {
        let valid_domain = ElectromagneticDomain::default();
        assert!(valid_domain.validate().is_ok());

        let invalid_domain = ElectromagneticDomain::new(
            EMProblemType::Electrostatic,
            -1.0, // Invalid permittivity
            4e-7 * std::f64::consts::PI,
            0.0,
            vec![1.0, 1.0],
        );
        assert!(invalid_domain.validate().is_err());
    }

    #[test]
    fn test_boundary_condition_builder() {
        let domain = ElectromagneticDomain::default()
            .add_pec_boundary(BoundaryPosition::Left)
            .add_pmc_boundary(BoundaryPosition::Right);

        assert_eq!(domain.boundary_specs.len(), 2);

        match &domain.boundary_specs[0] {
            ElectromagneticBoundarySpec::PerfectElectricConductor { .. } => {
                // Correct type
            }
            _ => panic!("Expected PerfectElectricConductor"),
        }
    }

    #[test]
    fn test_current_source_builder() {
        let domain = ElectromagneticDomain::default()
            .add_current_source((0.5, 0.5), vec![1e6, 0.0], 0.1);

        assert_eq!(domain.current_sources.len(), 1);
        assert_eq!(domain.current_sources[0].position, (0.5, 0.5));
    }

    #[test]
    fn test_physics_domain_interface() {
        let domain = ElectromagneticDomain::default();

        assert_eq!(domain.domain_name(), "electromagnetic");

        let weights = domain.loss_weights();
        assert_eq!(weights.pde_weight, 1.0);

        let metrics = domain.validation_metrics();
        assert!(metrics.len() >= 4); // At least common metrics

        assert!(!domain.supports_coupling()); // Not implemented yet
    }

    #[test]
    fn test_problem_type_specifics() {
        let electrostatic = ElectromagneticDomain::default()
            .with_problem_type(EMProblemType::Electrostatic);

        let magnetostatic = ElectromagneticDomain::default()
            .with_problem_type(EMProblemType::Magnetostatic);

        let quasi_static = ElectromagneticDomain::default()
            .with_problem_type(EMProblemType::QuasiStatic);

        assert_eq!(electrostatic.problem_type, EMProblemType::Electrostatic);
        assert_eq!(magnetostatic.problem_type, EMProblemType::Magnetostatic);
        assert_eq!(quasi_static.problem_type, EMProblemType::QuasiStatic);

        // Check that loss weights differ by problem type
        let es_weights = electrostatic.loss_weights();
        let qs_weights = quasi_static.loss_weights();
        assert_eq!(es_weights.boundary_weight, 10.0);
        assert_eq!(qs_weights.boundary_weight, 5.0);
    }
}
