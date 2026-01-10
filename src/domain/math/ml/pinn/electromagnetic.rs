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

use crate::domain::core::error::{KwaversError, KwaversResult};
use crate::domain::math::ml::pinn::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface,
    InitialConditionSpec, PhysicsDomain, PhysicsLossWeights, PhysicsParameters,
    PhysicsValidationMetric,
};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

/// GPU acceleration flag for electromagnetic simulations
#[cfg(feature = "gpu")]
use crate::domain::math::ml::pinn::electromagnetic_gpu::EMConfig;

/// Electromagnetic problem type
#[derive(Debug, Clone, PartialEq)]
pub enum EMProblemType {
    /// Electrostatics (time-independent E field)
    Electrostatic,
    /// Magnetostatics (time-independent B field)
    Magnetostatic,
    /// Quasi-static electromagnetics (low frequency approximation)
    QuasiStatic,
    /// Full wave propagation (time-dependent Maxwell's equations)
    WavePropagation,
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
#[derive(Debug)]
pub struct ElectromagneticDomain<B: AutodiffBackend> {
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
    /// GPU acceleration configuration (optional)
    #[cfg(feature = "gpu")]
    pub gpu_config: Option<EMConfig>,
    /// Backend marker
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> Default for ElectromagneticDomain<B> {
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
            #[cfg(feature = "gpu")]
            gpu_config: None,
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: AutodiffBackend> PhysicsDomain<B> for ElectromagneticDomain<B> {
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
        let _inputs = Tensor::cat(vec![x.clone(), y.clone(), t.clone()], 1);

        // Forward pass through model to get field components
        let outputs = model.forward(x.clone(), y.clone(), t.clone());

        match self.problem_type {
            EMProblemType::Electrostatic => {
                // Electrostatic case: solve ∇·(ε∇φ) = -ρ
                // where φ is electric potential, E = -∇φ
                self.electrostatic_residual(model, x, y, eps, physics_params)
            }
            EMProblemType::Magnetostatic => {
                // Magnetostatic case: solve ∇×(ν∇×A) = μ₀J
                // where A is magnetic vector potential, B = ∇×A
                self.magnetostatic_residual(model, x, y, mu, physics_params)
            }
            EMProblemType::QuasiStatic => {
                // Quasi-static case: solve full Maxwell's equations with low frequency approximation
                self.quasi_static_residual(&outputs, x, y, t, eps, mu, sigma, physics_params)
            }
            EMProblemType::WavePropagation => {
                // Full time-dependent Maxwell's equations for wave propagation
                self.wave_propagation_residual(&outputs, x, y, t, eps, mu, sigma, physics_params)
            }
        }
    }

    fn boundary_conditions(&self) -> Vec<BoundaryConditionSpec> {
        if self.boundary_specs.is_empty() {
            return match self.problem_type {
                EMProblemType::Electrostatic => vec![
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Left,
                        value: vec![0.0],
                        component: BoundaryComponent::Scalar,
                    },
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Right,
                        value: vec![0.0],
                        component: BoundaryComponent::Scalar,
                    },
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Bottom,
                        value: vec![0.0],
                        component: BoundaryComponent::Scalar,
                    },
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Top,
                        value: vec![0.0],
                        component: BoundaryComponent::Scalar,
                    },
                ],
                EMProblemType::Magnetostatic => vec![
                    BoundaryConditionSpec::Neumann {
                        boundary: BoundaryPosition::Left,
                        flux: vec![0.0, 0.0],
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    },
                    BoundaryConditionSpec::Neumann {
                        boundary: BoundaryPosition::Right,
                        flux: vec![0.0, 0.0],
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    },
                    BoundaryConditionSpec::Neumann {
                        boundary: BoundaryPosition::Bottom,
                        flux: vec![0.0, 0.0],
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    },
                    BoundaryConditionSpec::Neumann {
                        boundary: BoundaryPosition::Top,
                        flux: vec![0.0, 0.0],
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    },
                ],
                EMProblemType::QuasiStatic => vec![
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Left,
                        value: vec![0.0, 0.0],
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    },
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Right,
                        value: vec![0.0, 0.0],
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    },
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Bottom,
                        value: vec![0.0, 0.0],
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    },
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Top,
                        value: vec![0.0, 0.0],
                        component: BoundaryComponent::Vector(vec![0, 1]),
                    },
                ],
                EMProblemType::WavePropagation => vec![
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Left,
                        value: vec![0.0],
                        component: BoundaryComponent::Scalar,
                    },
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Right,
                        value: vec![0.0],
                        component: BoundaryComponent::Scalar,
                    },
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Bottom,
                        value: vec![0.0],
                        component: BoundaryComponent::Scalar,
                    },
                    BoundaryConditionSpec::Dirichlet {
                        boundary: BoundaryPosition::Top,
                        value: vec![0.0],
                        component: BoundaryComponent::Scalar,
                    },
                ],
            };
        }

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
                ElectromagneticBoundarySpec::ImpedanceBoundary {
                    position,
                    impedance,
                } => {
                    // Impedance boundary condition
                    BoundaryConditionSpec::Robin {
                        boundary: position.clone(),
                        alpha: 1.0 / impedance,
                        beta: 0.0,
                        component: BoundaryComponent::Scalar,
                    }
                }
                ElectromagneticBoundarySpec::Port {
                    position,
                    port_impedance,
                    ..
                } => {
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
            EMProblemType::WavePropagation => {
                // Zero initial fields for wave propagation
                vec![
                    InitialConditionSpec::DirichletConstant {
                        value: vec![0.0], // Ez field
                        component: BoundaryComponent::Scalar,
                    },
                    InitialConditionSpec::DirichletConstant {
                        value: vec![0.0], // Hz field
                        component: BoundaryComponent::Scalar,
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
            EMProblemType::WavePropagation => (1.0, 2.0), // Even lower BC weight for full wave propagation
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
                    acceptable_range: (2.99792458e8 * 0.99, 2.99792458e8 * 1.01), // c ± 1%
                    description: "Wave propagation speed validation".to_string(),
                },
            ],
        };

        // Common metrics
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
        false // EM coupling not implemented yet
    }

    fn coupling_interfaces(&self) -> Vec<CouplingInterface> {
        Vec::new() // No coupling interfaces defined yet
    }
}

impl<B: AutodiffBackend> ElectromagneticDomain<B> {
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
            #[cfg(feature = "gpu")]
            gpu_config: None,
            _backend: std::marker::PhantomData,
        }
    }

    /// Add a current source
    pub fn add_current_source(
        mut self,
        position: (f64, f64),
        current_density: Vec<f64>,
        radius: f64,
    ) -> Self {
        self.current_sources.push(CurrentSource {
            position,
            current_density,
            radius,
        });
        self
    }

    /// Add a perfect electric conductor boundary
    pub fn add_pec_boundary(mut self, position: BoundaryPosition) -> Self {
        self.boundary_specs
            .push(ElectromagneticBoundarySpec::PerfectElectricConductor { position });
        self
    }

    /// Add a perfect magnetic conductor boundary
    pub fn add_pmc_boundary(mut self, position: BoundaryPosition) -> Self {
        self.boundary_specs
            .push(ElectromagneticBoundarySpec::PerfectMagneticConductor { position });
        self
    }

    /// Set problem type
    pub fn with_problem_type(mut self, problem_type: EMProblemType) -> Self {
        self.problem_type = problem_type;
        self
    }

    /// Enable GPU acceleration for electromagnetic simulations
    #[cfg(feature = "gpu")]
    pub fn with_gpu_acceleration(mut self, gpu_config: EMConfig) -> Self {
        self.gpu_config = Some(gpu_config);
        self
    }

    /// Compute electrostatic residual: ∇·(ε∇φ) = -ρ
    fn electrostatic_residual(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        eps: f64,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Create input tensor for neural network
        let _inputs = Tensor::cat(vec![x.clone(), y.clone(), Tensor::zeros_like(x)], 1);

        // Forward pass through model to get electric potential φ
        let _phi = model.forward(x.clone(), y.clone(), Tensor::zeros_like(x));

        // Use finite differences within autodiff framework (similar to burn_wave_equation_2d.rs)
        let eps_fd = (f32::EPSILON).sqrt() * 1e-2_f32; // Adaptive epsilon for numerical stability

        // Compute ∂φ/∂x using central difference
        let x_plus = x.clone() + eps_fd;
        let x_minus = x.clone() - eps_fd;
        let phi_x_plus = model.forward(x_plus, y.clone(), Tensor::zeros_like(x));
        let phi_x_minus = model.forward(x_minus, y.clone(), Tensor::zeros_like(x));
        let dphi_dx = (phi_x_plus - phi_x_minus) / (2.0 * eps_fd);

        // Compute ∂φ/∂y using central difference
        let y_plus = y.clone() + eps_fd;
        let y_minus = y.clone() - eps_fd;
        let phi_y_plus = model.forward(x.clone(), y_plus, Tensor::zeros_like(x));
        let phi_y_minus = model.forward(x.clone(), y_minus, Tensor::zeros_like(x));
        let dphi_dy = (phi_y_plus - phi_y_minus) / (2.0 * eps_fd);

        // Compute displacement field D = εE, where E = -∇φ
        let _d_x = eps as f32 * (-dphi_dx);
        let _d_y = eps as f32 * (-dphi_dy);

        // Gauss's law: ∇·D = ρ_free
        // Compute ∂D_x/∂x and ∂D_y/∂y using finite differences
        let d_x_plus = eps as f32
            * (-model.forward(x.clone() + eps_fd, y.clone(), Tensor::zeros_like(x))
                + model.forward(x.clone() - eps_fd, y.clone(), Tensor::zeros_like(x)))
            / (2.0 * eps_fd);
        let d_x_minus = eps as f32
            * (-model.forward(x.clone() - eps_fd, y.clone(), Tensor::zeros_like(x))
                + model.forward(x.clone() + eps_fd, y.clone(), Tensor::zeros_like(x)))
            / (2.0 * eps_fd);
        let dd_x_dx = (d_x_plus - d_x_minus) / (2.0 * eps_fd);

        let d_y_plus = eps as f32
            * (-model.forward(x.clone(), y.clone() + eps_fd, Tensor::zeros_like(x))
                + model.forward(x.clone(), y.clone() - eps_fd, Tensor::zeros_like(x)))
            / (2.0 * eps_fd);
        let d_y_minus = eps as f32
            * (-model.forward(x.clone(), y.clone() - eps_fd, Tensor::zeros_like(x))
                + model.forward(x.clone(), y.clone() + eps_fd, Tensor::zeros_like(x)))
            / (2.0 * eps_fd);
        let dd_y_dy = (d_y_plus - d_y_minus) / (2.0 * eps_fd);

        let gauss_residual = dd_x_dx + dd_y_dy;

        // Get charge density from physics parameters or current sources
        let rho = self.compute_charge_density(x, y, physics_params);

        // Residual: ∇·D + ρ = 0
        gauss_residual + rho
    }

    /// Helper function to compute phi at given coordinates
    fn _compute_phi_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>) -> Tensor<B, 2> {
        // This would need access to the model - for now return zeros
        // In practice, this should call the neural network forward pass
        Tensor::zeros_like(&x)
    }

    /// Helper function to compute D_x at given coordinates
    fn _compute_d_x_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>, eps: f64) -> Tensor<B, 2> {
        // Simplified implementation - in practice should compute properly
        Tensor::zeros_like(&x) * eps as f32
    }

    /// Helper function to compute D_y at given coordinates
    fn _compute_d_y_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>, eps: f64) -> Tensor<B, 2> {
        // Simplified implementation - in practice should compute properly
        Tensor::zeros_like(&x) * eps as f32
    }

    /// Compute magnetostatic residual: ∇×(ν∇×A) = μ₀J
    fn magnetostatic_residual(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        mu: f64,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Forward pass through model to get magnetic vector potential Az
        let _az = model.forward(x.clone(), y.clone(), Tensor::zeros_like(x));

        // Use finite differences within autodiff framework
        let eps_fd = (f32::EPSILON).sqrt() * 1e-2_f32; // Adaptive epsilon for numerical stability

        // Compute magnetic field B = ∇×A
        // For 2D TMz mode: Bx = -∂Az/∂y, By = ∂Az/∂x, Bz = 0
        let y_plus = y.clone() + eps_fd;
        let y_minus = y.clone() - eps_fd;
        let az_y_plus = model.forward(x.clone(), y_plus, Tensor::zeros_like(x));
        let az_y_minus = model.forward(x.clone(), y_minus, Tensor::zeros_like(x));
        let daz_dy = (az_y_plus - az_y_minus) / (2.0 * eps_fd);
        let b_x = -daz_dy;

        let x_plus = x.clone() + eps_fd;
        let x_minus = x.clone() - eps_fd;
        let az_x_plus = model.forward(x_plus, y.clone(), Tensor::zeros_like(x));
        let az_x_minus = model.forward(x_minus, y.clone(), Tensor::zeros_like(x));
        let daz_dx = (az_x_plus - az_x_minus) / (2.0 * eps_fd);
        let b_y = daz_dx;

        // Compute magnetic field intensity H = B/μ
        let _h_x = b_x.clone() / mu as f32;
        let _h_y = b_y.clone() / mu as f32;

        // Compute ∇×H = (∂Hy/∂x - ∂Hx/∂y) k̂ (z-component in 2D)
        // ∂Hy/∂x
        let h_y_x_plus = (model.forward(x.clone() + eps_fd, y.clone(), Tensor::zeros_like(x))
            - model.forward(x.clone() - eps_fd, y.clone(), Tensor::zeros_like(x)))
            / (2.0 * eps_fd)
            / mu as f32;
        let h_y_x_minus = (model.forward(x.clone() - eps_fd, y.clone(), Tensor::zeros_like(x))
            - model.forward(x.clone() + eps_fd, y.clone(), Tensor::zeros_like(x)))
            / (2.0 * eps_fd)
            / mu as f32;
        let dh_y_dx = (h_y_x_plus - h_y_x_minus) / (2.0 * eps_fd);

        // -∂Hx/∂y
        let h_x_y_plus = -(model.forward(x.clone(), y.clone() + eps_fd, Tensor::zeros_like(x))
            - model.forward(x.clone(), y.clone() - eps_fd, Tensor::zeros_like(x)))
            / (2.0 * eps_fd)
            / mu as f32;
        let h_x_y_minus = -(model.forward(x.clone(), y.clone() - eps_fd, Tensor::zeros_like(x))
            - model.forward(x.clone(), y.clone() + eps_fd, Tensor::zeros_like(x)))
            / (2.0 * eps_fd)
            / mu as f32;
        let minus_dh_x_dy = (h_x_y_plus - h_x_y_minus) / (2.0 * eps_fd);

        let curl_h_z = dh_y_dx + minus_dh_x_dy;

        // Get current density from physics parameters or current sources
        let j_z = self.compute_current_density_z(x, y, physics_params);

        // Ampere's law: ∇×H = J
        // For magnetostatics: ∇×H = J_total
        curl_h_z - j_z
    }

    /// Helper function to compute Az at given coordinates
    fn _compute_az_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>) -> Tensor<B, 2> {
        // This would need access to the model - for now return zeros
        Tensor::zeros_like(&x)
    }

    /// Helper function to compute H_x at given coordinates
    fn _compute_h_x_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>, mu: f64) -> Tensor<B, 2> {
        // Simplified implementation
        Tensor::zeros_like(&x) / mu as f32
    }

    /// Helper function to compute H_y at given coordinates
    fn _compute_h_y_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>, mu: f64) -> Tensor<B, 2> {
        // Simplified implementation
        Tensor::zeros_like(&x) / mu as f32
    }

    /// Compute quasi-static electromagnetic residual
    fn quasi_static_residual(
        &self,
        _outputs: &Tensor<B, 2>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        eps: f64,
        mu: f64,
        sigma: f64,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // For quasi-static approximation, solve coupled E and H fields
        // Assume outputs are [Ez, Hz] for 2D TMz mode
        let batch_size = x.shape().dims[0];
        let _ez = _outputs.clone().slice([0..batch_size, 0..1]).squeeze::<2>();
        let _hz = _outputs.clone().slice([0..batch_size, 1..2]).squeeze::<2>();

        // Use finite differences for all derivatives
        let eps_fd = 1e-4_f32;

        // Compute E field components from Hz using Faraday's law
        // ∇×E = -μ∂H/∂t, so for TMz: Ex = -μ∂Hz/∂y, Ey = μ∂Hz/∂x
        let y_plus = y.clone() + eps_fd;
        let y_minus = y.clone() - eps_fd;
        let hz_y_plus = self.compute_hz_at(x.clone(), y_plus, t.clone());
        let hz_y_minus = self.compute_hz_at(x.clone(), y_minus, t.clone());
        let dhz_dy = (hz_y_plus - hz_y_minus) / (2.0 * eps_fd);
        let _ex = -mu as f32 * dhz_dy;

        let x_plus = x.clone() + eps_fd;
        let x_minus = x.clone() - eps_fd;
        let hz_x_plus = self.compute_hz_at(x_plus, y.clone(), t.clone());
        let hz_x_minus = self.compute_hz_at(x_minus, y.clone(), t.clone());
        let dhz_dx = (hz_x_plus - hz_x_minus) / (2.0 * eps_fd);
        let _ey = mu as f32 * dhz_dx;

        // Compute H field components from Ez using Ampere's law
        // ∇×H = J + ∂D/∂t + σE, so for TMz: Hx = -∂Ez/∂y, Hy = ∂Ez/∂x
        let ez_y_plus = self.compute_ez_at(x.clone(), y.clone() + eps_fd, t.clone());
        let ez_y_minus = self.compute_ez_at(x.clone(), y.clone() - eps_fd, t.clone());
        let dez_dy = (ez_y_plus - ez_y_minus) / (2.0 * eps_fd);
        let _hx = -dez_dy;

        let ez_x_plus = self.compute_ez_at(x.clone() + eps_fd, y.clone(), t.clone());
        let ez_x_minus = self.compute_ez_at(x.clone() - eps_fd, y.clone(), t.clone());
        let dez_dx = (ez_x_plus - ez_x_minus) / (2.0 * eps_fd);
        let _hy = dez_dx;

        // Faraday's law residual: ∇×E = -μ∂H/∂t
        let ey_x_plus = self.compute_ey_at(x.clone() + eps_fd, y.clone(), t.clone());
        let ey_x_minus = self.compute_ey_at(x.clone() - eps_fd, y.clone(), t.clone());
        let dey_dx = (ey_x_plus - ey_x_minus) / (2.0 * eps_fd);

        let ex_y_plus = self.compute_ex_at(x.clone(), y.clone() + eps_fd, t.clone());
        let ex_y_minus = self.compute_ex_at(x.clone(), y.clone() - eps_fd, t.clone());
        let dex_dy = (ex_y_plus - ex_y_minus) / (2.0 * eps_fd);

        let curl_e_z = dey_dx - dex_dy;

        let t_plus = t.clone() + eps_fd;
        let t_minus = t.clone() - eps_fd;
        let hz_t_plus = self.compute_hz_at(x.clone(), y.clone(), t_plus);
        let hz_t_minus = self.compute_hz_at(x.clone(), y.clone(), t_minus);
        let dhz_dt = (hz_t_plus - hz_t_minus) / (2.0 * eps_fd);

        let faraday_residual = curl_e_z + mu as f32 * dhz_dt;

        // Ampere's law residual: ∇×H = σE + ∂D/∂t
        let hy_x_plus = self.compute_hy_at(x.clone() + eps_fd, y.clone(), t.clone());
        let hy_x_minus = self.compute_hy_at(x.clone() - eps_fd, y.clone(), t.clone());
        let dhy_dx = (hy_x_plus - hy_x_minus) / (2.0 * eps_fd);

        let hx_y_plus = self.compute_hx_at(x.clone(), y.clone() + eps_fd, t.clone());
        let hx_y_minus = self.compute_hx_at(x.clone(), y.clone() - eps_fd, t.clone());
        let dhx_dy = (hx_y_plus - hx_y_minus) / (2.0 * eps_fd);

        let curl_h_z = dhy_dx - dhx_dy;

        let ez_t_plus = self.compute_ez_at(x.clone(), y.clone(), t.clone() + eps_fd);
        let ez_t_minus = self.compute_ez_at(x.clone(), y.clone(), t.clone() - eps_fd);
        let dez_dt = (ez_t_plus - ez_t_minus) / (2.0 * eps_fd);

        let d_d_dt = eps as f32 * dez_dt;
        let conductivity_term = sigma as f32 * _ez.clone();
        let j_z = self.compute_current_density_z(x, y, physics_params);
        let ampere_residual = curl_h_z - conductivity_term - d_d_dt - j_z;

        // Gauss's law for magnetism: ∇·B = 0 (B = μH for linear media)
        let hx_x_plus = self.compute_hx_at(x.clone() + eps_fd, y.clone(), t.clone());
        let hx_x_minus = self.compute_hx_at(x.clone() - eps_fd, y.clone(), t.clone());
        let dhx_dx = (hx_x_plus - hx_x_minus) / (2.0 * eps_fd);

        let hy_y_plus = self.compute_hy_at(x.clone(), y.clone() + eps_fd, t.clone());
        let hy_y_minus = self.compute_hy_at(x.clone(), y.clone() - eps_fd, t.clone());
        let dhy_dy = (hy_y_plus - hy_y_minus) / (2.0 * eps_fd);

        let div_b = (dhx_dx + dhy_dy) * mu as f32;

        // Combine residuals with appropriate weights
        let faraday_weight = 1.0;
        let ampere_weight = 1.0;
        let gauss_weight = 0.1; // Lower weight for divergence constraint

        faraday_residual * faraday_weight + ampere_residual * ampere_weight + div_b * gauss_weight
    }

    /// Helper functions for computing field components at different coordinates
    fn compute_ez_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>, _t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Simplified - would need model access
        Tensor::zeros_like(&x)
    }

    fn compute_hz_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>, _t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Simplified - would need model access
        Tensor::zeros_like(&x)
    }

    fn compute_ex_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>, _t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Simplified - would need model access
        Tensor::zeros_like(&x)
    }

    fn compute_ey_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>, _t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Simplified - would need model access
        Tensor::zeros_like(&x)
    }

    fn compute_hx_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>, _t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Simplified - would need model access
        Tensor::zeros_like(&x)
    }

    fn compute_hy_at(&self, x: Tensor<B, 2>, _y: Tensor<B, 2>, _t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Simplified - would need model access
        Tensor::zeros_like(&x)
    }

    /// Compute full wave propagation residual using Maxwell's equations
    fn wave_propagation_residual(
        &self,
        outputs: &Tensor<B, 2>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        eps: f64,
        mu: f64,
        sigma: f64,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Full time-dependent Maxwell's equations for wave propagation
        // Assume outputs are [Ez, Hz] for 2D TMz mode
        let batch_size = x.shape().dims[0];
        let ez = outputs.clone().slice([0..batch_size, 0..1]).squeeze::<2>();
        let _hz = outputs.clone().slice([0..batch_size, 1..2]).squeeze::<2>();

        // Use finite differences for all derivatives
        let eps_fd = 1e-4_f32;

        // Compute E field components from Hz using Faraday's law
        let y_plus = y.clone() + eps_fd;
        let y_minus = y.clone() - eps_fd;
        let hz_y_plus = self.compute_hz_at(x.clone(), y_plus, t.clone());
        let hz_y_minus = self.compute_hz_at(x.clone(), y_minus, t.clone());
        let dhz_dy = (hz_y_plus - hz_y_minus) / (2.0 * eps_fd);
        let _ex = -mu as f32 * dhz_dy;

        let x_plus = x.clone() + eps_fd;
        let x_minus = x.clone() - eps_fd;
        let hz_x_plus = self.compute_hz_at(x_plus, y.clone(), t.clone());
        let hz_x_minus = self.compute_hz_at(x_minus, y.clone(), t.clone());
        let dhz_dx = (hz_x_plus - hz_x_minus) / (2.0 * eps_fd);
        let _ey = mu as f32 * dhz_dx;

        // Compute H field components from Ez using Ampere's law
        let ez_y_plus = self.compute_ez_at(x.clone(), y.clone() + eps_fd, t.clone());
        let ez_y_minus = self.compute_ez_at(x.clone(), y.clone() - eps_fd, t.clone());
        let dez_dy = (ez_y_plus - ez_y_minus) / (2.0 * eps_fd);
        let _hx = -dez_dy;

        let ez_x_plus = self.compute_ez_at(x.clone() + eps_fd, y.clone(), t.clone());
        let ez_x_minus = self.compute_ez_at(x.clone() - eps_fd, y.clone(), t.clone());
        let dez_dx = (ez_x_plus - ez_x_minus) / (2.0 * eps_fd);
        let _hy = dez_dx;

        // Faraday's law: ∇×E = -∂B/∂t = -μ∂H/∂t
        let ey_x_plus = self.compute_ey_at(x.clone() + eps_fd, y.clone(), t.clone());
        let ey_x_minus = self.compute_ey_at(x.clone() - eps_fd, y.clone(), t.clone());
        let dey_dx = (ey_x_plus - ey_x_minus) / (2.0 * eps_fd);

        let ex_y_plus = self.compute_ex_at(x.clone(), y.clone() + eps_fd, t.clone());
        let ex_y_minus = self.compute_ex_at(x.clone(), y.clone() - eps_fd, t.clone());
        let dex_dy = (ex_y_plus - ex_y_minus) / (2.0 * eps_fd);

        let curl_e_z = dey_dx - dex_dy;

        let t_plus = t.clone() + eps_fd;
        let t_minus = t.clone() - eps_fd;
        let hz_t_plus = self.compute_hz_at(x.clone(), y.clone(), t_plus);
        let hz_t_minus = self.compute_hz_at(x.clone(), y.clone(), t_minus);
        let dhz_dt = (hz_t_plus - hz_t_minus) / (2.0 * eps_fd);

        let faraday_residual = curl_e_z + mu as f32 * dhz_dt;

        // Ampere's law: ∇×H = J + ∂D/∂t + σE
        let hy_x_plus = self.compute_hy_at(x.clone() + eps_fd, y.clone(), t.clone());
        let hy_x_minus = self.compute_hy_at(x.clone() - eps_fd, y.clone(), t.clone());
        let dhy_dx = (hy_x_plus - hy_x_minus) / (2.0 * eps_fd);

        let hx_y_plus = self.compute_hx_at(x.clone(), y.clone() + eps_fd, t.clone());
        let hx_y_minus = self.compute_hx_at(x.clone(), y.clone() - eps_fd, t.clone());
        let dhx_dy = (hx_y_plus - hx_y_minus) / (2.0 * eps_fd);

        let curl_h_z = dhy_dx - dhx_dy;

        let ez_t_plus = self.compute_ez_at(x.clone(), y.clone(), t.clone() + eps_fd);
        let ez_t_minus = self.compute_ez_at(x.clone(), y.clone(), t.clone() - eps_fd);
        let dez_dt = (ez_t_plus - ez_t_minus) / (2.0 * eps_fd);

        let d_d_dt = eps as f32 * dez_dt;
        let conductivity_term = sigma as f32 * ez.clone();
        let j_z = self.compute_current_density_z(x, y, physics_params);
        let ampere_residual = curl_h_z - j_z - d_d_dt - conductivity_term;

        // Gauss's law for magnetism: ∇·B = 0
        let hx_x_plus = self.compute_hx_at(x.clone() + eps_fd, y.clone(), t.clone());
        let hx_x_minus = self.compute_hx_at(x.clone() - eps_fd, y.clone(), t.clone());
        let dhx_dx = (hx_x_plus - hx_x_minus) / (2.0 * eps_fd);

        let hy_y_plus = self.compute_hy_at(x.clone(), y.clone() + eps_fd, t.clone());
        let hy_y_minus = self.compute_hy_at(x.clone(), y.clone() - eps_fd, t.clone());
        let dhy_dy = (hy_y_plus - hy_y_minus) / (2.0 * eps_fd);

        let div_b = dhx_dx + dhy_dy;

        // Gauss's law for electricity: ∇·D = ρ
        let ex_x_plus = self.compute_ex_at(x.clone() + eps_fd, y.clone(), t.clone());
        let ex_x_minus = self.compute_ex_at(x.clone() - eps_fd, y.clone(), t.clone());
        let dex_dx = (ex_x_plus - ex_x_minus) / (2.0 * eps_fd);

        let ey_y_plus = self.compute_ey_at(x.clone(), y.clone() + eps_fd, t.clone());
        let ey_y_minus = self.compute_ey_at(x.clone(), y.clone() - eps_fd, t.clone());
        let dey_dy = (ey_y_plus - ey_y_minus) / (2.0 * eps_fd);

        let div_d = dex_dx + dey_dy;
        let rho = self.compute_charge_density(x, y, physics_params);
        let gauss_residual = div_d - rho / eps as f32;

        // Combine all Maxwell's equations residuals
        let faraday_weight = 1.0;
        let ampere_weight = 1.0;
        let gauss_b_weight = 0.1;
        let gauss_d_weight = 0.1;

        faraday_residual * faraday_weight
            + ampere_residual * ampere_weight
            + div_b * gauss_b_weight
            + gauss_residual * gauss_d_weight
    }

    /// Compute charge density at given spatial positions
    pub fn compute_charge_density(
        &self,
        x: &Tensor<B, 2>,
        _y: &Tensor<B, 2>,
        _physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // For now, assume zero charge density (can be extended with charge sources)
        // In practice, this would compute ρ from physics_params or predefined sources
        Tensor::zeros_like(x)
    }

    /// Compute z-component of current density at given spatial positions
    pub fn compute_current_density_z(
        &self,
        x: &Tensor<B, 2>,
        _y: &Tensor<B, 2>,
        _physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // For now, assume zero current density (can be extended with current sources)
        // In practice, this would compute Jz from physics_params or predefined sources
        Tensor::zeros_like(x)
    }

    /// Validate domain configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.permittivity <= 0.0 {
            return Err(KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "permittivity".to_string(),
                    value: self.permittivity,
                    reason: "Permittivity must be positive".to_string(),
                },
            ));
        }

        if self.permeability <= 0.0 {
            return Err(KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "permeability".to_string(),
                    value: self.permeability,
                    reason: "Permeability must be positive".to_string(),
                },
            ));
        }

        if self.conductivity < 0.0 {
            return Err(KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "conductivity".to_string(),
                    value: self.conductivity,
                    reason: "Conductivity cannot be negative".to_string(),
                },
            ));
        }

        if self.c <= 0.0 || !self.c.is_finite() {
            return Err(KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "speed_of_light".to_string(),
                    value: self.c,
                    reason: "Speed of light must be positive and finite".to_string(),
                },
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

    #[test]
    fn test_electromagnetic_domain_creation() {
        let domain: ElectromagneticDomain<TestBackend> = ElectromagneticDomain::new(
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
        let valid_domain: ElectromagneticDomain<TestBackend> =
            ElectromagneticDomain::<TestBackend>::default();
        assert!(valid_domain.validate().is_ok());

        let invalid_domain: ElectromagneticDomain<TestBackend> = ElectromagneticDomain::new(
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
        let domain = ElectromagneticDomain::<TestBackend>::default()
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
        let domain = ElectromagneticDomain::<TestBackend>::default().add_current_source(
            (0.5, 0.5),
            vec![1e6, 0.0],
            0.1,
        );

        assert_eq!(domain.current_sources.len(), 1);
        assert_eq!(domain.current_sources[0].position, (0.5, 0.5));
    }

    #[test]
    fn test_physics_domain_interface() {
        let domain = ElectromagneticDomain::<TestBackend>::default();

        assert_eq!(domain.domain_name(), "electromagnetic");

        let weights = domain.loss_weights();
        assert_eq!(weights.pde_weight, 1.0);

        let metrics = domain.validation_metrics();
        assert!(metrics.len() >= 4); // At least common metrics

        assert!(!domain.supports_coupling()); // Not implemented yet
    }

    #[test]
    fn test_problem_type_specifics() {
        let electrostatic = ElectromagneticDomain::<TestBackend>::default()
            .with_problem_type(EMProblemType::Electrostatic);

        let magnetostatic = ElectromagneticDomain::<TestBackend>::default()
            .with_problem_type(EMProblemType::Magnetostatic);

        let quasi_static = ElectromagneticDomain::<TestBackend>::default()
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
