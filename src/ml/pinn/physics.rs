//! Physics Domain Framework for PINN
//!
//! This module provides a modular framework for implementing different physics domains
//! within the PINN ecosystem, enabling rapid extension to new physics while maintaining
//! a unified training and inference interface.

use crate::error::{KwaversError, KwaversResult};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

/// PDE Characteristics for adaptive sampling
#[derive(Debug, Clone)]
pub struct PDECharacteristics {
    /// Type of equation ("wave", "diffusion", "navier_stokes", etc.)
    pub equation_type: String,
    /// Wave propagation speed (for wave equations)
    pub wave_speed: Option<f64>,
    /// Diffusion coefficient (for diffusion equations)
    pub diffusion_coefficient: f64,
    /// Reynolds number (for fluid dynamics)
    pub reynolds_number: Option<f64>,
}

impl Default for PDECharacteristics {
    fn default() -> Self {
        Self {
            equation_type: "generic".to_string(),
            wave_speed: None,
            diffusion_coefficient: 1.0,
            reynolds_number: None,
        }
    }
}

/// Physics domain trait defining the interface for any physics domain
pub trait PhysicsDomain<B: AutodiffBackend>: std::fmt::Debug {
    /// Get the physics domain name
    fn domain_name(&self) -> &'static str;

    /// Get PDE characteristics for adaptive sampling
    fn pde_characteristics(&self) -> PDECharacteristics {
        PDECharacteristics::default()
    }

    /// Compute PDE residual for this physics domain
    fn pde_residual(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2>;

    /// Get boundary condition specifications
    fn boundary_conditions(&self) -> Vec<BoundaryConditionSpec>;

    /// Get initial condition specifications
    fn initial_conditions(&self) -> Vec<InitialConditionSpec>;

    /// Get physics-specific loss weights
    fn loss_weights(&self) -> PhysicsLossWeights;

    /// Get validation metrics for this physics domain
    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric>;

    /// Check if this domain supports multi-physics coupling
    fn supports_coupling(&self) -> bool {
        false
    }

    /// Get coupling interfaces if supported
    fn coupling_interfaces(&self) -> Vec<CouplingInterface> {
        Vec::new()
    }
}

/// Physics parameters container
#[derive(Debug, Clone)]
pub struct PhysicsParameters {
    /// Material properties (density, viscosity, etc.)
    pub material_properties: HashMap<String, f64>,
    /// Boundary condition values
    pub boundary_values: HashMap<String, Vec<f64>>,
    /// Initial condition values
    pub initial_values: HashMap<String, Vec<f64>>,
    /// Domain-specific parameters
    pub domain_params: HashMap<String, f64>,
}

/// Boundary condition specification
#[derive(Debug, Clone)]
pub enum BoundaryConditionSpec {
    /// Dirichlet boundary condition: u = g
    Dirichlet {
        boundary: BoundaryPosition,
        value: Vec<f64>,
        component: BoundaryComponent,
    },
    /// Neumann boundary condition: ∂u/∂n = g
    Neumann {
        boundary: BoundaryPosition,
        flux: Vec<f64>,
        component: BoundaryComponent,
    },
    /// Robin boundary condition: ∂u/∂n + αu = g
    Robin {
        boundary: BoundaryPosition,
        alpha: f64,
        beta: f64,
        component: BoundaryComponent,
    },
}

/// Boundary position specification
#[derive(Debug, Clone)]
pub enum BoundaryPosition {
    /// Left boundary (x = x_min)
    Left,
    /// Right boundary (x = x_max)
    Right,
    /// Bottom boundary (y = y_min)
    Bottom,
    /// Top boundary (y = y_max)
    Top,
    /// Arbitrary boundary defined by coordinates
    CustomRectangular {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    },
}

/// Boundary component (for multi-component physics)
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryComponent {
    /// Scalar field
    Scalar,
    /// Vector field component
    VelocityX,
    VelocityY,
    /// Multi-component field
    Vector(Vec<usize>),
    /// Custom component
    Custom(String),
}

/// Initial condition specification
#[derive(Debug, Clone)]
pub enum InitialConditionSpec {
    /// Initial condition: u(x,y,0) = constant value
    DirichletConstant {
        value: Vec<f64>,
        component: BoundaryComponent,
    },
    /// Initial derivative condition: ∂u/∂t(x,y,0) = constant flux
    NeumannConstant {
        flux: Vec<f64>,
        component: BoundaryComponent,
    },
}

/// Physics-specific loss weights
#[derive(Debug, Clone)]
pub struct PhysicsLossWeights {
    /// PDE residual weight
    pub pde_weight: f64,
    /// Boundary condition weight
    pub boundary_weight: f64,
    /// Initial condition weight
    pub initial_weight: f64,
    /// Additional physics-specific weights
    pub physics_weights: HashMap<String, f64>,
}

impl Default for PhysicsLossWeights {
    fn default() -> Self {
        Self {
            pde_weight: 1.0,
            boundary_weight: 10.0,
            initial_weight: 10.0,
            physics_weights: HashMap::new(),
        }
    }
}

/// Physics validation metric
#[derive(Debug, Clone)]
pub struct PhysicsValidationMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Acceptable range (min, max)
    pub acceptable_range: (f64, f64),
    /// Metric description
    pub description: String,
}

/// Multi-physics coupling interface
#[derive(Debug, Clone)]
pub struct CouplingInterface {
    /// Interface name
    pub name: String,
    /// Interface position
    pub position: BoundaryPosition,
    /// Coupled physics domains
    pub coupled_domains: Vec<String>,
    /// Coupling type
    pub coupling_type: CouplingType,
    /// Coupling strength/parameters
    pub coupling_params: HashMap<String, f64>,
}

/// Type of physics coupling
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingType {
    /// Continuity of solution and flux
    Conjugate,
    /// Continuity of solution only
    SolutionContinuity,
    /// Continuity of flux only
    FluxContinuity,
    /// Custom coupling relationship
    Custom(String),
}

/// Physics domain registry for managing available physics domains
#[derive(Debug)]
pub struct PhysicsDomainRegistry<B: AutodiffBackend> {
    domains: HashMap<String, Box<dyn PhysicsDomain<B> + Send + Sync>>,
}

impl<B: AutodiffBackend> PhysicsDomainRegistry<B> {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            domains: HashMap::new(),
        }
    }

    /// Register a physics domain
    pub fn register_domain<D>(&mut self, domain: D) -> KwaversResult<()>
    where
        D: PhysicsDomain<B> + Send + Sync + 'static,
    {
        let name = domain.domain_name().to_string();
        let boxed: Box<dyn PhysicsDomain<B> + Send + Sync> = Box::new(domain);
        self.domains.insert(name, boxed);
        Ok(())
    }

    /// Get a physics domain by name
    pub fn get_domain(&self, name: &str) -> Option<&(dyn PhysicsDomain<B> + Send + Sync)> {
        self.domains.get(name).map(|d| d.as_ref())
    }

    /// List all registered domains
    pub fn list_domains(&self) -> Vec<String> {
        self.domains.keys().cloned().collect()
    }

    /// Check if a domain is registered
    pub fn has_domain(&self, name: &str) -> bool {
        self.domains.contains_key(name)
    }

    /// Remove a domain
    pub fn remove_domain(&mut self, name: &str) -> KwaversResult<()> {
        if self.domains.remove(name).is_some() {
            Ok(())
        } else {
            Err(KwaversError::System(
                crate::error::SystemError::ResourceUnavailable {
                    resource: format!("physics domain {}", name),
                },
            ))
        }
    }
}

impl<B: AutodiffBackend> Default for PhysicsDomainRegistry<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_domain_registry() {
        use burn::backend::Autodiff;
        let mut registry = PhysicsDomainRegistry::<Autodiff<burn::backend::NdArray<f32>>>::new();

        // Test empty registry
        assert!(!registry.has_domain("test"));
        assert_eq!(registry.list_domains().len(), 0);

        // Note: We can't easily test domain registration without concrete implementations
        // This would be tested with actual physics domain implementations
    }

    #[test]
    fn test_physics_loss_weights() {
        let weights = PhysicsLossWeights::default();
        assert_eq!(weights.pde_weight, 1.0);
        assert_eq!(weights.boundary_weight, 10.0);
        assert_eq!(weights.initial_weight, 10.0);
        assert!(weights.physics_weights.is_empty());
    }
}
