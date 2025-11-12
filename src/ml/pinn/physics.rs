//! Physics Domain Framework for PINN
//!
//! This module provides a modular framework for implementing different physics domains
//! within the PINN ecosystem, enabling rapid extension to new physics while maintaining
//! a unified training and inference interface.

use crate::error::{KwaversError, KwaversResult};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

/// Physics domain trait defining the interface for any physics domain
pub trait PhysicsDomain<B: AutodiffBackend> {
    /// Get the physics domain name
    fn domain_name(&self) -> &'static str;

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
    CustomRectangular { x_min: f64, x_max: f64, y_min: f64, y_max: f64 },
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
            Err(KwaversError::System(crate::error::SystemError::ResourceUnavailable {
                resource: format!("physics domain {}", name),
            }))
        }
    }
}

impl<B: AutodiffBackend> Default for PhysicsDomainRegistry<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Universal PINN solver that can work with any physics domain
pub struct UniversalPINNSolver<B: AutodiffBackend> {
    /// Physics domain registry
    physics_registry: PhysicsDomainRegistry<B>,
    /// Neural network model
    model: Option<crate::ml::pinn::BurnPINN2DWave<B>>,
    /// Current physics domain
    current_domain: Option<String>,
    /// Training configuration
    config: UniversalTrainingConfig,
    /// Performance statistics
    stats: UniversalSolverStats,
}

/// Universal training configuration
#[derive(Debug, Clone)]
pub struct UniversalTrainingConfig {
    /// Number of epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Collocation points per dimension
    pub collocation_points: usize,
    /// Boundary points
    pub boundary_points: usize,
    /// Initial condition points
    pub initial_points: usize,
    /// Physics-aware adaptive sampling
    pub adaptive_sampling: bool,
    /// Early stopping patience
    pub patience: usize,
}

impl Default for UniversalTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1000,
            learning_rate: 0.001,
            batch_size: 32,
            collocation_points: 1000,
            boundary_points: 200,
            initial_points: 100,
            adaptive_sampling: true,
            patience: 50,
        }
    }
}

/// Universal solver statistics
#[derive(Debug, Clone)]
pub struct UniversalSolverStats {
    /// Training time
    pub training_time: std::time::Duration,
    /// Final loss values
    pub final_losses: HashMap<String, f64>,
    /// Convergence history
    pub loss_history: Vec<HashMap<String, f64>>,
    /// Physics validation metrics
    pub physics_metrics: Vec<PhysicsValidationMetric>,
}

impl<B: AutodiffBackend> UniversalPINNSolver<B> {
    /// Create a new universal PINN solver
    pub fn new() -> KwaversResult<Self> {
        Ok(Self {
            physics_registry: PhysicsDomainRegistry::new(),
            model: None,
            current_domain: None,
            config: UniversalTrainingConfig::default(),
            stats: UniversalSolverStats::default(),
        })
    }

    /// Register a physics domain
    pub fn register_physics_domain<D>(&mut self, domain: D) -> KwaversResult<()>
    where
        D: PhysicsDomain<B> + Send + Sync + 'static,
        B: AutodiffBackend + 'static,
    {
        self.physics_registry.register_domain(domain)
    }

    /// Configure for a specific physics domain
    pub fn configure_for_domain(&mut self, domain_name: &str) -> KwaversResult<()> {
        if !self.physics_registry.has_domain(domain_name) {
            return Err(KwaversError::System(crate::error::SystemError::InvalidConfiguration {
                parameter: "domain".to_string(),
                reason: format!("Physics domain '{}' not registered", domain_name),
            }));
        }

        self.current_domain = Some(domain_name.to_string());

        // Initialize model if needed
        if self.model.is_none() {
            // Initialize physics-informed neural network model
            // Reference: Raissi et al. (2019) "Physics-informed neural networks"
            use burn::backend::wgpu::WgpuDevice;

            let device = WgpuDevice::default();

            // Create default model configuration for physics domain
            // This provides mathematical stability while maintaining framework completeness
            let model_config = crate::ml::pinn::BurnPINN2DConfig {
                hidden_layers: vec![64, 64, 32], // Standard architecture for 2D wave equation
                learning_rate: 0.001,
                weight_decay: 1e-4,
                use_residual_connections: true,
                activation: crate::ml::pinn::Activation::Tanh,
                output_activation: None,
            };

            // Initialize the neural network model
            // This ensures runtime stability and mathematical completeness
            let pinn_model = crate::ml::pinn::BurnPINN2DWave::new(model_config, &device)?;
            self.model = Some(pinn_model);
        }

        Ok(())
    }

    /// Generate physics-aware collocation points
    pub fn generate_collocation_points(
        &self,
        geometry: &crate::ml::pinn::Geometry2D,
        domain: &dyn PhysicsDomain<B>,
    ) -> KwaversResult<Vec<(f64, f64, f64)>> {
        let mut points = Vec::new();
        let num_points = self.config.collocation_points;

        // Generate points within geometry bounds
        let bounds = geometry.bounding_box();
        let (x_min, x_max, y_min, y_max) = bounds;

        for _ in 0..num_points {
            let x = x_min + (x_max - x_min) * rand::random::<f64>();
            let y = y_min + (y_max - y_min) * rand::random::<f64>();
            let t = rand::random::<f64>() * 1.0; // Assuming t ∈ [0, 1]

            if geometry.contains(x, y) {
                points.push((x, y, t));
            }
        }

        // If adaptive sampling is enabled, refine based on physics
        if self.config.adaptive_sampling {
            points = self.adaptive_physics_sampling(points, domain);
        }

        Ok(points)
    }

    /// Adaptive physics-aware sampling based on PDE characteristics
    fn adaptive_physics_sampling(
        &self,
        mut points: Vec<(f64, f64, f64)>,
        domain: &dyn PhysicsDomain<B>,
    ) -> Vec<(f64, f64, f64)> {
        if points.is_empty() {
            return points;
        }

        // Analyze PDE characteristics to determine sampling strategy
        let pde_characteristics = domain.pde_characteristics();

        // Adapt sampling based on PDE properties
        match pde_characteristics.equation_type.as_str() {
            "wave" => {
                // For wave equations, concentrate points near wavefronts and boundaries
                self.adapt_wave_equation_sampling(&mut points, &pde_characteristics);
            }
            "diffusion" => {
                // For diffusion equations, concentrate points in high-gradient regions
                self.adapt_diffusion_equation_sampling(&mut points, &pde_characteristics);
            }
            "navier_stokes" => {
                // For fluid dynamics, concentrate points near boundaries and vortices
                self.adapt_fluid_dynamics_sampling(&mut points, &pde_characteristics);
            }
            _ => {
                // Default: use residual-based adaptive sampling
                self.adapt_residual_based_sampling(&mut points, domain);
            }
        }

        points
    }

    /// Adapt sampling for wave equations (concentrate near wavefronts)
    fn adapt_wave_equation_sampling(&self, points: &mut Vec<(f64, f64, f64)>, characteristics: &PDECharacteristics) {
        // For wave equations, add points near expected wavefront locations
        let wave_speed = characteristics.wave_speed.unwrap_or(1.0);
        let time_points: Vec<f64> = points.iter().map(|&(_, _, t)| t).collect();
        let max_time = time_points.iter().cloned().fold(0.0, f64::max);

        // Add points along expected wavefronts
        let wavefront_points = (0..10).map(|i| {
            let radius = wave_speed * max_time * (i as f64 / 9.0);
            (radius, 0.0, max_time) // Simplified circular wavefront
        });

        points.extend(wavefront_points);
    }

    /// Adapt sampling for diffusion equations (concentrate in high-gradient regions)
    fn adapt_diffusion_equation_sampling(&self, points: &mut Vec<(f64, f64, f64)>, _characteristics: &PDECharacteristics) {
        // Implement adaptive sampling based on PDE characteristics for diffusion equations
        // Following Raissi et al. (2019) adaptive sampling strategy
        let mut additional_points = Vec::new();

        for &(x, y, t) in points.iter() {
            // Estimate spatial gradients using finite differences (simplified)
            // In full implementation, would use solution gradients from PINN
            let dx = 0.01; // Small perturbation for gradient estimation
            let concentration_center = self.estimate_concentration(x, y, t, characteristics);
            let concentration_dx = self.estimate_concentration(x + dx, y, t, characteristics);
            let concentration_dy = self.estimate_concentration(x, y + dx, t, characteristics);

            let grad_x = (concentration_dx - concentration_center) / dx;
            let grad_y = (concentration_dy - concentration_center) / dx;
            let gradient_magnitude = (grad_x * grad_x + grad_y * grad_y).sqrt();

            // Add refinement points in high-gradient regions
            if gradient_magnitude > characteristics.diffusion_coefficient * 10.0 {
                // Add points along gradient direction for better resolution
                let grad_dir_x = grad_x / gradient_magnitude.max(1e-6);
                let grad_dir_y = grad_y / gradient_magnitude.max(1e-6);

                for step in 1..=3 {
                    let step_size = step as f64 * dx * 0.5;
                    additional_points.push((
                        x + grad_dir_x * step_size,
                        y + grad_dir_y * step_size,
                        t
                    ));
                }
            }

            // Always add points near boundaries for proper BC enforcement
            if x.abs() < 0.1 || y.abs() < 0.1 {
                additional_points.push((x * 0.95, y * 0.95, t));
                additional_points.push((x * 1.05, y * 1.05, t));
            }
        }

        points.extend(additional_points);
    }

    /// Estimate concentration at a point for adaptive sampling (simplified analytical estimate)
    fn estimate_concentration(&self, x: f64, y: f64, t: f64, characteristics: &PDECharacteristics) -> f64 {
        // Provide analytical estimate of concentration for gradient-based sampling
        // For diffusion equation: ∂c/∂t = D∇²c

        // Assume initial condition c(x,y,0) = exp(-(x² + y²)/σ²) (Gaussian)
        let sigma = 0.5;
        let initial_concentration = (-(x * x + y * y) / (sigma * sigma)).exp();

        // Simple diffusion solution approximation: c(x,y,t) ≈ c0 * (1/(4πDt)) * exp(-r²/(4Dt))
        let diffusion_time = characteristics.diffusion_coefficient * t.max(1e-6);
        let r_squared = x * x + y * y;

        if diffusion_time > 1e-6 {
            initial_concentration / (4.0 * std::f64::consts::PI * diffusion_time).sqrt() *
            (-r_squared / (4.0 * diffusion_time)).exp()
        } else {
            initial_concentration
        }
    }

    /// Adapt sampling for fluid dynamics (concentrate near boundaries and vortices)
    fn adapt_fluid_dynamics_sampling(&self, points: &mut Vec<(f64, f64, f64)>, _characteristics: &PDECharacteristics) {
        // For Navier-Stokes, concentrate near walls and expected vortex locations
        let mut additional_points = Vec::new();

        for &(x, y, t) in points.iter() {
            // Add points near boundaries (walls)
            if x.abs() > 0.8 || y.abs() > 0.8 {
                additional_points.push((x * 0.95, y * 0.95, t));
            }

            // Add points in potential vortex regions
            if x.abs() < 0.5 && y.abs() < 0.5 {
                additional_points.push((x + 0.1, y + 0.1, t));
            }
        }

        points.extend(additional_points);
    }

    /// Default residual-based adaptive sampling
    fn adapt_residual_based_sampling(&self, points: &mut Vec<(f64, f64, f64)>, domain: &dyn PhysicsDomain<B>) {
        // Use simple residual estimation to guide sampling
        // In practice, this would compute actual PDE residuals
        let mut additional_points = Vec::new();

        for &(x, y, t) in points.iter() {
            // Add points in regions likely to have high residuals
            // Simplified: add points near domain boundaries and center
            if (x * x + y * y).sqrt() < 0.3 {
                additional_points.push((x * 1.2, y * 1.2, t));
            }
        }

        points.extend(additional_points);
    }

    /// Get current physics domain
    pub fn current_domain(&self) -> Option<&str> {
        self.current_domain.as_deref()
    }

    /// Get available physics domains
    pub fn available_domains(&self) -> Vec<String> {
        self.physics_registry.list_domains()
    }

    /// Get solver statistics
    pub fn stats(&self) -> &UniversalSolverStats {
        &self.stats
    }
}

impl Default for UniversalSolverStats {
    fn default() -> Self {
        Self {
            training_time: std::time::Duration::default(),
            final_losses: HashMap::new(),
            loss_history: Vec::new(),
            physics_metrics: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_domain_registry() {
        let mut registry = PhysicsDomainRegistry::new();

        // Test empty registry
        assert!(!registry.has_domain("test"));
        assert_eq!(registry.list_domains().len(), 0);

        // Note: We can't easily test domain registration without concrete implementations
        // This would be tested with actual physics domain implementations
    }

    #[test]
    fn test_universal_solver_creation() {
        let solver = UniversalPINNSolver::<burn::backend::NdArray<f32>>::new();
        assert!(solver.is_ok());

        let solver = solver.unwrap();
        assert_eq!(solver.available_domains().len(), 0);
        assert!(solver.current_domain().is_none());
    }

    #[test]
    fn test_physics_loss_weights() {
        let weights = PhysicsLossWeights::default();
        assert_eq!(weights.pde_weight, 1.0);
        assert_eq!(weights.boundary_weight, 10.0);
        assert_eq!(weights.initial_weight, 10.0);
        assert!(weights.physics_weights.is_empty());
    }

    #[test]
    fn test_universal_training_config() {
        let config = UniversalTrainingConfig::default();
        assert_eq!(config.epochs, 1000);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.collocation_points, 1000);
        assert!(config.adaptive_sampling);
    }
}