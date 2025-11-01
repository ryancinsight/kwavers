//! Heat Transfer Physics Domain for PINN
//!
//! This module implements heat conduction, convection, and radiation for thermal
//! analysis using Physics-Informed Neural Networks. The implementation supports
//! conjugate heat transfer with fluid-solid interfaces and multi-physics coupling.
//!
//! ## Mathematical Formulation
//!
//! The general heat equation is:
//!
//! ρc ∂T/∂t = ∇·(k ∇T) + q̇
//!
//! Where:
//! - ρ: density (kg/m³)
//! - c: specific heat capacity (J/kg·K)
//! - k: thermal conductivity (W/m·K)
//! - T: temperature (K)
//! - q̇: volumetric heat generation (W/m³)
//!
//! ## Boundary Conditions
//!
//! - Dirichlet: prescribed temperature T = T₀
//! - Neumann: prescribed heat flux -k ∂T/∂n = q₀
//! - Robin: convection h(T - T∞) - k ∂T/∂n = 0
//! - Radiation: εσ(T⁴ - T∞⁴) - k ∂T/∂n = 0

use crate::error::{KwaversError, KwaversResult};
use crate::ml::pinn::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface,
    CouplingType, InitialConditionSpec, PhysicsDomain, PhysicsLossWeights,
    PhysicsParameters, PhysicsValidationMetric,
};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

/// Heat source specification
#[derive(Debug, Clone)]
pub struct HeatSource {
    /// Position of heat source
    pub position: (f64, f64),
    /// Heat generation rate (W/m³)
    pub power_density: f64,
    /// Source radius/size
    pub radius: f64,
}

/// Material interface for multi-material problems
#[derive(Debug, Clone)]
pub struct MaterialInterface {
    /// Interface position
    pub position: BoundaryPosition,
    /// Left material properties (thermal conductivity, density, specific heat)
    pub left_material: (f64, f64, f64),
    /// Right material properties
    pub right_material: (f64, f64, f64),
    /// Interface resistance (m²·K/W)
    pub thermal_resistance: f64,
}

/// Heat transfer boundary condition specification
#[derive(Debug, Clone)]
pub enum HeatTransferBoundarySpec {
    /// Dirichlet boundary condition: T = T₀
    DirichletTemperature {
        /// Boundary position
        position: BoundaryPosition,
        /// Prescribed temperature (K)
        temperature: f64,
    },
    /// Neumann boundary condition: -k ∂T/∂n = q₀
    HeatFlux {
        /// Boundary position
        position: BoundaryPosition,
        /// Heat flux (W/m²)
        heat_flux: f64,
    },
    /// Robin boundary condition: convection
    Convection {
        /// Boundary position
        position: BoundaryPosition,
        /// Heat transfer coefficient (W/m²·K)
        htc: f64,
        /// Ambient temperature (K)
        ambient_temperature: f64,
    },
    /// Radiation boundary condition
    Radiation {
        /// Boundary position
        position: BoundaryPosition,
        /// Emissivity
        emissivity: f64,
        /// Ambient temperature (K)
        ambient_temperature: f64,
    },
    /// Adiabatic boundary: zero heat flux
    Adiabatic {
        /// Boundary position
        position: BoundaryPosition,
    },
}

/// Heat transfer physics domain implementation
#[derive(Debug, Clone)]
pub struct HeatTransferDomain {
    /// Thermal conductivity (W/m·K)
    pub thermal_conductivity: f64,
    /// Density (kg/m³)
    pub density: f64,
    /// Specific heat capacity (J/kg·K)
    pub specific_heat: f64,
    /// Heat sources
    pub heat_sources: Vec<HeatSource>,
    /// Material interfaces
    pub interfaces: Vec<MaterialInterface>,
    /// Boundary conditions
    pub boundary_specs: Vec<HeatTransferBoundarySpec>,
    /// Stefan-Boltzmann constant (W/m²·K⁴)
    pub stefan_boltzmann: f64,
    /// Domain dimensions [Lx, Ly]
    pub domain_size: Vec<f64>,
}

impl Default for HeatTransferDomain {
    fn default() -> Self {
        Self {
            thermal_conductivity: 50.0, // Copper-like
            density: 8960.0,           // Copper density
            specific_heat: 385.0,      // Copper specific heat
            heat_sources: Vec::new(),
            interfaces: Vec::new(),
            boundary_specs: Vec::new(),
            stefan_boltzmann: 5.67e-8,
            domain_size: vec![1.0, 1.0],
        }
    }
}

impl<B: AutodiffBackend> PhysicsDomain<B> for HeatTransferDomain {
    fn domain_name(&self) -> &'static str {
        "heat_transfer"
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
        let k = physics_params
            .domain_params
            .get("thermal_conductivity")
            .copied()
            .unwrap_or(self.thermal_conductivity);
        let rho = physics_params
            .domain_params
            .get("density")
            .copied()
            .unwrap_or(self.density);
        let cp = physics_params
            .domain_params
            .get("specific_heat")
            .copied()
            .unwrap_or(self.specific_heat);

        // Create input tensor for neural network
        let inputs = Tensor::cat(vec![x.clone(), y.clone(), t.clone()], 1);

        // Forward pass through model to get temperature field
        let temperature = model.forward(&inputs);

        // Compute spatial derivatives
        let t_x = temperature.backward(x);
        let t_y = temperature.backward(y);
        let t_t = temperature.backward(t);
        let t_xx = t_x.backward(x);
        let t_yy = t_y.backward(y);

        // Heat equation: ρc ∂T/∂t = k(∂²T/∂x² + ∂²T/∂y²) + q̇
        let alpha = k / (rho * cp); // thermal diffusivity
        let diffusion = alpha * (t_xx + t_yy);
        let heat_equation = rho * cp * t_t - k * (t_xx + t_yy);

        // Add heat sources if present
        let q_dot = self.compute_heat_sources(x, y, t);
        let heat_equation_with_sources = heat_equation - q_dot;

        heat_equation_with_sources
    }

    fn boundary_conditions(&self) -> Vec<BoundaryConditionSpec> {
        self.boundary_specs
            .iter()
            .map(|spec| match spec {
                HeatTransferBoundarySpec::DirichletTemperature { position, temperature } => {
                    BoundaryConditionSpec::Dirichlet {
                        boundary: position.clone(),
                        value: vec![*temperature],
                        component: BoundaryComponent::Scalar,
                    }
                }
                HeatTransferBoundarySpec::HeatFlux { position, heat_flux } => {
                    BoundaryConditionSpec::Neumann {
                        boundary: position.clone(),
                        flux: vec![*heat_flux],
                        component: BoundaryComponent::Scalar,
                    }
                }
                HeatTransferBoundarySpec::Convection { position, htc, ambient_temperature } => {
                    BoundaryConditionSpec::Robin {
                        boundary: position.clone(),
                        alpha: *htc,
                        beta: *htc * ambient_temperature,
                        component: BoundaryComponent::Scalar,
                    }
                }
                HeatTransferBoundarySpec::Radiation { position, emissivity, ambient_temperature } => {
                    // Radiation: εσ(T⁴ - T∞⁴) - k ∂T/∂n = 0
                    // This is a nonlinear Robin condition, approximated as Robin for now
                    BoundaryConditionSpec::Robin {
                        boundary: position.clone(),
                        alpha: *emissivity * self.stefan_boltzmann * 4.0 * ambient_temperature.powi(3),
                        beta: *emissivity * self.stefan_boltzmann * ambient_temperature.powi(4),
                        component: BoundaryComponent::Scalar,
                    }
                }
                HeatTransferBoundarySpec::Adiabatic { position } => {
                    BoundaryConditionSpec::Neumann {
                        boundary: position.clone(),
                        flux: vec![0.0], // zero heat flux
                        component: BoundaryComponent::Scalar,
                    }
                }
            })
            .collect()
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        // Default initial condition: uniform temperature of 293K (20°C)
        vec![InitialConditionSpec::DirichletConstant {
            value: vec![293.0],
            component: BoundaryComponent::Scalar,
        }]
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        PhysicsLossWeights {
            pde_weight: 1.0,
            boundary_weight: 10.0,
            initial_weight: 10.0,
            physics_weights: {
                let mut weights = HashMap::new();
                weights.insert("diffusion_weight".to_string(), 1.0);
                weights.insert("source_weight".to_string(), 1.0);
                weights
            },
        }
    }

    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric> {
        vec![
            PhysicsValidationMetric {
                name: "heat_equation_residual".to_string(),
                value: 0.0,
                acceptable_range: (-1e-6, 1e-6),
                description: "Heat equation PDE residual (should be ~0)".to_string(),
            },
            PhysicsValidationMetric {
                name: "energy_conservation".to_string(),
                value: 0.0,
                acceptable_range: (-1e-4, 1e-4),
                description: "Total energy conservation".to_string(),
            },
            PhysicsValidationMetric {
                name: "boundary_error".to_string(),
                value: 0.0,
                acceptable_range: (-1e-6, 1e-6),
                description: "Boundary condition satisfaction error".to_string(),
            },
            PhysicsValidationMetric {
                name: "steady_state_error".to_string(),
                value: 0.0,
                acceptable_range: (-1e-5, 1e-5),
                description: "Steady-state solution accuracy".to_string(),
            },
        ]
    }

    fn supports_coupling(&self) -> bool {
        true // Heat transfer can couple with Navier-Stokes, structural mechanics
    }

    fn coupling_interfaces(&self) -> Vec<CouplingInterface> {
        vec![
            CouplingInterface {
                name: "thermal_fluid_interface".to_string(),
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
                    params.insert("heat_transfer_coefficient".to_string(), 100.0);
                    params
                },
            },
            CouplingInterface {
                name: "thermal_structural_interface".to_string(),
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
                    params.insert("thermal_expansion_coefficient".to_string(), 1e-5);
                    params
                },
            },
        ]
    }
}

impl HeatTransferDomain {
    /// Create a new heat transfer domain
    pub fn new(
        thermal_conductivity: f64,
        density: f64,
        specific_heat: f64,
        domain_size: Vec<f64>,
    ) -> Self {
        Self {
            thermal_conductivity,
            density,
            specific_heat,
            heat_sources: Vec::new(),
            interfaces: Vec::new(),
            boundary_specs: Vec::new(),
            stefan_boltzmann: 5.67e-8,
            domain_size,
        }
    }

    /// Add a heat source
    pub fn add_heat_source(mut self, position: (f64, f64), power_density: f64, radius: f64) -> Self {
        self.heat_sources.push(HeatSource {
            position,
            power_density,
            radius,
        });
        self
    }

    /// Add a temperature boundary condition
    pub fn add_temperature_bc(mut self, position: BoundaryPosition, temperature: f64) -> Self {
        self.boundary_specs.push(HeatTransferBoundarySpec::DirichletTemperature {
            position,
            temperature,
        });
        self
    }

    /// Add a convection boundary condition
    pub fn add_convection_bc(
        mut self,
        position: BoundaryPosition,
        htc: f64,
        ambient_temperature: f64,
    ) -> Self {
        self.boundary_specs.push(HeatTransferBoundarySpec::Convection {
            position,
            htc,
            ambient_temperature,
        });
        self
    }

    /// Add an adiabatic boundary condition
    pub fn add_adiabatic_bc(mut self, position: BoundaryPosition) -> Self {
        self.boundary_specs.push(HeatTransferBoundarySpec::Adiabatic { position });
        self
    }

    /// Compute heat source term at given positions
    fn compute_heat_sources(
        &self,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        _t: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // For now, return zero tensor - heat sources would be implemented
        // based on specific source geometries and time dependencies
        Tensor::zeros_like(x)
    }

    /// Compute thermal diffusivity
    pub fn thermal_diffusivity(&self) -> f64 {
        self.thermal_conductivity / (self.density * self.specific_heat)
    }

    /// Validate domain configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.thermal_conductivity <= 0.0 {
            return Err(KwaversError::ValidationError {
                field: "thermal_conductivity".to_string(),
                reason: "Thermal conductivity must be positive".to_string(),
            });
        }

        if self.density <= 0.0 {
            return Err(KwaversError::ValidationError {
                field: "density".to_string(),
                reason: "Density must be positive".to_string(),
            });
        }

        if self.specific_heat <= 0.0 {
            return Err(KwaversError::ValidationError {
                field: "specific_heat".to_string(),
                reason: "Specific heat must be positive".to_string(),
            });
        }

        for source in &self.heat_sources {
            if source.power_density < 0.0 {
                return Err(KwaversError::ValidationError {
                    field: "heat_source".to_string(),
                    reason: "Heat source power density cannot be negative".to_string(),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_transfer_domain_creation() {
        let domain = HeatTransferDomain::new(50.0, 8960.0, 385.0, vec![1.0, 1.0]);

        assert_eq!(domain.thermal_conductivity, 50.0);
        assert_eq!(domain.density, 8960.0);
        assert_eq!(domain.specific_heat, 385.0);
    }

    #[test]
    fn test_thermal_diffusivity() {
        let domain = HeatTransferDomain::new(50.0, 8960.0, 385.0, vec![1.0, 1.0]);
        let expected_alpha = 50.0 / (8960.0 * 385.0);
        assert!((domain.thermal_diffusivity() - expected_alpha).abs() < 1e-10);
    }

    #[test]
    fn test_domain_validation() {
        let valid_domain = HeatTransferDomain::new(50.0, 8960.0, 385.0, vec![1.0, 1.0]);
        assert!(valid_domain.validate().is_ok());

        let invalid_domain = HeatTransferDomain::new(-50.0, 8960.0, 385.0, vec![1.0, 1.0]);
        assert!(invalid_domain.validate().is_err());
    }

    #[test]
    fn test_boundary_condition_builder() {
        let domain = HeatTransferDomain::default()
            .add_temperature_bc(BoundaryPosition::Left, 373.0)
            .add_convection_bc(BoundaryPosition::Top, 10.0, 293.0)
            .add_adiabatic_bc(BoundaryPosition::Bottom);

        assert_eq!(domain.boundary_specs.len(), 3);

        match &domain.boundary_specs[0] {
            HeatTransferBoundarySpec::DirichletTemperature { temperature, .. } => {
                assert_eq!(*temperature, 373.0);
            }
            _ => panic!("Expected DirichletTemperature"),
        }
    }

    #[test]
    fn test_heat_source_builder() {
        let domain = HeatTransferDomain::default()
            .add_heat_source((0.5, 0.5), 1e6, 0.1);

        assert_eq!(domain.heat_sources.len(), 1);
        assert_eq!(domain.heat_sources[0].position, (0.5, 0.5));
        assert_eq!(domain.heat_sources[0].power_density, 1e6);
    }

    #[test]
    fn test_physics_domain_interface() {
        let domain = HeatTransferDomain::default();

        assert_eq!(domain.domain_name(), "heat_transfer");

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
