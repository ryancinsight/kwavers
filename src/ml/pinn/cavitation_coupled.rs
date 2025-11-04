//! Cavitation-Acoustic Coupled Physics Domain for PINN
//!
//! This module implements the coupling between acoustic wave propagation and cavitation
//! bubble dynamics. The acoustic pressure field drives bubble oscillations, while bubble
//! dynamics can modify the acoustic field through scattering and nonlinear effects.
//!
//! ## Mathematical Formulation
//!
//! The coupled system solves:
//! - Acoustic wave equation: ∂²p/∂t² = c²∇²p + nonlinear terms + bubble scattering
//! - Bubble dynamics: Keller-Miksis equation with acoustic forcing
//! - Coupling: p_acoustic drives bubble wall acceleration
//!
//! ## Coupling Types
//!
//! 1. **Weak Coupling**: Acoustic field drives bubble dynamics (one-way)
//! 2. **Strong Coupling**: Mutual interaction with scattering and nonlinear effects
//! 3. **Multi-bubble Coupling**: Collective bubble effects and Bjerknes forces

use crate::error::{KwaversError, KwaversResult};
use crate::ml::pinn::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface,
    CouplingType, InitialConditionSpec, PhysicsDomain, PhysicsLossWeights,
    PhysicsParameters, PhysicsValidationMetric,
};
use crate::physics::bubble_dynamics::{BubbleParameters, BubbleState, KellerMiksisModel};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

/// Cavitation coupling configuration
#[derive(Debug, Clone)]
pub struct CavitationCouplingConfig {
    /// Enable bubble-acoustic coupling
    pub enable_coupling: bool,
    /// Coupling strength (0 = no coupling, 1 = full coupling)
    pub coupling_strength: f64,
    /// Bubble parameters for cavitation
    pub bubble_params: BubbleParameters,
    /// Number of bubbles per coupling point
    pub bubbles_per_point: usize,
    /// Enable multi-bubble interactions
    pub multi_bubble_effects: bool,
    /// Enable nonlinear acoustic effects from bubbles
    pub nonlinear_acoustic: bool,
    /// Domain size for bubble field [Lx, Ly, Lz]
    pub domain_size: Vec<f64>,
}

impl Default for CavitationCouplingConfig {
    fn default() -> Self {
        Self {
            enable_coupling: true,
            coupling_strength: 0.5,
            bubble_params: BubbleParameters::default(),
            bubbles_per_point: 1,
            multi_bubble_effects: false,
            nonlinear_acoustic: true,
            domain_size: vec![1e-2, 1e-2, 1e-2], // 1cm³ domain
        }
    }
}

/// Cavitation-acoustic coupling problem type
#[derive(Debug, Clone, PartialEq)]
pub enum CavitationCouplingType {
    /// Weak coupling: acoustic drives bubbles, no back-coupling
    Weak,
    /// Strong coupling: mutual interaction with scattering
    Strong,
    /// Multi-bubble coupling with collective effects
    MultiBubble,
}

/// Cavitation-coupled physics domain
#[derive(Debug, Clone)]
pub struct CavitationCoupledDomain<B: AutodiffBackend> {
    /// Coupling configuration
    pub config: CavitationCouplingConfig,
    /// Coupling type
    pub coupling_type: CavitationCouplingType,
    /// Bubble dynamics model
    pub bubble_model: KellerMiksisModel,
    /// Bubble states at coupling points
    pub bubble_states: Vec<BubbleState>,
    /// Coupling interfaces
    pub coupling_interfaces: Vec<CouplingInterface>,
    /// Domain dimensions
    pub domain_dims: Vec<f64>,
    /// Backend marker
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> CavitationCoupledDomain<B> {
    /// Create new cavitation-coupled domain
    pub fn new(
        config: CavitationCouplingConfig,
        coupling_type: CavitationCouplingType,
        domain_dims: Vec<f64>,
    ) -> Self {
        let bubble_model = KellerMiksisModel::new(config.bubble_params.clone());

        // Initialize bubble states at coupling points
        let mut bubble_states = Vec::new();
        let n_points = config.bubbles_per_point * 100; // Assume 100 coupling points
        for _ in 0..n_points {
            bubble_states.push(BubbleState::new(&config.bubble_params));
        }

        // Define coupling interfaces
        let coupling_interfaces = Self::create_coupling_interfaces(&config, &coupling_type);

        Self {
            config,
            coupling_type,
            bubble_model,
            bubble_states,
            coupling_interfaces,
            domain_dims,
            _backend: std::marker::PhantomData,
        }
    }

    /// Create coupling interfaces for the domain
    fn create_coupling_interfaces(
        config: &CavitationCouplingConfig,
        coupling_type: &CavitationCouplingType,
    ) -> Vec<CouplingInterface> {
        let mut interfaces = Vec::new();

        // Acoustic-bubble coupling interface
        let acoustic_bubble_coupling = CouplingInterface {
            name: "acoustic_bubble_coupling".to_string(),
            position: BoundaryPosition::CustomRectangular {
                x_min: 0.0,
                x_max: config.domain_size[0],
                y_min: 0.0,
                y_max: config.domain_size[1],
            },
            coupled_domains: vec!["acoustic".to_string(), "cavitation".to_string()],
            coupling_type: match coupling_type {
                CavitationCouplingType::Weak => CouplingType::FluxContinuity,
                CavitationCouplingType::Strong => CouplingType::Conjugate,
                CavitationCouplingType::MultiBubble => CouplingType::Custom("multi_bubble".to_string()),
            },
            coupling_params: {
                let mut params = HashMap::new();
                params.insert("coupling_strength".to_string(), config.coupling_strength);
                params.insert("bubbles_per_point".to_string(), config.bubbles_per_point as f64);
                params.insert("nonlinear_acoustic".to_string(), if config.nonlinear_acoustic { 1.0 } else { 0.0 });
                params
            },
        };

        interfaces.push(acoustic_bubble_coupling);

        // Add multi-bubble interface if enabled
        if config.multi_bubble_effects {
            let multi_bubble_coupling = CouplingInterface {
                name: "multi_bubble_interactions".to_string(),
                position: BoundaryPosition::CustomRectangular {
                    x_min: 0.0,
                    x_max: config.domain_size[0],
                    y_min: 0.0,
                    y_max: config.domain_size[1],
                },
                coupled_domains: vec!["cavitation".to_string()],
                coupling_type: CouplingType::Custom("bjerknes_forces".to_string()),
                coupling_params: {
                    let mut params = HashMap::new();
                    params.insert("enable_bjerknes".to_string(), 1.0);
                    params.insert("collective_effects".to_string(), 1.0);
                    params
                },
            };
            interfaces.push(multi_bubble_coupling);
        }

        interfaces
    }

    /// Compute cavitation PDE residual
    fn cavitation_residual(
        &self,
        acoustic_pressure: &Tensor<B, 2>,
        bubble_positions: &Tensor<B, 2>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Extract bubble parameters from physics params
        let ambient_pressure = physics_params
            .domain_params
            .get("ambient_pressure")
            .copied()
            .unwrap_or(101325.0); // 1 atm

        let viscosity = physics_params
            .domain_params
            .get("liquid_viscosity")
            .copied()
            .unwrap_or(0.001); // Water viscosity

        // For each coupling point, compute bubble dynamics residual
        // This is a simplified implementation - in practice would integrate
        // the Keller-Miksis equation with acoustic forcing

        // Acoustic forcing term: pressure difference drives bubble wall motion
        let pressure_forcing = acoustic_pressure.clone() - ambient_pressure as f32;

        // Bubble dynamics residual (simplified Rayleigh-Plesset form)
        // d²R/dt² + (3/2)(dR/dt)²/R = (1/ρR)(Δp - viscous terms)
        // For PINN: enforce this PDE constraint

        // Simplified residual: acoustic pressure should match bubble wall acceleration
        // In practice, this would be much more complex with full KM equation integration
        pressure_forcing * self.config.coupling_strength as f32
    }

    /// Compute scattering effects from bubbles on acoustic field
    fn bubble_scattering_residual(
        &self,
        acoustic_field: &Tensor<B, 2>,
        bubble_positions: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // Simplified scattering: bubbles act as point scatterers
        // In practice: compute Mie scattering, multiple scattering, etc.

        if !self.config.nonlinear_acoustic {
            return Tensor::zeros_like(acoustic_field);
        }

        // Add nonlinear scattering term (simplified)
        // Real implementation would solve wave equation with bubble-induced sources
        acoustic_field.clone() * 0.1_f32 // Small scattering contribution
    }
}

impl<B: AutodiffBackend> PhysicsDomain<B> for CavitationCoupledDomain<B> {
    fn domain_name(&self) -> &'static str {
        "cavitation_coupled"
    }

    fn pde_residual(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Get acoustic field from model
        let acoustic_field = model.forward(x.clone(), y.clone(), t.clone());

        // Create bubble position tensor (simplified - would be based on actual bubble locations)
        let bubble_positions = Tensor::cat(vec![x.clone(), y.clone()], 1);

        // Compute cavitation coupling residual
        let cavitation_residual = self.cavitation_residual(&acoustic_field, &bubble_positions, physics_params);

        // Add bubble scattering effects to acoustic field
        let scattering_residual = self.bubble_scattering_residual(&acoustic_field, &bubble_positions);

        // Combine residuals
        cavitation_residual + scattering_residual
    }

    fn boundary_conditions(&self) -> Vec<BoundaryConditionSpec> {
        // Return boundary conditions for the coupled system
        // In practice, this would include acoustic boundaries and bubble-related conditions
        vec![
            BoundaryConditionSpec::Dirichlet {
                boundary: BoundaryPosition::Left,
                value: vec![0.0], // Zero pressure at boundary
                component: BoundaryComponent::Scalar,
            },
            BoundaryConditionSpec::Dirichlet {
                boundary: BoundaryPosition::Right,
                value: vec![0.0],
                component: BoundaryComponent::Scalar,
            },
        ]
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        vec![
            // Zero initial pressure
            InitialConditionSpec::DirichletConstant {
                value: vec![0.0],
                component: BoundaryComponent::Scalar,
            },
            // Initial bubble equilibrium radius
            InitialConditionSpec::DirichletConstant {
                value: vec![self.config.bubble_params.r0],
                component: BoundaryComponent::Custom("bubble_radius".to_string()),
            },
        ]
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        let (pde_weight, bc_weight) = match self.coupling_type {
            CavitationCouplingType::Weak => (1.0, 10.0),
            CavitationCouplingType::Strong => (1.0, 5.0),
            CavitationCouplingType::MultiBubble => (1.0, 3.0),
        };

        PhysicsLossWeights {
            pde_weight,
            boundary_weight: bc_weight,
            initial_weight: 10.0,
            physics_weights: {
                let mut weights = HashMap::new();
                weights.insert("cavitation_weight".to_string(), self.config.coupling_strength);
                weights.insert("scattering_weight".to_string(), if self.config.nonlinear_acoustic { 0.5 } else { 0.0 });
                weights
            },
        }
    }

    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric> {
        vec![
            PhysicsValidationMetric {
                name: "cavitation_efficiency".to_string(),
                value: 0.0,
                acceptable_range: (0.0, 1.0),
                description: "Efficiency of acoustic-to-cavitation energy conversion".to_string(),
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

    fn coupling_interfaces(&self) -> Vec<CouplingInterface> {
        self.coupling_interfaces.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cavitation_coupled_domain_creation() {
        let config = CavitationCouplingConfig::default();
        let domain: CavitationCoupledDomain<burn::backend::Autodiff<burn::backend::NdArray<f32>>> = CavitationCoupledDomain::new(
            config,
            CavitationCouplingType::Weak,
            vec![1e-2, 1e-2],
        );

        assert_eq!(domain.domain_name(), "cavitation_coupled");
        assert!(domain.supports_coupling());
        assert!(!domain.coupling_interfaces().is_empty());
    }

    #[test]
    fn test_coupling_interfaces() {
        let config = CavitationCouplingConfig {
            multi_bubble_effects: true,
            ..Default::default()
        };
        let domain: CavitationCoupledDomain<burn::backend::Autodiff<burn::backend::NdArray<f32>>> = CavitationCoupledDomain::new(
            config,
            CavitationCouplingType::MultiBubble,
            vec![1e-2, 1e-2],
        );

        let interfaces = domain.coupling_interfaces();
        assert!(interfaces.len() >= 2); // Should have acoustic-bubble and multi-bubble interfaces

        // Check that we have the expected coupling types
        let has_acoustic_bubble = interfaces.iter().any(|i| i.name == "acoustic_bubble_coupling");
        let has_multi_bubble = interfaces.iter().any(|i| i.name == "multi_bubble_interactions");

        assert!(has_acoustic_bubble);
        assert!(has_multi_bubble);
    }

    #[test]
    fn test_loss_weights_by_coupling_type() {
        let config = CavitationCouplingConfig::default();

        let weak_domain = CavitationCoupledDomain::new(
            config.clone(),
            CavitationCouplingType::Weak,
            vec![1e-2, 1e-2],
        );

        let strong_domain = CavitationCoupledDomain::new(
            config.clone(),
            CavitationCouplingType::Strong,
            vec![1e-2, 1e-2],
        );

        let weak_weights = weak_domain.loss_weights();
        let strong_weights = strong_domain.loss_weights();

        // Strong coupling should have lower boundary weight (more complex coupling)
        assert!(weak_weights.boundary_weight >= strong_weights.boundary_weight);
    }
}
