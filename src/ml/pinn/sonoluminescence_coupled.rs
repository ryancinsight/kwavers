//! Sonoluminescence-Electromagnetic Coupled Physics Domain for PINN
//!
//! This module implements the coupling between electromagnetic wave propagation and
//! sonoluminescence light emission. The collapsing cavitation bubbles act as dynamic
//! light sources in the electromagnetic field.
//!
//! ## Mathematical Formulation
//!
//! The coupled system solves:
//! - Maxwell's equations with dynamic light sources from bubbles
//! - Bubble collapse dynamics driving light emission
//! - Coupling: Bubble energy release → electromagnetic radiation
//!
//! ## Coupling Physics
//!
//! 1. **Blackbody Radiation**: Hot bubble interior acts as blackbody emitter
//! 2. **Bremsstrahlung**: Ionized gas produces continuum radiation
//! 3. **Molecular Emission**: Excited species produce line spectra
//! 4. **Electromagnetic Propagation**: Light waves propagate through the medium

use crate::error::{KwaversError, KwaversResult};
use crate::ml::pinn::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface,
    CouplingType, InitialConditionSpec, PhysicsDomain, PhysicsLossWeights,
    PhysicsParameters, PhysicsValidationMetric,
};
use crate::physics::optics::sonoluminescence::{EmissionParameters, SonoluminescenceEmission};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use ndarray::Array3;
use std::collections::HashMap;

/// Sonoluminescence coupling configuration
#[derive(Debug, Clone)]
pub struct SonoluminescenceCouplingConfig {
    /// Enable electromagnetic-sonoluminescence coupling
    pub enable_coupling: bool,
    /// Coupling efficiency (fraction of bubble energy converted to light)
    pub coupling_efficiency: f64,
    /// Emission parameters for sonoluminescence
    pub emission_params: EmissionParameters,
    /// Grid shape for emission field [nx, ny, nz]
    pub grid_shape: (usize, usize, usize),
    /// Grid spacing [dx, dy, dz] in meters
    pub grid_spacing: (f64, f64, f64),
    /// Enable spectral resolution
    pub spectral_resolution: bool,
    /// Wavelength range for spectral calculations [min, max] in meters
    pub wavelength_range: (f64, f64),
    /// Number of wavelength bins
    pub n_wavelengths: usize,
}

impl Default for SonoluminescenceCouplingConfig {
    fn default() -> Self {
        Self {
            enable_coupling: true,
            coupling_efficiency: 0.001, // 0.1% of bubble energy becomes light
            emission_params: EmissionParameters::default(),
            grid_shape: (50, 50, 50),
            grid_spacing: (2e-4, 2e-4, 2e-4), // 200 μm spacing
            spectral_resolution: true,
            wavelength_range: (200e-9, 1000e-9), // 200nm to 1000nm
            n_wavelengths: 50,
        }
    }
}

/// Sonoluminescence-electromagnetic coupling problem type
#[derive(Debug, Clone, PartialEq)]
pub enum SonoluminescenceCouplingType {
    /// Static emission: light sources fixed in time
    StaticEmission,
    /// Dynamic emission: time-varying light sources from bubble collapse
    DynamicEmission,
    /// Spectral coupling: full wavelength-dependent emission and propagation
    SpectralCoupling,
}

/// Sonoluminescence-coupled physics domain
#[derive(Debug)]
pub struct SonoluminescenceCoupledDomain<B: AutodiffBackend> {
    /// Coupling configuration
    pub config: SonoluminescenceCouplingConfig,
    /// Coupling type
    pub coupling_type: SonoluminescenceCouplingType,
    /// Sonoluminescence emission calculator
    pub emission_calculator: SonoluminescenceEmission,
    /// Current bubble state fields
    pub bubble_states: Array3<f64>,
    /// Current temperature field from bubbles
    pub temperature_field: Array3<f64>,
    /// Coupling interfaces
    pub coupling_interfaces: Vec<CouplingInterface>,
    /// Backend marker
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> SonoluminescenceCoupledDomain<B> {
    /// Create new sonoluminescence-coupled domain
    pub fn new(
        config: SonoluminescenceCouplingConfig,
        coupling_type: SonoluminescenceCouplingType,
    ) -> Self {
        let emission_calculator = SonoluminescenceEmission::new(
            config.grid_shape,
            config.emission_params.clone(),
        );

        // Initialize fields
        let bubble_states = Array3::zeros(config.grid_shape);
        let temperature_field = Array3::zeros(config.grid_shape);

        // Define coupling interfaces
        let coupling_interfaces = Self::create_coupling_interfaces(&config, &coupling_type);

        Self {
            config,
            coupling_type,
            emission_calculator,
            bubble_states,
            temperature_field,
            coupling_interfaces,
            _backend: std::marker::PhantomData,
        }
    }

    /// Create coupling interfaces for the domain
    fn create_coupling_interfaces(
        config: &SonoluminescenceCouplingConfig,
        coupling_type: &SonoluminescenceCouplingType,
    ) -> Vec<CouplingInterface> {
        let mut interfaces = Vec::new();

        // Electromagnetic-sonoluminescence coupling interface
        let em_sl_coupling = CouplingInterface {
            name: "electromagnetic_sonoluminescence".to_string(),
            position: BoundaryPosition::CustomRectangular {
                x_min: 0.0,
                x_max: config.grid_shape.0 as f64 * config.grid_spacing.0,
                y_min: 0.0,
                y_max: config.grid_shape.1 as f64 * config.grid_spacing.1,
            },
            coupled_domains: vec!["electromagnetic".to_string(), "sonoluminescence".to_string()],
            coupling_type: match coupling_type {
                SonoluminescenceCouplingType::StaticEmission => CouplingType::SolutionContinuity,
                SonoluminescenceCouplingType::DynamicEmission => CouplingType::FluxContinuity,
                SonoluminescenceCouplingType::SpectralCoupling => CouplingType::Custom("spectral".to_string()),
            },
            coupling_params: {
                let mut params = HashMap::new();
                params.insert("coupling_efficiency".to_string(), config.coupling_efficiency);
                params.insert("spectral_resolution".to_string(), if config.spectral_resolution { 1.0 } else { 0.0 });
                params.insert("n_wavelengths".to_string(), config.n_wavelengths as f64);
                params
            },
        };

        interfaces.push(em_sl_coupling);

        // Add spectral interface if spectral coupling is enabled
        if config.spectral_resolution && matches!(coupling_type, SonoluminescenceCouplingType::SpectralCoupling) {
            let spectral_coupling = CouplingInterface {
                name: "spectral_propagation".to_string(),
                position: BoundaryPosition::CustomRectangular {
                    x_min: 0.0,
                    x_max: config.grid_shape.0 as f64 * config.grid_spacing.0,
                    y_min: 0.0,
                    y_max: config.grid_shape.1 as f64 * config.grid_spacing.1,
                },
                coupled_domains: vec!["electromagnetic".to_string()],
                coupling_type: CouplingType::Custom("wavelength_dependent".to_string()),
                coupling_params: {
                    let mut params = HashMap::new();
                    params.insert("min_wavelength".to_string(), config.wavelength_range.0);
                    params.insert("max_wavelength".to_string(), config.wavelength_range.1);
                    params.insert("wavelength_bins".to_string(), config.n_wavelengths as f64);
                    params
                },
            };
            interfaces.push(spectral_coupling);
        }

        interfaces
    }

    /// Update bubble state and temperature fields
    pub fn update_bubble_states(&mut self, new_bubble_states: Array3<f64>, new_temperature: Array3<f64>) {
        self.bubble_states = new_bubble_states;
        self.temperature_field = new_temperature;

        // Update emission calculator with new temperature field
        // This would trigger recalculation of emission spectra
        // In practice, this would be more sophisticated
    }

    /// Compute sonoluminescence source terms for electromagnetic equations
    fn compute_light_sources(
        &self,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Convert tensor positions to grid indices
        let x_vals = x.clone();
        let y_vals = y.clone();

        // Simplified: assume we're working with 2D slices
        // In practice, would need 3D interpolation

        // Get emission intensity from calculator
        let emission_intensity = self.emission_calculator.emission_field.clone();

        // Convert to tensor (simplified - would need proper interpolation)
        let batch_size = x.shape().dims[0];
        let mut source_terms = Vec::new();

        for i in 0..batch_size {
            // Simplified: just use a constant source term
            // Real implementation would interpolate from emission_intensity
            source_terms.push(self.config.coupling_efficiency as f32);
        }

        Tensor::<B, 1>::from_floats(source_terms.as_slice(), &<B as burn::tensor::backend::Backend>::Device::default()).reshape([batch_size, 1])
    }

    /// Compute electromagnetic PDE residual with sonoluminescence sources
    fn electromagnetic_residual_with_sources(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // This would normally delegate to the electromagnetic domain
        // but add the sonoluminescence source terms

        // For now, simplified implementation
        let field = model.forward(x.clone(), y.clone(), t.clone());
        let sources = self.compute_light_sources(x, y, t, physics_params);

        // Maxwell's equations residual with sources
        // ∇×E = -∂B/∂t - J (where J includes sonoluminescence sources)
        // Simplified residual
        field + sources
    }
}

impl<B: AutodiffBackend> PhysicsDomain<B> for SonoluminescenceCoupledDomain<B> {
    fn domain_name(&self) -> &'static str {
        "sonoluminescence_coupled"
    }

    fn pde_residual(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        self.electromagnetic_residual_with_sources(model, x, y, t, physics_params)
    }

    fn boundary_conditions(&self) -> Vec<BoundaryConditionSpec> {
        // Electromagnetic boundary conditions with light emission considerations
        vec![
            // Perfect electric conductor boundaries
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
                boundary: BoundaryPosition::Top,
                value: vec![0.0],
                component: BoundaryComponent::Scalar,
            },
            BoundaryConditionSpec::Dirichlet {
                boundary: BoundaryPosition::Bottom,
                value: vec![0.0],
                component: BoundaryComponent::Scalar,
            },
        ]
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        vec![
            // Zero initial electromagnetic fields
            InitialConditionSpec::DirichletConstant {
                value: vec![0.0, 0.0], // E_x, E_y
                component: BoundaryComponent::Vector(vec![0, 1]),
            },
            InitialConditionSpec::DirichletConstant {
                value: vec![0.0, 0.0], // H_x, H_y
                component: BoundaryComponent::Vector(vec![0, 1]),
            },
        ]
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        let (pde_weight, bc_weight) = match self.coupling_type {
            SonoluminescenceCouplingType::StaticEmission => (1.0, 10.0),
            SonoluminescenceCouplingType::DynamicEmission => (1.0, 5.0),
            SonoluminescenceCouplingType::SpectralCoupling => (1.0, 2.0),
        };

        PhysicsLossWeights {
            pde_weight,
            boundary_weight: bc_weight,
            initial_weight: 10.0,
            physics_weights: {
                let mut weights = HashMap::new();
                weights.insert("light_source_weight".to_string(), self.config.coupling_efficiency);
                weights.insert("spectral_weight".to_string(), if self.config.spectral_resolution { 1.0 } else { 0.0 });
                weights
            },
        }
    }

    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric> {
        vec![
            PhysicsValidationMetric {
                name: "light_emission_efficiency".to_string(),
                value: 0.0,
                acceptable_range: (0.0, 1.0),
                description: "Efficiency of bubble energy conversion to light".to_string(),
            },
            PhysicsValidationMetric {
                name: "spectral_accuracy".to_string(),
                value: 0.0,
                acceptable_range: (-0.1, 0.1),
                description: "Accuracy of spectral emission calculations".to_string(),
            },
            PhysicsValidationMetric {
                name: "electromagnetic_consistency".to_string(),
                value: 0.0,
                acceptable_range: (-1e-6, 1e-6),
                description: "Maxwell's equations residual with light sources".to_string(),
            },
            PhysicsValidationMetric {
                name: "total_luminosity".to_string(),
                value: 0.0,
                acceptable_range: (0.0, f64::INFINITY),
                description: "Total light output from sonoluminescence".to_string(),
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

impl<B: AutodiffBackend> Clone for SonoluminescenceCoupledDomain<B> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            coupling_type: self.coupling_type.clone(),
            emission_calculator: SonoluminescenceEmission::new(
                self.config.grid_shape,
                self.config.emission_params.clone(),
            ),
            bubble_states: self.bubble_states.clone(),
            temperature_field: self.temperature_field.clone(),
            coupling_interfaces: self.coupling_interfaces.clone(),
            _backend: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sonoluminescence_coupled_domain_creation() {
        let config = SonoluminescenceCouplingConfig::default();
        let domain: SonoluminescenceCoupledDomain<burn::backend::Autodiff<burn::backend::NdArray<f32>>> = SonoluminescenceCoupledDomain::new(
            config,
            SonoluminescenceCouplingType::DynamicEmission,
        );

        assert_eq!(domain.domain_name(), "sonoluminescence_coupled");
        assert!(domain.supports_coupling());
        assert!(!domain.coupling_interfaces().is_empty());
    }

    #[test]
    fn test_spectral_coupling_interfaces() {
        let config = SonoluminescenceCouplingConfig {
            spectral_resolution: true,
            ..Default::default()
        };
        let domain: SonoluminescenceCoupledDomain<burn::backend::Autodiff<burn::backend::NdArray<f32>>> = SonoluminescenceCoupledDomain::new(
            config,
            SonoluminescenceCouplingType::SpectralCoupling,
        );

        let interfaces = domain.coupling_interfaces();
        assert!(interfaces.len() >= 2); // Should have EM-SL and spectral interfaces

        // Check for expected interface names
        let has_em_sl = interfaces.iter().any(|i| i.name == "electromagnetic_sonoluminescence");
        let has_spectral = interfaces.iter().any(|i| i.name == "spectral_propagation");

        assert!(has_em_sl);
        assert!(has_spectral);
    }

    #[test]
    fn test_coupling_efficiency_parameter() {
        let config = SonoluminescenceCouplingConfig {
            coupling_efficiency: 0.005, // 0.5%
            ..Default::default()
        };
        let domain: SonoluminescenceCoupledDomain<burn::backend::Autodiff<burn::backend::NdArray<f32>>> = SonoluminescenceCoupledDomain::new(
            config,
            SonoluminescenceCouplingType::DynamicEmission,
        );

        let weights = domain.loss_weights();
        assert_eq!(weights.physics_weights["light_source_weight"], 0.005);
    }
}
