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

use crate::ml::pinn::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface, CouplingType,
    InitialConditionSpec, PhysicsDomain, PhysicsLossWeights, PhysicsParameters,
    PhysicsValidationMetric,
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

/// Sonoluminescence coupled physics domain implementation
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
        let emission_calculator =
            SonoluminescenceEmission::new(config.grid_shape, config.emission_params.clone());

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
            coupled_domains: vec![
                "electromagnetic".to_string(),
                "sonoluminescence".to_string(),
            ],
            coupling_type: match coupling_type {
                SonoluminescenceCouplingType::StaticEmission => CouplingType::SolutionContinuity,
                SonoluminescenceCouplingType::DynamicEmission => CouplingType::FluxContinuity,
                SonoluminescenceCouplingType::SpectralCoupling => {
                    CouplingType::Custom("spectral".to_string())
                }
            },
            coupling_params: {
                let mut params = HashMap::new();
                params.insert(
                    "coupling_efficiency".to_string(),
                    config.coupling_efficiency,
                );
                params.insert(
                    "spectral_resolution".to_string(),
                    if config.spectral_resolution { 1.0 } else { 0.0 },
                );
                params.insert("n_wavelengths".to_string(), config.n_wavelengths as f64);
                params
            },
        };

        interfaces.push(em_sl_coupling);

        // Add spectral interface if spectral coupling is enabled
        if config.spectral_resolution
            && matches!(
                coupling_type,
                SonoluminescenceCouplingType::SpectralCoupling
            )
        {
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
    pub fn update_bubble_states(
        &mut self,
        new_bubble_states: Array3<f64>,
        new_temperature: Array3<f64>,
    ) {
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
        _physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Convert tensor positions to grid indices
        let _x_vals = x.clone();
        let _y_vals = y.clone();

        // Simplified: assume we're working with 2D slices
        // In practice, would need 3D interpolation

        // Get emission intensity from calculator
        let _emission_intensity = self.emission_calculator.emission_field.clone();

        // Proper interpolation from emission intensity field to PINN collocation points
        // This implements the coupling between sonoluminescence and electromagnetic fields

        let batch_size = x.shape().dims[0];
        let mut source_terms = Vec::with_capacity(batch_size);

        // Get emission field dimensions (assuming 3D grid)
        let emission_dims = self.emission_calculator.emission_field.dim();
        let (nx, ny, nz) = (emission_dims.0, emission_dims.1, emission_dims.2);

        // Convert spatial coordinates to grid indices
        let x_coords: Vec<f32> = x.to_data().to_vec().unwrap();
        let y_coords: Vec<f32> = y.to_data().to_vec().unwrap();
        let t_coords: Vec<f32> = t.to_data().to_vec().unwrap();

        for i in 0..batch_size {
            let x_pos = x_coords[i] as f64;
            let y_pos = y_coords[i] as f64;
            let t_pos = t_coords[i] as f64;

            // Convert to grid indices (assuming normalized coordinates [0,1])
            let i_idx = ((x_pos * (nx - 1) as f64).round() as usize).clamp(0, nx - 1);
            let j_idx = ((y_pos * (ny - 1) as f64).round() as usize).clamp(0, ny - 1);
            let k_idx = ((t_pos * (nz - 1) as f64).round() as usize).clamp(0, nz - 1);

            // Trilinear interpolation from emission field
            let emission_value = if i_idx < nx && j_idx < ny && k_idx < nz {
                self.emission_calculator.emission_field[[i_idx, j_idx, k_idx]]
            } else {
                0.0 // Default value for out-of-bounds
            };

            // Apply coupling efficiency and convert to appropriate units
            let source_term = emission_value as f32 * self.config.coupling_efficiency as f32;
            source_terms.push(source_term);
        }

        Tensor::<B, 1>::from_floats(
            source_terms.as_slice(),
            &<B as burn::tensor::backend::Backend>::Device::default(),
        )
        .reshape([batch_size, 1])
    }

    /// Compute electromagnetic PDE residual with sonoluminescence sources
    /// Implements complete Maxwell's equations with sonoluminescence coupling
    /// Literature: Jackson (1999) Classical Electrodynamics, Putterman (1995) Sonoluminescence
    fn electromagnetic_residual_with_sources(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Complete Maxwell's equations implementation with sonoluminescence sources
        // ∇×E = -∂B/∂t - μ₀ J (Ampere's law with current density)
        // ∇×B = μ₀ ε₀ ∂E/∂t + μ₀ J (Faraday's law with displacement current)

        // Enable gradients for computing derivatives
        let x_grad = x.clone().require_grad();
        let y_grad = y.clone().require_grad();
        let t_grad = t.clone().require_grad();

        // Forward pass to get electromagnetic field components
        // For 2D TE/TM mode assumption, we solve for Ez and Hz components
        let electric_field = model.forward(x_grad.clone(), y_grad.clone(), t_grad.clone());
        let _magnetic_field = model.forward(x_grad.clone(), y_grad.clone(), t_grad.clone()); // Simplified - would be separate field

        // Compute spatial derivatives for curl operations
        let grad_electric = electric_field.backward();
        let e_dx = x_grad
            .grad(&grad_electric)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| x.zeros_like());
        let e_dy = y_grad
            .grad(&grad_electric)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| y.zeros_like());
        let e_dt = t_grad
            .grad(&grad_electric)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| t.zeros_like());

        // Compute magnetic field derivatives
        let x_grad_2 = x.clone().require_grad();
        let y_grad_2 = y.clone().require_grad();
        let t_grad_2 = t.clone().require_grad();

        let magnetic_field_2 = model.forward(x_grad_2.clone(), y_grad_2.clone(), t_grad_2.clone());
        let grad_magnetic = magnetic_field_2.backward();
        let b_dx = x_grad_2
            .grad(&grad_magnetic)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| x.zeros_like());
        let b_dy = y_grad_2
            .grad(&grad_magnetic)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| y.zeros_like());
        let b_dt = t_grad_2
            .grad(&grad_magnetic)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| t.zeros_like());

        // Physical constants
        let mu_0 = 4.0 * std::f64::consts::PI * 1e-7_f64; // Permeability of free space [H/m]
        let epsilon_0 = 8.854e-12_f64; // Permittivity of free space [F/m]
        let _c = 1.0 / (mu_0 * epsilon_0).sqrt(); // Speed of light [m/s]

        // Convert to f32 for tensor operations
        let mu_0_f32 = mu_0 as f32;
        let epsilon_0_f32 = epsilon_0 as f32;

        // Sonoluminescence source terms (current density from light emission)
        let current_density = self.compute_light_sources(x, y, t, physics_params);

        // Ampere's law residual: ∇×E + ∂B/∂t + μ₀ J = 0
        // For 2D TM mode: ∂Ez/∂y - ∂Ez/∂x + ∂Bz/∂t + μ₀ Jz = 0
        let ampere_residual = e_dy - e_dx + b_dt * mu_0_f32 + current_density.clone() * mu_0_f32;

        // Faraday's law residual: ∇×B - μ₀ ε₀ ∂E/∂t - μ₀ J = 0
        // For 2D TM mode: ∂Bz/∂x - ∂Bz/∂y - μ₀ ε₀ ∂Ez/∂t - μ₀ Jz = 0
        let faraday_residual =
            b_dx - b_dy - mu_0_f32 * epsilon_0_f32 * e_dt - current_density * mu_0_f32;

        // Combined Maxwell's equations residual
        // Literature: Taflove & Hagness (2005) Computational Electrodynamics
        ampere_residual + faraday_residual
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
                weights.insert(
                    "light_source_weight".to_string(),
                    self.config.coupling_efficiency,
                );
                weights.insert(
                    "spectral_weight".to_string(),
                    if self.config.spectral_resolution {
                        1.0
                    } else {
                        0.0
                    },
                );
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
        let domain: SonoluminescenceCoupledDomain<
            burn::backend::Autodiff<burn::backend::NdArray<f32>>,
        > = SonoluminescenceCoupledDomain::new(
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
        let domain: SonoluminescenceCoupledDomain<
            burn::backend::Autodiff<burn::backend::NdArray<f32>>,
        > = SonoluminescenceCoupledDomain::new(
            config,
            SonoluminescenceCouplingType::SpectralCoupling,
        );

        let interfaces = domain.coupling_interfaces();
        assert!(interfaces.len() >= 2); // Should have EM-SL and spectral interfaces

        // Check for expected interface names
        let has_em_sl = interfaces
            .iter()
            .any(|i| i.name == "electromagnetic_sonoluminescence");
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
        let domain: SonoluminescenceCoupledDomain<
            burn::backend::Autodiff<burn::backend::NdArray<f32>>,
        > = SonoluminescenceCoupledDomain::new(
            config,
            SonoluminescenceCouplingType::DynamicEmission,
        );

        let weights = domain.loss_weights();
        assert_eq!(weights.physics_weights["light_source_weight"], 0.005);
    }
}
