//! Constructor, coupling interface factory, state update, and `Clone` impl.

use std::collections::HashMap;

use kwavers_physics::optics::sonoluminescence::SonoluminescenceEmission;
use crate::inverse::pinn::ml::physics::{
    BoundaryPosition, CouplingInterface, CouplingType,
};
use burn::tensor::backend::AutodiffBackend;

use super::super::config::{SonoluminescenceCouplingConfig, SonoluminescenceCouplingType};
use super::SonoluminescenceCoupledDomain;

impl<B: AutodiffBackend> SonoluminescenceCoupledDomain<B> {
    /// Create a new sonoluminescence-coupled domain.
    pub fn new(
        config: SonoluminescenceCouplingConfig,
        coupling_type: SonoluminescenceCouplingType,
    ) -> Self {
        let emission_calculator =
            SonoluminescenceEmission::new(config.grid_shape, config.emission_params.clone());

        let bubble_states = ndarray::Array3::zeros(config.grid_shape);
        let temperature_field = ndarray::Array3::zeros(config.grid_shape);

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

    pub(super) fn create_coupling_interfaces(
        config: &SonoluminescenceCouplingConfig,
        coupling_type: &SonoluminescenceCouplingType,
    ) -> Vec<CouplingInterface> {
        let mut interfaces = Vec::new();

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

    /// Update bubble state and temperature fields.
    pub fn update_bubble_states(
        &mut self,
        new_bubble_states: ndarray::Array3<f64>,
        new_temperature: ndarray::Array3<f64>,
    ) {
        self.bubble_states = new_bubble_states;
        self.temperature_field = new_temperature;
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
