//! Chemistry module for sonochemical reactions and radical formation
//!
//! Design principles:
//! - Separation of Concerns: Each sub-module handles a specific aspect
//! - Open/Closed: Easy to add new reaction types without modifying existing code  
//! - Interface Segregation: Traits for specific chemical behaviors
//! - Dependency Inversion: Depends on abstractions (traits) not concrete types
//! - Single Responsibility: Each component has one clear purpose

use crate::error::{KwaversResult, PhysicsError};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::plugin::PluginContext;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::traits::ChemicalModelTrait;
use ndarray::Array3;
use ndarray::ArrayView3;
use std::collections::HashMap;
use std::time::Instant;
use log::debug;

// Sub-modules
pub mod photochemistry;
pub mod radical_initiation;
pub mod reaction_kinetics;
pub mod ros_plasma;

// Re-export commonly used types
pub use ros_plasma::{ROSSpecies, ROSConcentrations, SonochemistryModel, SonochemicalYield};

// Define reaction types locally
#[derive(Debug, Clone)]
pub struct ChemicalReaction {
    pub name: String,
    pub rate_constant: f64,
}

impl ChemicalReaction {
    pub fn rate_constant(&self, _temperature: f64, _pressure: f64) -> f64 {
        self.rate_constant
    }
}

#[derive(Debug, Clone)]
pub struct ReactionRate {
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct Species {
    pub name: String,
    pub concentration: f64,
}

/// Parameters for chemical update operations
/// Follows SOLID principles by grouping related parameters
#[derive(Debug, Clone)]
pub struct ChemicalUpdateParams<'a> {
    pub pressure: &'a Array3<f64>,
    pub light: &'a Array3<f64>,
    pub emission_spectrum: &'a Array3<f64>,
    pub bubble_radius: &'a Array3<f64>,
    pub temperature: &'a Array3<f64>,
    pub grid: &'a Grid,
    pub dt: f64,
    pub medium: &'a dyn Medium,
    pub frequency: f64,
}

impl<'a> ChemicalUpdateParams<'a> {
    /// Create new chemical update parameters
    /// Follows Information Expert principle - validates its own parameters
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pressure: &'a Array3<f64>,
        light: &'a Array3<f64>,
        emission_spectrum: &'a Array3<f64>,
        bubble_radius: &'a Array3<f64>,
        temperature: &'a Array3<f64>,
        grid: &'a Grid,
        dt: f64,
        medium: &'a dyn Medium,
        frequency: f64,
    ) -> KwaversResult<Self> {
        // Validate parameters following Information Expert principle
        if dt <= 0.0 {
            return Err(crate::error::NumericalError::Instability {
                operation: "ChemicalUpdateParams validation".to_string(),
                condition: "Time step must be positive".to_string(),
            }.into());
        }

        if frequency <= 0.0 {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalUpdateParams".to_string(),
                reason: "Frequency must be positive".to_string(),
            }.into());
        }

        // Validate array dimensions match grid
        let (nx, ny, nz) = grid.dimensions();
        let expected_shape = [nx, ny, nz];
        
        if pressure.shape() != expected_shape {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalUpdateParams".to_string(),
                reason: format!("Pressure array shape {:?} doesn't match grid dimensions {:?}", 
                               pressure.shape(), expected_shape),
            }.into());
        }

        if light.shape() != expected_shape {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalUpdateParams".to_string(),
                reason: format!("Light array shape {:?} doesn't match grid dimensions {:?}", 
                               light.shape(), expected_shape),
            }.into());
        }

        if emission_spectrum.shape() != expected_shape {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalUpdateParams".to_string(),
                reason: format!("Emission spectrum array shape {:?} doesn't match grid dimensions {:?}", 
                               emission_spectrum.shape(), expected_shape),
            }.into());
        }

        if bubble_radius.shape() != expected_shape {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalUpdateParams".to_string(),
                reason: format!("Bubble radius array shape {:?} doesn't match grid dimensions {:?}", 
                               bubble_radius.shape(), expected_shape),
            }.into());
        }

        if temperature.shape() != expected_shape {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalUpdateParams".to_string(),
                reason: format!("Temperature array shape {:?} doesn't match grid dimensions {:?}", 
                               temperature.shape(), expected_shape),
            }.into());
        }

        Ok(Self {
            pressure,
            light,
            emission_spectrum,
            bubble_radius,
            temperature,
            grid,
            dt,
            medium,
            frequency,
        })
    }

    /// Validate chemical parameters for numerical stability
    pub fn validate(&self) -> KwaversResult<()> {
        // Check for NaN or infinite values in pressure field
        let invalid_pressure_values: Vec<f64> = self.pressure.iter()
            .filter(|&&x| x.is_nan() || x.is_infinite())
            .copied()
            .collect();
            
        if !invalid_pressure_values.is_empty() {
            return Err(crate::error::NumericalError::NaN {
                operation: "Chemical parameter validation".to_string(),
                inputs: invalid_pressure_values,
            }.into());
        }

        // Check for negative temperatures
        let negative_temperatures: Vec<f64> = self.temperature.iter()
            .filter(|&&x| x < 0.0)
            .copied()
            .collect();
            
        if !negative_temperatures.is_empty() {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalUpdateParams".to_string(),
                reason: format!("Temperature must be non-negative. Found {} negative values, first: {:.2e} K", 
                               negative_temperatures.len(), 
                               negative_temperatures[0]),
            }.into());
        }

        // Check for extremely high temperatures (> 1000 K)
        let high_temperatures: Vec<f64> = self.temperature.iter()
            .filter(|&&x| x > 1000.0)
            .copied()
            .collect();
            
        if !high_temperatures.is_empty() {
            log::warn!("High temperatures detected in chemical model: {} values > 1000K, max: {:.2e} K", 
                      high_temperatures.len(), 
                      high_temperatures.iter().fold(0.0f64, |a, &b| a.max(b)));
        }

        // Check for extremely high pressures (> 100 MPa)
        let high_pressures: Vec<f64> = self.pressure.iter()
            .filter(|&&x| x.abs() > 1e8)
            .copied()
            .collect();
            
        if !high_pressures.is_empty() {
            log::warn!("High pressures detected in chemical model: {} values > 100 MPa, max: {:.2e} Pa", 
                      high_pressures.len(), 
                      high_pressures.iter().fold(0.0f64, |a, &b| a.max(b.abs())));
        }

        Ok(())
    }
}

/// Performance metrics for chemical reactions
/// Follows SSOT principle for performance tracking
#[derive(Debug, Clone)]
pub struct ChemicalMetrics {
    pub radical_initiation_time: f64,
    pub kinetics_time: f64,
    pub photochemical_time: f64,
    pub total_update_time: f64,
    pub call_count: usize,
    pub memory_usage: usize,
    pub reaction_rates: HashMap<String, f64>,
}

impl Default for ChemicalMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ChemicalMetrics {
    pub fn new() -> Self {
        Self {
            radical_initiation_time: 0.0,
            kinetics_time: 0.0,
            photochemical_time: 0.0,
            total_update_time: 0.0,
            call_count: 0,
            memory_usage: 0,
            reaction_rates: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn get_average_times(&self) -> HashMap<String, f64> {
        if self.call_count == 0 {
            return HashMap::new();
        }
        
        let count = self.call_count as f64;
        let mut averages = HashMap::new();
        averages.insert("radical_initiation_time".to_string(), self.radical_initiation_time / count);
        averages.insert("kinetics_time".to_string(), self.kinetics_time / count);
        averages.insert("photochemical_time".to_string(), self.photochemical_time / count);
        averages.insert("total_update_time".to_string(), self.total_update_time / count);
        averages
    }

    pub fn add_reaction_rate(&mut self, reaction: String, rate: f64) {
        self.reaction_rates.insert(reaction, rate);
    }
}

/// Chemical reaction configuration
/// Follows SSOT principle for reaction definitions
#[derive(Debug, Clone)]
pub struct ChemicalReactionConfig {
    pub reaction_type: ReactionType,
    pub activation_energy: f64,
    pub pre_exponential_factor: f64,
    pub temperature_dependence: TemperatureDependence,
    pub pressure_dependence: PressureDependence,
    pub light_dependence: LightDependence,
}

#[derive(Debug, Clone)]
pub enum ReactionType {
    RadicalFormation,
    RadicalRecombination,
    HydrogenPeroxideFormation,
    ReactiveOxygenSpecies,
    Photochemical,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum TemperatureDependence {
    Arrhenius { activation_energy: f64 },
    ModifiedArrhenius { activation_energy: f64, temperature_exponent: f64 },
    None,
}

#[derive(Debug, Clone)]
pub enum PressureDependence {
    Linear { coefficient: f64 },
    Exponential { coefficient: f64 },
    None,
}

#[derive(Debug, Clone)]
pub enum LightDependence {
    Linear { coefficient: f64 },
    Exponential { coefficient: f64 },
    Threshold { threshold: f64, coefficient: f64 },
    None,
}

impl ChemicalReactionConfig {
    pub fn new(reaction_type: ReactionType) -> Self {
        Self {
            reaction_type,
            activation_energy: 0.0,
            pre_exponential_factor: 1.0,
            temperature_dependence: TemperatureDependence::None,
            pressure_dependence: PressureDependence::None,
            light_dependence: LightDependence::None,
        }
    }

    pub fn with_arrhenius(mut self, activation_energy: f64, pre_exponential_factor: f64) -> Self {
        self.activation_energy = activation_energy;
        self.pre_exponential_factor = pre_exponential_factor;
        self.temperature_dependence = TemperatureDependence::Arrhenius { activation_energy };
        self
    }

    pub fn with_pressure_dependence(mut self, dependence: PressureDependence) -> Self {
        self.pressure_dependence = dependence;
        self
    }

    pub fn with_light_dependence(mut self, dependence: LightDependence) -> Self {
        self.light_dependence = dependence;
        self
    }

    pub fn validate(&self) -> KwaversResult<()> {
        if self.pre_exponential_factor <= 0.0 {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalReactionConfig".to_string(),
                reason: "Pre-exponential factor must be positive".to_string(),
            }.into());
        }

        if self.activation_energy < 0.0 {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalReactionConfig".to_string(),
                reason: "Activation energy must be non-negative".to_string(),
            }.into());
        }

        Ok(())
    }
}

use radical_initiation::RadicalInitiation;
use photochemistry::PhotochemicalEffects;
use reaction_kinetics::ReactionKinetics;

/// Enhanced chemical model with better design principles
/// 
/// Design Principles Implemented:
/// - SOLID: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
/// - CUPID: Composable, Unix-like, Predictable, Idiomatic, Domain-focused
/// - GRASP: Information expert, creator, controller, low coupling, high cohesion
/// - SSOT: Single source of truth for chemical state and metrics
/// - ADP: Acyclic dependency principle
#[derive(Debug, Clone)]
pub struct ChemicalModel {
    radical_initiation: RadicalInitiation,
    kinetics: Option<ReactionKinetics>,
    photochemical: Option<PhotochemicalEffects>,
    enable_kinetics: bool,
    enable_photochemical: bool,
    metrics: ChemicalMetrics,
    reaction_configs: HashMap<String, ChemicalReactionConfig>,
    state: ChemicalModelState,
    computation_time: std::time::Duration,
    update_count: usize,
    reactions: HashMap<String, ChemicalReaction>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChemicalModelState {
    Initialized,
    Ready,
    Running,
    Completed,
    Error(String),
}

impl ChemicalModel {
    /// Create a new chemical model
    /// Follows GRASP Creator principle - creates objects it has information to create
    pub fn new(
        grid: &Grid, 
        enable_kinetics: bool, 
        enable_photochemical: bool
    ) -> KwaversResult<Self> {
        debug!(
            "Initializing ChemicalModel, kinetics: {}, photochemical: {}",
            enable_kinetics, enable_photochemical
        );

        // Validate grid following Information Expert principle
        let (nx, ny, nz) = grid.dimensions();
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalModel".to_string(),
                reason: "Grid dimensions must be positive".to_string(),
            }.into());
        }

        let radical_initiation = RadicalInitiation::new(grid);
        let kinetics = if enable_kinetics { 
            Some(ReactionKinetics::new(grid)) 
        } else { 
            None 
        };
        let photochemical = if enable_photochemical {
            Some(PhotochemicalEffects::new(grid))
        } else {
            None
        };

        Ok(Self {
            radical_initiation,
            kinetics,
            photochemical,
            enable_kinetics,
            enable_photochemical,
            metrics: ChemicalMetrics::new(),
            reaction_configs: HashMap::new(),
            state: ChemicalModelState::Initialized,
            computation_time: std::time::Duration::ZERO,
            update_count: 0,
            reactions: HashMap::new(),
        })
    }

    /// Add a chemical reaction configuration
    /// Follows Open/Closed principle - extends functionality without modification
    pub fn add_reaction_config(&mut self, name: String, config: ChemicalReactionConfig) -> KwaversResult<()> {
        config.validate()?;
        self.reaction_configs.insert(name, config);
        self.state = ChemicalModelState::Ready;
        Ok(())
    }

    /// Update chemical effects using parameter struct
    /// Follows SOLID principles by reducing parameter coupling
    pub fn update_chemical(&mut self, params: &ChemicalUpdateParams) -> KwaversResult<()> {
        let total_start_time = Instant::now();
        
        // Validate parameters
        params.validate()?;
        
        // Validate model state
        if self.state != ChemicalModelState::Ready && self.state != ChemicalModelState::Running {
            return Err(PhysicsError::InvalidState {
                field: "state".to_string(),
                value: format!("{:?}", self.state),
                reason: "Expected Ready or Running state".to_string(),
            }.into());
        }

        self.state = ChemicalModelState::Running;
        debug!("Updating chemical effects");

        // Update radical initiation
        let radical_start = Instant::now();
        self.radical_initiation.update_radicals(
            params.pressure, 
            params.light, 
            params.bubble_radius, 
            params.grid, 
            params.dt, 
            params.medium, 
            params.frequency
        );
        self.metrics.radical_initiation_time += radical_start.elapsed().as_secs_f64();

        // Update reaction kinetics if enabled
        if self.enable_kinetics {
            if let Some(ref mut kinetics) = self.kinetics {
                let kinetics_start = Instant::now();
                kinetics.update_reactions(
                    &self.radical_initiation.radical_concentration,
                    params.temperature,
                    params.grid,
                    params.dt,
                    params.medium,
                );
                self.metrics.kinetics_time += kinetics_start.elapsed().as_secs_f64();
            }
        }

        // Update photochemical effects if enabled
        if self.enable_photochemical {
            if let Some(ref mut photochemical) = self.photochemical {
                let photochemical_start = Instant::now();
                photochemical.update_photochemical(
                    params.light,
                    params.emission_spectrum,
                    params.bubble_radius,
                    params.temperature,
                    params.grid,
                    params.dt,
                    params.medium,
                );
                self.metrics.photochemical_time += photochemical_start.elapsed().as_secs_f64();
            }
        }

        // Update metrics
        self.metrics.call_count += 1;
        self.metrics.total_update_time += total_start_time.elapsed().as_secs_f64();
        self.metrics.memory_usage = self.calculate_memory_usage();

        self.state = ChemicalModelState::Completed;
        Ok(())
    }

    /// Update chemical effects using views to avoid unnecessary cloning
    /// This is the efficient version that works with array views
    pub fn update_chemical_with_views(
        &mut self,
        pressure: ArrayView3<f64>,
        light: ArrayView3<f64>,
        emission_spectrum: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        temperature: ArrayView3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Create params struct from views
        let pressure_owned = pressure.to_owned();
        let light_owned = light.to_owned();
        let temperature_owned = temperature.to_owned();
        let params = ChemicalUpdateParams {
            pressure: &pressure_owned,
            light: &light_owned,
            emission_spectrum,
            bubble_radius,
            temperature: &temperature_owned,
            grid,
            dt,
            medium,
            frequency: 1e6, // Default frequency, should be passed as parameter
        };
        
        self.update_chemical(&params)
    }

    /// Get radical concentration
    pub fn radical_concentration(&self) -> &Array3<f64> {
        &self.radical_initiation.radical_concentration
    }

    /// Get hydroxyl concentration
    pub fn hydroxyl_concentration(&self) -> Option<&Array3<f64>> {
        self.kinetics.as_ref().map(|k| &k.hydroxyl_concentration)
    }

    /// Get hydrogen peroxide concentration
    pub fn hydrogen_peroxide(&self) -> Option<&Array3<f64>> {
        self.kinetics.as_ref().map(|k| &k.hydrogen_peroxide)
    }

    /// Get reactive oxygen species concentration
    pub fn reactive_oxygen_species(&self) -> Option<&Array3<f64>> {
        self.photochemical.as_ref().map(|p| &p.reactive_oxygen_species)
    }

    /// Get performance metrics
    /// Follows SSOT principle - single source of truth for metrics
    pub fn get_metrics(&self) -> &ChemicalMetrics {
        &self.metrics
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }

    /// Get model state
    pub fn state(&self) -> &ChemicalModelState {
        &self.state
    }

    /// Reset model to initial state
    pub fn reset(&mut self) -> KwaversResult<()> {
        // Reset arrays to zero
        self.radical_initiation.radical_concentration.fill(0.0);
        if let Some(ref mut kinetics) = self.kinetics {
            kinetics.hydroxyl_concentration.fill(0.0);
            kinetics.hydrogen_peroxide.fill(0.0);
        }
        if let Some(ref mut photochemical) = self.photochemical {
            photochemical.reactive_oxygen_species.fill(0.0);
        }
        self.state = ChemicalModelState::Initialized;
        Ok(())
    }

    /// Validate model configuration
    /// Follows Information Expert principle - knows how to validate itself
    pub fn validate(&self) -> KwaversResult<()> {
        // Basic validation - check for NaN or infinite values
        if self.radical_initiation.radical_concentration.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(crate::error::NumericalError::NaN {
                operation: "Chemical model validation".to_string(),
                inputs: vec![0.0],
            }.into());
        }
        Ok(())
    }

    /// Calculate memory usage
    fn calculate_memory_usage(&self) -> usize {
        let mut total = 0;
        total += self.radical_initiation.radical_concentration.len() * std::mem::size_of::<f64>();
        if let Some(ref kinetics) = self.kinetics {
            total += kinetics.hydroxyl_concentration.len() * std::mem::size_of::<f64>();
            total += kinetics.hydrogen_peroxide.len() * std::mem::size_of::<f64>();
        }
        if let Some(ref photochemical) = self.photochemical {
            total += photochemical.reactive_oxygen_species.len() * std::mem::size_of::<f64>();
        }
        total
    }

    /// Report performance metrics
    pub fn report_performance(&self) {
        let averages = self.metrics.get_average_times();
        debug!("ChemicalModel Performance Report:");
        debug!("  Total calls: {}", self.metrics.call_count);
        debug!("  Average radical initiation time: {:.6} ms", 
               averages.get("radical_initiation_time").unwrap_or(&0.0) * 1000.0);
        debug!("  Average kinetics time: {:.6} ms", 
               averages.get("kinetics_time").unwrap_or(&0.0) * 1000.0);
        debug!("  Average photochemical time: {:.6} ms", 
               averages.get("photochemical_time").unwrap_or(&0.0) * 1000.0);
        debug!("  Average total update time: {:.6} ms", 
               averages.get("total_update_time").unwrap_or(&0.0) * 1000.0);
        debug!("  Memory usage: {} MB", self.metrics.memory_usage / (1024 * 1024));
        
        if !self.metrics.reaction_rates.is_empty() {
            debug!("  Reaction rates:");
            for (reaction, rate) in &self.metrics.reaction_rates {
                debug!("    {}: {:.6e}", reaction, rate);
            }
        }
    }
}

impl ChemicalModelTrait for ChemicalModel {
    fn update_chemical(
        &mut self,
        p: &Array3<f64>,
        light: &Array3<f64>,
        emission_spectrum: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        temperature: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
        frequency: f64,
    ) {
        let start_time = Instant::now();
        
        // Create fields map for update
        let mut fields = HashMap::new();
        fields.insert(UnifiedFieldType::Pressure, p.clone());
        fields.insert(UnifiedFieldType::Light, light.clone());
        fields.insert(UnifiedFieldType::Temperature, temperature.clone());
        fields.insert(UnifiedFieldType::Cavitation, bubble_radius.clone());
        
        // Create context with proper structure
        let mut context = PluginContext::new(0, 1000, frequency);
        context.parameters.insert("dt".to_string(), dt);
        context.parameters.insert("time".to_string(), 0.0);
        
        // Add grid parameters
        context.parameters.insert("nx".to_string(), grid.nx as f64);
        context.parameters.insert("ny".to_string(), grid.ny as f64);
        context.parameters.insert("nz".to_string(), grid.nz as f64);
        context.parameters.insert("dx".to_string(), grid.dx);
        context.parameters.insert("dy".to_string(), grid.dy);
        context.parameters.insert("dz".to_string(), grid.dz);
        
        // Add medium parameters
        context.parameters.insert("density".to_string(), medium.density(0.0, 0.0, 0.0, grid));
        context.parameters.insert("sound_speed".to_string(), medium.sound_speed(0.0, 0.0, 0.0, grid));
        
        // Update using the parameters approach
        let params = ChemicalUpdateParams::new(
            p, light, emission_spectrum, bubble_radius, temperature,
            grid, dt, medium, frequency
        ).unwrap_or_else(|e| {
            log::error!("Failed to create chemical update params: {}", e);
            panic!("Chemical update params creation failed");
        });
        
        if let Err(e) = self.update_chemical(&params) {
            log::error!("Chemical update failed: {}", e);
        }
        
        self.computation_time += start_time.elapsed();
        self.update_count += 1;
    }

    fn radical_concentration(&self) -> &Array3<f64> {
        self.radical_concentration()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_chemical_update_params_validation() {
        let grid = Grid::new(10, 10, 10, 1e-4, 1e-4, 1e-4);
        let pressure = Array3::zeros((10, 10, 10));
        let light = Array3::zeros((10, 10, 10));
        let emission_spectrum = Array3::zeros((10, 10, 10));
        let bubble_radius = Array3::zeros((10, 10, 10));
        let temperature = Array3::zeros((10, 10, 10));
        let medium = crate::medium::homogeneous::HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);

        // Valid parameters
        let params = ChemicalUpdateParams::new(
            &pressure, &light, &emission_spectrum, &bubble_radius, &temperature,
            &grid, 1e-8, &medium, 1e6
        ).unwrap();
        assert!(params.validate().is_ok());

        // Invalid time step
        let result = ChemicalUpdateParams::new(
            &pressure, &light, &emission_spectrum, &bubble_radius, &temperature,
            &grid, -1e-8, &medium, 1e6
        );
        assert!(result.is_err());

        // Invalid frequency
        let result = ChemicalUpdateParams::new(
            &pressure, &light, &emission_spectrum, &bubble_radius, &temperature,
            &grid, 1e-8, &medium, -1e6
        );
        assert!(result.is_err());

        // Invalid pressure array shape
        let wrong_pressure = Array3::zeros((5, 5, 5));
        let result = ChemicalUpdateParams::new(
            &wrong_pressure, &light, &emission_spectrum, &bubble_radius, &temperature,
            &grid, 1e-8, &medium, 1e6
        );
        assert!(result.is_err());

        // Invalid light array shape
        let wrong_light = Array3::zeros((5, 5, 5));
        let result = ChemicalUpdateParams::new(
            &pressure, &wrong_light, &emission_spectrum, &bubble_radius, &temperature,
            &grid, 1e-8, &medium, 1e6
        );
        assert!(result.is_err());

        // Invalid emission_spectrum array shape
        let wrong_emission_spectrum = Array3::zeros((5, 5, 5));
        let result = ChemicalUpdateParams::new(
            &pressure, &light, &wrong_emission_spectrum, &bubble_radius, &temperature,
            &grid, 1e-8, &medium, 1e6
        );
        assert!(result.is_err());

        // Invalid bubble_radius array shape
        let wrong_bubble_radius = Array3::zeros((5, 5, 5));
        let result = ChemicalUpdateParams::new(
            &pressure, &light, &emission_spectrum, &wrong_bubble_radius, &temperature,
            &grid, 1e-8, &medium, 1e6
        );
        assert!(result.is_err());

        // Invalid temperature array shape
        let wrong_temperature = Array3::zeros((5, 5, 5));
        let result = ChemicalUpdateParams::new(
            &pressure, &light, &emission_spectrum, &bubble_radius, &wrong_temperature,
            &grid, 1e-8, &medium, 1e6
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_chemical_model_creation() {
        let grid = Grid::new(10, 10, 10, 1e-4, 1e-4, 1e-4);
        let model = ChemicalModel::new(&grid, true, true).unwrap();
        assert_eq!(*model.state(), ChemicalModelState::Initialized);
    }

    #[test]
    fn test_reaction_config_validation() {
        let config = ChemicalReactionConfig::new(ReactionType::RadicalFormation)
            .with_arrhenius(1000.0, 1e12);
        assert!(config.validate().is_ok());

        let invalid_config = ChemicalReactionConfig::new(ReactionType::RadicalFormation)
            .with_arrhenius(-1000.0, 1e12);
        assert!(invalid_config.validate().is_err());
    }
}

