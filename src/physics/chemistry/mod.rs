// physics/chemistry/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::error::{KwaversResult, PhysicsError, NumericalError};
use log::{debug, warn, trace};
use ndarray::Array3;
use std::collections::HashMap;
use std::time::Instant;

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
            return Err(NumericalError::Instability {
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
        let expected_shape = (nx, ny, nz);
        
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
        // Check for NaN or infinite values
        if self.pressure.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(NumericalError::NaN {
                operation: "Chemical parameter validation".to_string(),
                inputs: vec![0.0], // Placeholder
            }.into());
        }

        if self.temperature.iter().any(|&x| x < 0.0) {
            return Err(PhysicsError::InvalidConfiguration {
                component: "ChemicalUpdateParams".to_string(),
                reason: "Temperature must be non-negative".to_string(),
            }.into());
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

pub mod radical_initiation;
pub mod photochemistry;
pub mod reaction_kinetics;

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
#[derive(Debug)]
pub struct ChemicalModel {
    radical_initiation: RadicalInitiation,
    kinetics: Option<ReactionKinetics>,
    photochemical: Option<PhotochemicalEffects>,
    enable_kinetics: bool,
    enable_photochemical: bool,
    metrics: ChemicalMetrics,
    reaction_configs: HashMap<String, ChemicalReactionConfig>,
    state: ChemicalModelState,
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

        let radical_initiation = RadicalInitiation::new(grid)?;
        let kinetics = if enable_kinetics { 
            Some(ReactionKinetics::new(grid)?) 
        } else { 
            None 
        };
        let photochemical = if enable_photochemical {
            Some(PhotochemicalEffects::new(grid)?)
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
                expected: "Ready or Running".to_string(),
                actual: format!("{:?}", self.state),
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
        )?;
        self.metrics.radical_initiation_time += radical_start.elapsed().as_secs_f64();

        // Update reaction kinetics if enabled
        if self.enable_kinetics {
            if let Some(ref mut kinetics) = self.kinetics {
                let kinetics_start = Instant::now();
                kinetics.update_kinetics(
                    params.pressure,
                    params.temperature,
                    params.grid,
                    params.dt,
                    params.medium,
                    &self.reaction_configs,
                )?;
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
                    params.temperature,
                    params.grid,
                    params.dt,
                    params.medium,
                )?;
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

    /// Legacy update method for backward compatibility
    /// Follows Interface Segregation principle - provides alternative interface
    pub fn update_chemical_legacy(
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
    ) -> KwaversResult<()> {
        let params = ChemicalUpdateParams::new(
            p, light, emission_spectrum, bubble_radius, temperature,
            grid, dt, medium, frequency
        )?;
        self.update_chemical(&params)
    }

    /// Get radical concentration
    pub fn radical_concentration(&self) -> &Array3<f64> {
        self.radical_initiation.radical_concentration()
    }

    /// Get hydroxyl concentration
    pub fn hydroxyl_concentration(&self) -> Option<&Array3<f64>> {
        self.kinetics.as_ref().and_then(|k| k.hydroxyl_concentration())
    }

    /// Get hydrogen peroxide concentration
    pub fn hydrogen_peroxide(&self) -> Option<&Array3<f64>> {
        self.kinetics.as_ref().and_then(|k| k.hydrogen_peroxide())
    }

    /// Get reactive oxygen species concentration
    pub fn reactive_oxygen_species(&self) -> Option<&Array3<f64>> {
        self.kinetics.as_ref().and_then(|k| k.reactive_oxygen_species())
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
        self.radical_initiation.reset()?;
        if let Some(ref mut kinetics) = self.kinetics {
            kinetics.reset()?;
        }
        if let Some(ref mut photochemical) = self.photochemical {
            photochemical.reset()?;
        }
        self.state = ChemicalModelState::Initialized;
        Ok(())
    }

    /// Validate model configuration
    /// Follows Information Expert principle - knows how to validate itself
    pub fn validate(&self) -> KwaversResult<()> {
        self.radical_initiation.validate()?;
        if let Some(ref kinetics) = self.kinetics {
            kinetics.validate()?;
        }
        if let Some(ref photochemical) = self.photochemical {
            photochemical.validate()?;
        }
        Ok(())
    }

    /// Calculate memory usage
    fn calculate_memory_usage(&self) -> usize {
        let mut total = 0;
        total += self.radical_initiation.memory_usage();
        if let Some(ref kinetics) = self.kinetics {
            total += kinetics.memory_usage();
        }
        if let Some(ref photochemical) = self.photochemical {
            total += photochemical.memory_usage();
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

/// Trait for chemical model components
/// Follows Interface Segregation principle - defines minimal interface
pub trait ChemicalModelTrait {
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
    );

    fn radical_concentration(&self) -> &Array3<f64>;
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
        if let Err(e) = self.update_chemical_legacy(p, light, emission_spectrum, bubble_radius, temperature, grid, dt, medium, frequency) {
            warn!("Chemical update failed: {}", e);
        }
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
        let grid = Grid::new(10, 10, 10, 1e-4, 1e-4, 1e-4).unwrap();
        let pressure = Array3::zeros((10, 10, 10));
        let light = Array3::zeros((10, 10, 10));
        let emission_spectrum = Array3::zeros((10, 10, 10));
        let bubble_radius = Array3::zeros((10, 10, 10));
        let temperature = Array3::zeros((10, 10, 10));
        let medium = crate::medium::HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0).unwrap();

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
    }

    #[test]
    fn test_chemical_model_creation() {
        let grid = Grid::new(10, 10, 10, 1e-4, 1e-4, 1e-4).unwrap();
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