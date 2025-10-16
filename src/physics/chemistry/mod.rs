//! Chemistry module for sonochemical reactions and radical formation
//!
//! Design principles:
//! - Separation of Concerns: Each sub-module handles a specific aspect
//! - Open/Closed: Easy to add new reaction types without modifying existing code  
//! - Interface Segregation: Traits for specific chemical behaviors
//! - Dependency Inversion: Depends on abstractions (traits) not concrete types
//! - Single Responsibility: Each component has one clear purpose

use crate::error::{KwaversResult, ValidationError};
use crate::grid::Grid;
use crate::physics::plugin::PluginContext;
use crate::physics::traits::ChemicalModelTrait;
use log::debug;
use ndarray::Array3;
use std::collections::HashMap;
use std::time::Instant;

// Sub-modules
pub mod parameters;
pub mod photochemistry;
pub mod radical_initiation;
pub mod reaction_kinetics;
pub mod reactions;
pub mod ros_plasma;

// Re-export commonly used types from submodules
pub use parameters::{ChemicalMetrics, ChemicalUpdateParams};
pub use reactions::{
    ChemicalReaction, ChemicalReactionConfig, LightDependence, PressureDependence, ReactionRate,
    ReactionType, Species, ThermalDependence,
};
pub use ros_plasma::{ROSConcentrations, ROSSpecies, SonochemicalYield, SonochemistryModel};

// Import submodule types for internal use
use photochemistry::PhotochemicalEffects;
use radical_initiation::RadicalInitiation;
use reaction_kinetics::ReactionKinetics;

// Constants for chemical modeling
#[allow(dead_code)] // Configuration constants for chemistry modeling
const HIGH_TEMPERATURE_THRESHOLD: f64 = 1000.0; // Kelvin
#[allow(dead_code)]
const EXTREME_PRESSURE_THRESHOLD: f64 = 100e6; // 100 MPa
#[allow(dead_code)]
const DEFAULT_REACTION_RATE: f64 = 1e-3; // Default rate constant

/// State of the chemical model
#[derive(Debug, Clone, PartialEq)]
pub enum ChemicalModelState {
    Initialized,
    Ready,
    Running,
    Completed,
    Error(String),
}

/// Main chemical model implementing sonochemistry and radical reactions
///
/// Design Principles:
/// - SOLID: Single responsibility per component, open for extension
/// - CUPID: Composable sub-models, predictable behavior
/// - GRASP: Information expert pattern for validation
/// - SSOT: Single source of truth for chemical state
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

impl ChemicalModel {
    /// Create a new chemical model
    pub fn new(
        grid: &Grid,
        enable_kinetics: bool,
        enable_photochemical: bool,
    ) -> KwaversResult<Self> {
        debug!(
            "Initializing ChemicalModel, kinetics: {}, photochemical: {}",
            enable_kinetics, enable_photochemical
        );

        // Validate grid dimensions
        let (nx, ny, nz) = grid.dimensions();
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(ValidationError::FieldValidation {
                field: "grid_dimensions".to_string(),
                value: format!("({nx}, {ny}, {nz})"),
                constraint: "All dimensions must be positive".to_string(),
            }
            .into());
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

    /// Update chemical reactions based on acoustic field
    pub fn update(
        &mut self,
        params: &ChemicalUpdateParams,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        let start = Instant::now();
        self.state = ChemicalModelState::Running;

        // Update radical initiation
        let radical_start = Instant::now();
        self.radical_initiation.update_radicals(
            params.pressure,
            params.light,
            params.bubble_radius,
            params.grid,
            params.dt,
            params.medium,
            params.frequency,
        );
        self.metrics
            .set_computation_time(radical_start.elapsed().as_secs_f64() * 1000.0);

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
                let kinetics_time = kinetics_start.elapsed().as_secs_f64() * 1000.0;
                self.metrics
                    .set_computation_time(self.metrics.computation_time_ms + kinetics_time);
            }
        }

        // Update photochemical effects if enabled
        if self.enable_photochemical {
            if let Some(ref mut photochemical) = self.photochemical {
                let photo_start = Instant::now();
                photochemical.update_photochemical(
                    params.light,
                    params.emission_spectrum,
                    params.bubble_radius,
                    params.temperature,
                    params.grid,
                    params.dt,
                    params.medium,
                );
                let photo_time = photo_start.elapsed().as_secs_f64() * 1000.0;
                self.metrics
                    .set_computation_time(self.metrics.computation_time_ms + photo_time);
            }
        }

        self.computation_time = start.elapsed();
        self.update_count += 1;
        self.state = ChemicalModelState::Completed;
        self.metrics.increment_reactions(1);

        Ok(())
    }

    /// Get radical concentrations for all tracked species
    ///
    /// Returns concentrations for:
    /// - OH (hydroxyl radical) from reaction kinetics (if enabled)
    /// - H2O2 (hydrogen peroxide) from reaction kinetics (if enabled)
    /// - General radical precursors from initiation
    #[must_use]
    pub fn get_radical_concentrations(&self) -> HashMap<String, Array3<f64>> {
        let mut map = HashMap::new();
        
        // If kinetics is enabled, get tracked species concentrations
        if let Some(ref kinetics) = self.kinetics {
            // Primary hydroxyl radical from reaction kinetics
            map.insert(
                "OH".to_string(),
                kinetics.hydroxyl_concentration().clone(),
            );
            
            // Hydrogen peroxide from reaction kinetics
            map.insert(
                "H2O2".to_string(),
                kinetics.hydrogen_peroxide().clone(),
            );
        }
        
        // General radical precursors (H•, OH• precursors) from initiation
        // Always available regardless of kinetics setting
        map.insert(
            "radical_precursors".to_string(),
            self.radical_initiation.radical_concentration.clone(),
        );
        
        map
    }

    /// Get reaction rates for tracked chemical reactions
    ///
    /// Returns rate constants for:
    /// - OH production rate (from radical initiation)
    /// - H2O2 production rate (from OH recombination)
    /// - Total reaction count
    ///
    /// Rates are computed from the current concentrations and reaction kinetics
    #[must_use]
    pub fn get_reaction_rates(&self) -> HashMap<String, f64> {
        let mut rates = HashMap::new();
        
        if let Some(ref kinetics) = self.kinetics {
            // Calculate average OH production rate from radical initiation
            let radical_conc = &self.radical_initiation.radical_concentration;
            let oh_conc = kinetics.hydroxyl_concentration();
            let h2o2_conc = kinetics.hydrogen_peroxide();
            
            // Average concentrations across grid
            let avg_radical = radical_conc.mean().unwrap_or(0.0);
            let avg_oh = oh_conc.mean().unwrap_or(0.0);
            let avg_h2o2 = h2o2_conc.mean().unwrap_or(0.0);
            
            // OH production from radicals (using base rate constant)
            rates.insert(
                "OH_production_rate".to_string(),
                avg_radical * crate::physics::constants::thermodynamic::SONOCHEMISTRY_BASE_RATE,
            );
            
            // H2O2 production from OH recombination (2 OH → H2O2)
            rates.insert(
                "H2O2_production_rate".to_string(),
                avg_oh * avg_oh * crate::physics::constants::thermodynamic::SECONDARY_REACTION_RATE,
            );
            
            // Store current average concentrations for reference
            rates.insert("OH_concentration_avg".to_string(), avg_oh);
            rates.insert("H2O2_concentration_avg".to_string(), avg_h2o2);
        }
        
        // Total number of reaction updates performed
        rates.insert("update_count".to_string(), self.update_count as f64);
        rates.insert("reactions_total".to_string(), self.metrics.total_reactions as f64);
        
        rates
    }

    /// Get photochemical emission spectrum
    #[must_use]
    pub fn get_emission_spectrum(&self) -> Option<&Array3<f64>> {
        // Return emission spectrum if photochemical is enabled
        self.photochemical
            .as_ref()
            .map(|p| &p.reactive_oxygen_species)
    }

    /// Get performance metrics
    #[must_use]
    pub fn get_metrics(&self) -> &ChemicalMetrics {
        &self.metrics
    }

    /// Reset the model state
    pub fn reset(&mut self) {
        // Reset radical concentrations
        self.radical_initiation.radical_concentration.fill(0.0);
        if let Some(ref mut kinetics) = self.kinetics {
            kinetics.hydroxyl_concentration.fill(0.0);
            kinetics.hydrogen_peroxide.fill(0.0);
        }
        if let Some(ref mut photochemical) = self.photochemical {
            photochemical.reactive_oxygen_species.fill(0.0);
        }
        self.metrics = ChemicalMetrics::new();
        self.state = ChemicalModelState::Initialized;
        self.computation_time = std::time::Duration::ZERO;
        self.update_count = 0;
    }

    /// Add a reaction configuration
    pub fn add_reaction_config(&mut self, name: String, config: ChemicalReactionConfig) {
        self.reaction_configs.insert(name.clone(), config.clone());
        self.reactions.insert(
            name,
            ChemicalReaction {
                name: config.reaction_type.to_string(),
                rate_constant: config.rate_constant,
            },
        );
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> &ChemicalModelState {
        &self.state
    }

    /// Get computation statistics
    #[must_use]
    pub fn get_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("update_count".to_string(), self.update_count as f64);
        stats.insert(
            "avg_computation_time_ms".to_string(),
            if self.update_count > 0 {
                self.computation_time.as_secs_f64() * 1000.0 / self.update_count as f64
            } else {
                0.0
            },
        );
        stats.insert(
            "total_computation_time_ms".to_string(),
            self.computation_time.as_secs_f64() * 1000.0,
        );
        stats
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
        medium: &dyn crate::medium::Medium,
        frequency: f64,
    ) {
        // Create ChemicalUpdateParams for the update
        let params_result = ChemicalUpdateParams::new(
            p,
            light,
            emission_spectrum,
            bubble_radius,
            temperature,
            grid,
            dt,
            medium,
            frequency,
        );

        match params_result {
            Ok(params) => {
                // Create a dummy context for now
                let dummy_pressure = Array3::zeros((1, 1, 1));
                let context = PluginContext::new(dummy_pressure);
                if let Err(e) = self.update(&params, &context) {
                    log::error!("Chemical update failed: {}", e);
                }
            }
            Err(e) => {
                log::error!("Failed to create chemical update params: {}", e);
            }
        }
    }

    fn radical_concentration(&self) -> &Array3<f64> {
        // Return the main radical concentration field
        &self.radical_initiation.radical_concentration
    }
}

impl std::fmt::Display for ReactionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ReactionType::Dissociation => "Dissociation",
            ReactionType::Recombination => "Recombination",
            ReactionType::Oxidation => "Oxidation",
            ReactionType::Reduction => "Reduction",
            ReactionType::Polymerization => "Polymerization",
        };
        write!(f, "{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chemical_model_creation() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let model = ChemicalModel::new(&grid, true, true).unwrap();
        assert_eq!(model.state(), &ChemicalModelState::Initialized);
    }

    #[test]
    fn test_reaction_config() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let mut model = ChemicalModel::new(&grid, true, false).unwrap();

        let config = ChemicalReactionConfig {
            reaction_type: ReactionType::Dissociation,
            thermal_dependence: ThermalDependence::Constant,
            pressure_dependence: PressureDependence::Constant,
            light_dependence: LightDependence::None,
            rate_constant: DEFAULT_REACTION_RATE,
            activation_energy: 0.0,
        };

        model.add_reaction_config("test_reaction".to_string(), config);
        assert_eq!(model.reactions.len(), 1);
    }
}
