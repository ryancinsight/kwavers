//! ChemicalModel struct and core lifecycle methods

use crate::core::error::{KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use log::debug;
use std::collections::HashMap;

use super::parameters::ChemicalMetrics;
use super::photochemistry::PhotochemicalEffects;
use super::radical_initiation::RadicalInitiation;
use super::reaction_kinetics::ReactionKinetics;
use super::reactions::{ChemicalReaction, ChemicalReactionConfig};

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
#[derive(Debug, Clone)]
pub struct ChemicalModel {
    pub(crate) radical_initiation: RadicalInitiation,
    pub(crate) kinetics: Option<ReactionKinetics>,
    pub(crate) photochemical: Option<PhotochemicalEffects>,
    pub(crate) enable_kinetics: bool,
    pub(crate) enable_photochemical: bool,
    pub(crate) metrics: ChemicalMetrics,
    pub(crate) reaction_configs: HashMap<String, ChemicalReactionConfig>,
    pub(crate) state: ChemicalModelState,
    pub(crate) computation_time: std::time::Duration,
    pub(crate) update_count: usize,
    pub(crate) reactions: HashMap<String, ChemicalReaction>,
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

    /// Reset the model state
    pub fn reset(&mut self) {
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
