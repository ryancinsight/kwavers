//! State management for interactive controls

use super::parameter::{ParameterDefinition, ParameterValue};
use super::validation::ParameterValidator;
use crate::core::error::{KwaversError, KwaversResult};
use log::{debug, info, warn};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Control state for a single parameter
#[derive(Debug, Clone)]
pub struct ControlState {
    pub definition: ParameterDefinition,
    pub current_value: ParameterValue,
    pub last_update: Instant,
    pub update_count: usize,
}

impl ControlState {
    /// Create a control state from definition
    pub fn from_definition(definition: ParameterDefinition) -> Self {
        Self {
            current_value: definition.default_value.clone(),
            definition,
            last_update: Instant::now(),
            update_count: 0,
        }
    }

    /// Update the value with validation
    pub fn update_value(&mut self, value: ParameterValue) -> KwaversResult<()> {
        let validated =
            ParameterValidator::validate_and_apply(value, &self.definition.parameter_type)?;

        self.current_value = validated;
        self.last_update = Instant::now();
        self.update_count += 1;

        Ok(())
    }

    /// Reset to default value
    pub fn reset(&mut self) {
        self.current_value = self.definition.default_value.clone();
        self.last_update = Instant::now();
        self.update_count += 1;
    }
}

/// State snapshot for saving/loading
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub parameters: HashMap<String, ParameterValue>,
    pub timestamp: Instant,
}

impl StateSnapshot {
    /// Create a snapshot from current states
    pub fn from_states(states: &HashMap<String, ControlState>) -> Self {
        let parameters = states
            .iter()
            .map(|(k, v)| (k.clone(), v.current_value.clone()))
            .collect();

        Self {
            parameters,
            timestamp: Instant::now(),
        }
    }

    /// Apply snapshot to states
    pub fn apply_to_states(&self, states: &mut HashMap<String, ControlState>) -> KwaversResult<()> {
        for (name, value) in &self.parameters {
            if let Some(state) = states.get_mut(name) {
                state.update_value(value.clone())?;
            } else {
                warn!("Parameter {} not found in current state", name);
            }
        }
        Ok(())
    }
}

/// Interactive controls system with thread-safe state management
pub struct InteractiveControls {
    states: Arc<RwLock<HashMap<String, ControlState>>>,
    #[allow(clippy::type_complexity)]
    update_callbacks: Arc<RwLock<HashMap<String, Box<dyn Fn(&ParameterValue) + Send + Sync>>>>,
    history: Arc<RwLock<Vec<StateSnapshot>>>,
    max_history_size: usize,
}

impl std::fmt::Debug for InteractiveControls {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveControls")
            .field("states", &self.states)
            .field("history", &self.history)
            .field("max_history_size", &self.max_history_size)
            .finish()
    }
}

impl InteractiveControls {
    /// Create a control system
    pub fn new() -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            update_callbacks: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            max_history_size: 100,
        }
    }

    /// Create a control system with configuration (alias for new)
    pub fn create(
        _config: &crate::analysis::visualization::VisualizationConfig,
    ) -> KwaversResult<Self> {
        Ok(Self::new())
    }

    /// Register a parameter
    pub fn register_parameter(&self, definition: ParameterDefinition) -> KwaversResult<()> {
        let mut states = self
            .states
            .write()
            .map_err(|_| KwaversError::ConcurrencyError {
                message: "register_parameter on states: Lock poisoned".to_string(),
            })?;

        let name = definition.name.clone();
        states.insert(name.clone(), ControlState::from_definition(definition));

        info!("Registered parameter: {}", name);
        Ok(())
    }

    /// Update a parameter value
    pub fn update_parameter(&self, name: &str, value: ParameterValue) -> KwaversResult<()> {
        // Update state
        {
            let mut states = self
                .states
                .write()
                .map_err(|_| KwaversError::ConcurrencyError {
                    message: "update_parameter on states: Lock poisoned".to_string(),
                })?;

            let state = states.get_mut(name).ok_or_else(|| {
                KwaversError::InvalidInput(format!("Parameter {} not found", name))
            })?;

            state.update_value(value.clone())?;
        }

        // Call update callback if registered
        {
            let callbacks =
                self.update_callbacks
                    .read()
                    .map_err(|_| KwaversError::ConcurrencyError {
                        message: "update_parameter on callbacks: Lock poisoned".to_string(),
                    })?;

            if let Some(callback) = callbacks.get(name) {
                callback(&value);
            }
        }

        debug!("Updated parameter {}: {}", name, value);
        Ok(())
    }

    /// Get current value of a parameter
    pub fn get_value(&self, name: &str) -> KwaversResult<ParameterValue> {
        let states = self
            .states
            .read()
            .map_err(|_| KwaversError::ConcurrencyError {
                message: "get_value on states: Lock poisoned".to_string(),
            })?;

        states
            .get(name)
            .map(|s| s.current_value.clone())
            .ok_or_else(|| KwaversError::InvalidInput(format!("Parameter {} not found", name)))
    }

    /// Register an update callback
    pub fn register_callback<F>(&self, name: String, callback: F) -> KwaversResult<()>
    where
        F: Fn(&ParameterValue) + Send + Sync + 'static,
    {
        let mut callbacks =
            self.update_callbacks
                .write()
                .map_err(|_| KwaversError::ConcurrencyError {
                    message: "register_callback on callbacks: Lock poisoned".to_string(),
                })?;

        callbacks.insert(name.clone(), Box::new(callback));
        debug!("Registered callback for parameter: {}", name);
        Ok(())
    }

    /// Save current state to history
    pub fn save_snapshot(&self) -> KwaversResult<()> {
        let states = self
            .states
            .read()
            .map_err(|_| KwaversError::ConcurrencyError {
                message: "save_snapshot on states: Lock poisoned".to_string(),
            })?;

        let snapshot = StateSnapshot::from_states(&states);

        let mut history = self
            .history
            .write()
            .map_err(|_| KwaversError::ConcurrencyError {
                message: "save_snapshot on history: Lock poisoned".to_string(),
            })?;

        history.push(snapshot);

        // Limit history size
        if history.len() > self.max_history_size {
            history.remove(0);
        }

        Ok(())
    }

    /// Restore a snapshot
    pub fn restore_snapshot(&self, index: usize) -> KwaversResult<()> {
        let history = self
            .history
            .read()
            .map_err(|_| KwaversError::ConcurrencyError {
                message: "restore_snapshot on history: Lock poisoned".to_string(),
            })?;

        let snapshot = history.get(index).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Snapshot index {} out of range", index))
        })?;

        let mut states = self
            .states
            .write()
            .map_err(|_| KwaversError::ConcurrencyError {
                message: "restore_snapshot on states: Lock poisoned".to_string(),
            })?;

        snapshot.apply_to_states(&mut states)?;

        info!("Restored snapshot {}", index);
        Ok(())
    }

    /// Get all current parameter states
    pub fn get_all_states(&self) -> KwaversResult<HashMap<String, ControlState>> {
        let states = self
            .states
            .read()
            .map_err(|_| KwaversError::ConcurrencyError {
                message: "get_all_states on states: Lock poisoned".to_string(),
            })?;

        Ok(states.clone())
    }
}

impl Default for InteractiveControls {
    fn default() -> Self {
        Self::new()
    }
}
