//! # Interactive Controls - Real-Time Parameter Adjustment
//!
//! This module provides real-time interactive controls for simulation parameters
//! during visualization. It implements immediate feedback systems with validation
//! and state management for live parameter updates.

use crate::error::{KwaversError, KwaversResult};
use crate::visualization::VisualizationConfig;
use log::{debug, info, warn};
use std::collections::HashMap;
use std::time::Instant;

#[cfg(feature = "advanced-visualization")]
use egui::{Context, Ui, Vec2, Window};

/// Parameter types supported by the interactive control system
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterType {
    /// Floating point parameter with range
    Float { min: f64, max: f64, step: f64 },
    /// Integer parameter with range
    Integer { min: i64, max: i64, step: i64 },
    /// Boolean toggle parameter
    Boolean,
    /// Enumeration with predefined values
    Enum { options: Vec<String> },
    /// 3D vector parameter
    Vector3 { min: f64, max: f64, step: f64 },
    /// Color parameter (RGB)
    Color,
}

/// Parameter definition for the control system
#[derive(Debug, Clone)]
pub struct ParameterDefinition {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub parameter_type: ParameterType,
    pub default_value: ParameterValue,
    pub group: String,
    pub is_realtime: bool,
}

/// Parameter value storage
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    Enum(String),
    Vector3([f64; 3]),
    Color([f32; 3]),
}

impl ParameterValue {
    /// Convert to f64 if possible
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(v) => Some(*v),
            ParameterValue::Integer(v) => Some(*v as f64),
            ParameterValue::Boolean(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    
    /// Convert to boolean if possible
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Boolean(v) => Some(*v),
            ParameterValue::Float(v) => Some(*v > 0.5),
            ParameterValue::Integer(v) => Some(*v != 0),
            _ => None,
        }
    }
}

/// Parameter change event for callbacks
#[derive(Debug, Clone)]
pub struct ParameterChangeEvent {
    pub parameter_name: String,
    pub old_value: ParameterValue,
    pub new_value: ParameterValue,
    pub timestamp: Instant,
}

/// Validation result for parameter changes
#[derive(Debug, Clone)]
pub enum ValidationResult {
    Valid,
    Warning(String),
    Error(String),
}

/// Interactive controls manager
pub struct InteractiveControls {
    config: VisualizationConfig,
    parameters: HashMap<String, ParameterDefinition>,
    current_values: HashMap<String, ParameterValue>,
    parameter_history: Vec<ParameterChangeEvent>,
    validation_callbacks: HashMap<String, Box<dyn Fn(&ParameterValue) -> ValidationResult + Send + Sync>>,
    change_callbacks: HashMap<String, Box<dyn Fn(&ParameterChangeEvent) -> KwaversResult<()> + Send + Sync>>,
    last_update_time: Instant,
    
    #[cfg(feature = "advanced-visualization")]
    ui_state: UiState,
}

#[cfg(feature = "advanced-visualization")]
#[derive(Debug)]
struct UiState {
    show_controls: bool,
    selected_group: String,
    parameter_groups: Vec<String>,
    search_filter: String,
    show_advanced: bool,
}

impl InteractiveControls {
    /// Create a new interactive controls manager
    pub fn new(config: &VisualizationConfig) -> KwaversResult<Self> {
        info!("Initializing interactive controls system");
        
        let mut controls = Self {
            config: config.clone(),
            parameters: HashMap::new(),
            current_values: HashMap::new(),
            parameter_history: Vec::new(),
            validation_callbacks: HashMap::new(),
            change_callbacks: HashMap::new(),
            last_update_time: Instant::now(),
            
            #[cfg(feature = "advanced-visualization")]
            ui_state: UiState {
                show_controls: true,
                selected_group: "Simulation".to_string(),
                parameter_groups: vec![
                    "Simulation".to_string(),
                    "Visualization".to_string(),
                    "Physics".to_string(),
                    "Advanced".to_string(),
                ],
                search_filter: String::new(),
                show_advanced: false,
            },
        };
        
        // Register default simulation parameters
        controls.register_default_parameters()?;
        
        Ok(controls)
    }
    
    /// Register a new parameter for interactive control
    pub fn register_parameter(&mut self, definition: ParameterDefinition) -> KwaversResult<()> {
        debug!("Registering parameter: {}", definition.name);
        
        // Set initial value
        let name = definition.name.clone();
        let default_value = definition.default_value.clone();
        self.current_values.insert(name.clone(), default_value);
        
        // Store parameter definition
        self.parameters.insert(name, definition);
        
        Ok(())
    }
    
    /// Update a parameter value with validation
    pub fn update_parameter(&mut self, name: &str, value: ParameterValue) -> KwaversResult<()> {
        let definition = self.parameters.get(name)
            .ok_or_else(|| KwaversError::Visualization(format!("Parameter '{}' not found", name)))?;
        
        // Validate parameter type and range
        self.validate_parameter_value(&definition.parameter_type, &value)?;
        
        // Run custom validation if available
        if let Some(validator) = self.validation_callbacks.get(name) {
            match validator(&value) {
                ValidationResult::Error(msg) => {
                    return Err(KwaversError::Visualization(format!("Parameter '{}': {}", name, msg)));
                }
                ValidationResult::Warning(msg) => {
                    warn!("Parameter '{}': {}", name, msg);
                }
                ValidationResult::Valid => {}
            }
        }
        
        // Create change event
        let old_value = self.current_values.get(name).cloned()
            .unwrap_or_else(|| definition.default_value.clone());
        
        let change_event = ParameterChangeEvent {
            parameter_name: name.to_string(),
            old_value,
            new_value: value.clone(),
            timestamp: Instant::now(),
        };
        
        // Update value
        self.current_values.insert(name.to_string(), value);
        self.parameter_history.push(change_event.clone());
        
        // Trigger change callback
        if let Some(callback) = self.change_callbacks.get(name) {
            callback(&change_event)?;
        }
        
        debug!("Updated parameter '{}' to {:?}", name, change_event.new_value);
        Ok(())
    }
    
    /// Get current parameter value
    pub fn get_parameter(&self, name: &str) -> Option<&ParameterValue> {
        self.current_values.get(name)
    }
    
    /// Get parameter as f64
    pub fn get_parameter_f64(&self, name: &str) -> Option<f64> {
        self.get_parameter(name)?.as_f64()
    }
    
    /// Get parameter as boolean
    pub fn get_parameter_bool(&self, name: &str) -> Option<bool> {
        self.get_parameter(name)?.as_bool()
    }
    
    /// Register validation callback for a parameter
    pub fn register_validator<F>(&mut self, name: &str, validator: F) -> KwaversResult<()>
    where
        F: Fn(&ParameterValue) -> ValidationResult + Send + Sync + 'static,
    {
        self.validation_callbacks.insert(name.to_string(), Box::new(validator));
        Ok(())
    }
    
    /// Register change callback for a parameter
    pub fn register_change_callback<F>(&mut self, name: &str, callback: F) -> KwaversResult<()>
    where
        F: Fn(&ParameterChangeEvent) -> KwaversResult<()> + Send + Sync + 'static,
    {
        self.change_callbacks.insert(name.to_string(), Box::new(callback));
        Ok(())
    }
    
    /// Get parameter change history
    pub fn get_parameter_history(&self) -> &[ParameterChangeEvent] {
        &self.parameter_history
    }
    
    /// Clear parameter history
    pub fn clear_history(&mut self) {
        self.parameter_history.clear();
    }
    
    /// Undo last parameter change
    pub fn undo_last_change(&mut self) -> KwaversResult<()> {
        if let Some(last_change) = self.parameter_history.pop() {
            self.current_values.insert(last_change.parameter_name.clone(), last_change.old_value);
            info!("Undid change to parameter: {}", last_change.parameter_name);
            Ok(())
        } else {
            Err(KwaversError::Visualization("No changes to undo".to_string()))
        }
    }
    
    /// Save current parameter state
    pub fn save_state(&self, name: &str) -> KwaversResult<HashMap<String, ParameterValue>> {
        info!("Saving parameter state: {}", name);
        Ok(self.current_values.clone())
    }
    
    /// Load parameter state
    pub fn load_state(&mut self, state: HashMap<String, ParameterValue>) -> KwaversResult<()> {
        info!("Loading parameter state with {} parameters", state.len());
        
        for (name, value) in state {
            if self.parameters.contains_key(&name) {
                self.update_parameter(&name, value)?;
            }
        }
        
        Ok(())
    }
    
    /// Render interactive controls UI
    #[cfg(feature = "advanced-visualization")]
    pub fn render_ui(&mut self, ctx: &Context) -> KwaversResult<()> {
        let mut show_controls = self.ui_state.show_controls;
        Window::new("Simulation Controls")
            .open(&mut show_controls)
            .default_size(Vec2::new(400.0, 600.0))
            .show(ctx, |ui| {
                self.render_control_panel(ui)
            });
        self.ui_state.show_controls = show_controls;
        
        Ok(())
    }
    
    #[cfg(feature = "advanced-visualization")]
    fn render_control_panel(&mut self, ui: &mut Ui) {
        // Group selection
        ui.horizontal(|ui| {
            ui.label("Group:");
            egui::ComboBox::from_label("")
                .selected_text(&self.ui_state.selected_group)
                .show_ui(ui, |ui| {
                    for group in &self.ui_state.parameter_groups.clone() {
                        ui.selectable_value(&mut self.ui_state.selected_group, group.clone(), group);
                    }
                });
        });
        
        ui.separator();
        
        // Search filter
        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.ui_state.search_filter);
        });
        
        ui.separator();
        
        // Parameter controls
        egui::ScrollArea::vertical().show(ui, |ui| {
            let filtered_params: Vec<_> = self.parameters
                .iter()
                .filter(|(_, def)| {
                    def.group == self.ui_state.selected_group &&
                    (self.ui_state.search_filter.is_empty() || 
                     def.display_name.to_lowercase().contains(&self.ui_state.search_filter.to_lowercase()))
                })
                .map(|(name, def)| (name.clone(), def.clone()))
                .collect();
            
            for (name, definition) in filtered_params {
                self.render_parameter_control(ui, &name, &definition);
            }
        });
        
        ui.separator();
        
        // Control buttons
        ui.horizontal(|ui| {
            if ui.button("Reset All").clicked() {
                self.reset_all_parameters();
            }
            
            if ui.button("Undo Last").clicked() {
                let _ = self.undo_last_change();
            }
            
            ui.checkbox(&mut self.ui_state.show_advanced, "Show Advanced");
        });
    }
    
    #[cfg(feature = "advanced-visualization")]
    fn render_parameter_control(&mut self, ui: &mut Ui, name: &str, definition: &ParameterDefinition) {
        ui.group(|ui| {
            ui.label(&definition.display_name);
            ui.small(&definition.description);
            
            // Get current value to avoid borrowing issues
            if let Some(current_value) = self.current_values.get(name).cloned() {
                match (&definition.parameter_type, current_value) {
                    (ParameterType::Float { min, max, step }, ParameterValue::Float(mut value)) => {
                        let old_value = value;
                        ui.add(egui::Slider::new(&mut value, *min..=*max).step_by(*step));
                        if old_value != value {
                            let _ = self.update_parameter(name, ParameterValue::Float(value));
                        }
                    }
                    (ParameterType::Boolean, ParameterValue::Boolean(mut value)) => {
                        let old_value = value;
                        ui.checkbox(&mut value, "");
                        if old_value != value {
                            let _ = self.update_parameter(name, ParameterValue::Boolean(value));
                        }
                    }
                    (ParameterType::Vector3 { min, max, step }, ParameterValue::Vector3(mut values)) => {
                        let old_values = values;
                        let mut changed = false;
                        ui.horizontal(|ui| {
                            for (_, value) in values.iter_mut().enumerate() {
                                let old_value = *value;
                                ui.add(egui::DragValue::new(value)
                                    .clamp_range(*min..=*max)
                                    .speed(*step));
                                if old_value != *value {
                                    changed = true;
                                }
                            }
                        });
                        if changed {
                            let _ = self.update_parameter(name, ParameterValue::Vector3(values));
                        }
                    }
                    _ => {
                        ui.label("(Unsupported parameter type)");
                    }
                }
            }
        });
    }
    
    /// Reset all parameters to default values
    fn reset_all_parameters(&mut self) {
        info!("Resetting all parameters to default values");
        
        for (name, definition) in &self.parameters {
            self.current_values.insert(name.clone(), definition.default_value.clone());
        }
        
        self.parameter_history.clear();
    }
    
    /// Register default simulation parameters
    fn register_default_parameters(&mut self) -> KwaversResult<()> {
        // Simulation parameters
        self.register_parameter(ParameterDefinition {
            name: "frequency".to_string(),
            display_name: "Frequency (MHz)".to_string(),
            description: "Ultrasound frequency".to_string(),
            parameter_type: ParameterType::Float { min: 0.1, max: 50.0, step: 0.1 },
            default_value: ParameterValue::Float(1.0),
            group: "Simulation".to_string(),
            is_realtime: true,
        })?;
        
        self.register_parameter(ParameterDefinition {
            name: "amplitude".to_string(),
            display_name: "Amplitude (MPa)".to_string(),
            description: "Pressure amplitude".to_string(),
            parameter_type: ParameterType::Float { min: 0.0, max: 10.0, step: 0.1 },
            default_value: ParameterValue::Float(1.0),
            group: "Simulation".to_string(),
            is_realtime: true,
        })?;
        
        // Visualization parameters
        self.register_parameter(ParameterDefinition {
            name: "transparency".to_string(),
            display_name: "Transparency".to_string(),
            description: "Volume rendering transparency".to_string(),
            parameter_type: ParameterType::Float { min: 0.0, max: 1.0, step: 0.01 },
            default_value: ParameterValue::Float(0.5),
            group: "Visualization".to_string(),
            is_realtime: true,
        })?;
        
        self.register_parameter(ParameterDefinition {
            name: "iso_value".to_string(),
            display_name: "Isosurface Value".to_string(),
            description: "Threshold for isosurface extraction".to_string(),
            parameter_type: ParameterType::Float { min: 0.0, max: 1.0, step: 0.01 },
            default_value: ParameterValue::Float(0.5),
            group: "Visualization".to_string(),
            is_realtime: true,
        })?;
        
        self.register_parameter(ParameterDefinition {
            name: "enable_gpu_acceleration".to_string(),
            display_name: "GPU Acceleration".to_string(),
            description: "Enable GPU-accelerated rendering".to_string(),
            parameter_type: ParameterType::Boolean,
            default_value: ParameterValue::Boolean(true),
            group: "Advanced".to_string(),
            is_realtime: false,
        })?;
        
        Ok(())
    }
    
    /// Validate parameter value against type constraints
    fn validate_parameter_value(&self, param_type: &ParameterType, value: &ParameterValue) -> KwaversResult<()> {
        match (param_type, value) {
            (ParameterType::Float { min, max, .. }, ParameterValue::Float(v)) => {
                if *v < *min || *v > *max {
                    return Err(KwaversError::Visualization(
                        format!("Float value {} out of range [{}, {}]", v, min, max)
                    ));
                }
            }
            (ParameterType::Integer { min, max, .. }, ParameterValue::Integer(v)) => {
                if *v < *min || *v > *max {
                    return Err(KwaversError::Visualization(
                        format!("Integer value {} out of range [{}, {}]", v, min, max)
                    ));
                }
            }
            (ParameterType::Boolean, ParameterValue::Boolean(_)) => {
                // Always valid
            }
            (ParameterType::Vector3 { min, max, .. }, ParameterValue::Vector3(values)) => {
                for v in values {
                    if *v < *min || *v > *max {
                        return Err(KwaversError::Visualization(
                            format!("Vector3 component {} out of range [{}, {}]", v, min, max)
                        ));
                    }
                }
            }
            _ => {
                return Err(KwaversError::Visualization(
                    "Parameter type mismatch".to_string()
                ));
            }
        }
        
        Ok(())
    }
}