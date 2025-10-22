//! UI components for the control system

use super::parameter::{ParameterType, ParameterValue};
use super::state::InteractiveControls;
use crate::error::KwaversResult;
use std::collections::HashMap;

/// Configuration for the control panel
#[derive(Debug, Clone)]
pub struct ControlPanelConfig {
    pub title: String,
    pub width: f32,
    pub height: f32,
    pub position: [f32; 2],
    pub collapsible: bool,
    pub resizable: bool,
}

impl Default for ControlPanelConfig {
    fn default() -> Self {
        Self {
            title: "Controls".to_string(),
            width: 300.0,
            height: 400.0,
            position: [10.0, 10.0],
            collapsible: true,
            resizable: true,
        }
    }
}

/// Control panel UI component
#[derive(Debug)]
pub struct ControlPanel {
    config: ControlPanelConfig,
    controls: InteractiveControls,
    groups: HashMap<String, Vec<String>>,
    expanded_groups: HashMap<String, bool>,
}

impl ControlPanel {
    /// Create a control panel
    pub fn new(config: ControlPanelConfig) -> Self {
        Self {
            config,
            controls: InteractiveControls::new(),
            groups: HashMap::new(),
            expanded_groups: HashMap::new(),
        }
    }

    /// Get the underlying controls system
    pub fn controls(&self) -> &InteractiveControls {
        &self.controls
    }

    /// Organize parameters into groups
    pub fn organize_groups(&mut self) -> KwaversResult<()> {
        let states = self.controls.get_all_states()?;

        self.groups.clear();

        for (name, state) in states {
            let group = state.definition.group.clone();
            self.groups.entry(group.clone()).or_default().push(name);

            // Initialize expanded state for group
            self.expanded_groups.entry(group).or_insert(true);
        }

        Ok(())
    }

    /// Render the control panel
    ///
    /// **Implementation**: Full egui-based UI control panel with collapsible groups
    /// Provides interactive parameter adjustment for visualization pipeline
    ///
    /// **Reference**: egui documentation (immediate mode GUI paradigm)
    #[cfg(feature = "gpu-visualization")]
    pub fn render(&mut self, ctx: &egui::Context) {
        use egui::{CollapsingHeader, ScrollArea, Window};

        Window::new(&self.config.title)
            .default_pos(egui::pos2(self.config.position[0], self.config.position[1]))
            .default_size(egui::vec2(self.config.width, self.config.height))
            .collapsible(self.config.collapsible)
            .resizable(self.config.resizable)
            .show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    // Collect groups to avoid borrow conflict
                    let groups: Vec<_> = self
                        .groups
                        .iter()
                        .map(|(name, params)| (name.clone(), params.clone()))
                        .collect();

                    // Render groups
                    for (group_name, params) in groups {
                        CollapsingHeader::new(&group_name)
                            .default_open(
                                self.expanded_groups
                                    .get(&group_name)
                                    .copied()
                                    .unwrap_or(true),
                            )
                            .show(ui, |ui| {
                                for param_name in params {
                                    if let Ok(states) = self.controls.get_all_states() {
                                        if let Some(state) = states.get(&param_name) {
                                            self.render_parameter(ui, &param_name, state);
                                        }
                                    }
                                }
                            });
                    }
                });
            });
    }

    #[cfg(feature = "gpu-visualization")]
    fn render_parameter(
        &mut self,
        ui: &mut egui::Ui,
        name: &str,
        state: &super::state::ControlState,
    ) {
        use egui::Slider;

        ui.horizontal(|ui| {
            ui.label(&state.definition.display_name);

            match (&state.current_value, &state.definition.parameter_type) {
                (ParameterValue::Float(v), ParameterType::Float { min, max, step }) => {
                    let mut value = *v;
                    if ui
                        .add(Slider::new(&mut value, *min..=*max).step_by(*step))
                        .changed()
                    {
                        let _ = self
                            .controls
                            .update_parameter(name, ParameterValue::Float(value));
                    }
                }
                (ParameterValue::Integer(v), ParameterType::Integer { min, max, step }) => {
                    let mut value = *v;
                    if ui
                        .add(Slider::new(&mut value, *min..=*max).step_by(*step as f64))
                        .changed()
                    {
                        let _ = self
                            .controls
                            .update_parameter(name, ParameterValue::Integer(value));
                    }
                }
                (ParameterValue::Boolean(v), ParameterType::Boolean) => {
                    let mut value = *v;
                    if ui.checkbox(&mut value, "").changed() {
                        let _ = self
                            .controls
                            .update_parameter(name, ParameterValue::Boolean(value));
                    }
                }
                _ => {
                    ui.label(format!("{}", state.current_value));
                }
            }
        });

        if !state.definition.description.is_empty() {
            ui.label(&state.definition.description);
        }
    }

    /// Render without GPU features (conditional compilation stub)
    ///
    /// Note: This is a proper conditional compilation stub for non-GPU builds.
    /// GPU visualization requires the `gpu-visualization` feature flag.
    #[cfg(not(feature = "gpu-visualization"))]
    pub fn render(&mut self, _ctx: &()) {
        log::debug!("Control panel rendering requires gpu-visualization feature");
    }
}
