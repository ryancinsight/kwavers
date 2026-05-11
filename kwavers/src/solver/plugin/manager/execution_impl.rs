//! Plugin execution: `execute`, `execute_with_metrics`, and `add_plugin`.

use super::PluginManager;
use crate::core::error::{KwaversError, KwaversResult, PhysicsError, ValidationError};
use crate::domain::boundary::Boundary;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::plugin::{Plugin, PluginContext};
use crate::domain::source::Source;
use ndarray::Array4;
use std::time::Instant;

impl PluginManager {
    /// Add a plugin, checking for duplicate IDs and resolving dependency order.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if a duplicate plugin ID is detected.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn add_plugin(&mut self, plugin: Box<dyn Plugin>) -> KwaversResult<()> {
        let new_id = plugin.metadata().id.clone();
        for existing in &self.plugins {
            if existing.metadata().id == new_id {
                return Err(ValidationError::FieldValidation {
                    field: "plugin_id".to_owned(),
                    value: new_id,
                    constraint: "Plugin ID must be unique".to_owned(),
                }
                .into());
            }
        }
        self.plugins.push(plugin);
        self.resolve_dependencies()?;
        Ok(())
    }

    /// Execute all plugins for one time step in dependency order.
    /// # Errors
    /// - Returns [`KwaversError::Physics`] if a plugin index is out of bounds.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn execute(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        sources: &[Box<dyn Source>],
        boundary: &mut dyn Boundary,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        let mut context = PluginContext {
            extra_fields: &self.extra_fields,
            sources,
            boundary,
        };

        for &idx in &self.execution_order {
            if idx >= self.plugins.len() {
                return Err(KwaversError::Physics(PhysicsError::InvalidState {
                    field: "plugin_index".to_owned(),
                    value: idx.to_string(),
                    reason: format!(
                        "Index {} out of bounds for {} plugins",
                        idx,
                        self.plugins.len()
                    ),
                }));
            }
            self.plugins[idx].update(fields, grid, medium, dt, t, &mut context)?;
        }

        Ok(())
    }

    /// Execute plugins with per-plugin timing collection.
    /// # Errors
    /// - Returns [`KwaversError::Physics`] if a plugin index is out of bounds.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn execute_with_metrics(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        sources: &[Box<dyn Source>],
        boundary: &mut dyn Boundary,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        let start = Instant::now();

        let mut context = PluginContext {
            extra_fields: &self.extra_fields,
            sources,
            boundary,
        };

        for &idx in &self.execution_order {
            if idx >= self.plugins.len() {
                return Err(KwaversError::Physics(PhysicsError::InvalidState {
                    field: "plugin_index".to_owned(),
                    value: idx.to_string(),
                    reason: format!(
                        "Index {} out of bounds for {} plugins",
                        idx,
                        self.plugins.len()
                    ),
                }));
            }

            let plugin_start = Instant::now();
            let plugin = &mut self.plugins[idx];
            plugin.update(fields, grid, medium, dt, t, &mut context)?;
            let plugin_duration = plugin_start.elapsed();
            self.performance_metrics
                .record_plugin_execution(&plugin.metadata().id, plugin_duration);
        }

        self.performance_metrics.record_total_execution(start.elapsed());
        Ok(())
    }
}
