//! Plugin-based solver implementation
//!
//! Orchestrates physics simulations using a plugin architecture.
//! Follows SOLID principles with clear separation of concerns.

use crate::boundary::Boundary;
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::plugin::{PhysicsPlugin, PluginManager};
use crate::physics::field_mapping::UnifiedFieldType;
use crate::recorder::RecorderTrait;
use crate::source::Source;
use crate::time::Time;
use log::{debug, info};
use std::sync::Arc;

use super::field_registry::FieldRegistry;
use super::performance::PerformanceMonitor;

/// Plugin-based solver for acoustic simulations
pub struct PluginBasedSolver {
    /// Simulation grid
    grid: Grid,
    /// Time configuration
    time: Time,
    /// Medium properties
    medium: Arc<dyn Medium>,
    /// Boundary conditions
    boundary: Box<dyn Boundary>,
    /// Acoustic sources
    sources: Vec<Box<dyn Source>>,
    /// Field registry for data management
    field_registry: FieldRegistry,
    /// Plugin manager for physics modules
    plugin_manager: PluginManager,
    /// Performance monitor
    performance: PerformanceMonitor,
    /// Optional recorder
    recorder: Option<Box<dyn RecorderTrait>>,
    /// Current time step
    current_step: usize,
}

impl PluginBasedSolver {
    /// Create a new plugin-based solver
    pub fn new(
        grid: Grid,
        time: Time,
        medium: Arc<dyn Medium>,
        boundary: Box<dyn Boundary>,
        source: Box<dyn Source>,
    ) -> Self {
        let mut sources = Vec::new();
        sources.push(source);
        
        Self {
            grid: grid.clone(),
            time,
            medium,
            boundary,
            sources,
            field_registry: FieldRegistry::new(&grid),
            plugin_manager: PluginManager::new(),
            performance: PerformanceMonitor::new(),
            recorder: None,
            current_step: 0,
        }
    }

    /// Add a physics plugin
    pub fn add_plugin(&mut self, plugin: Box<dyn PhysicsPlugin>) -> KwaversResult<()> {
        // Register required fields
        for field in plugin.required_fields() {
            self.field_registry.register_field(
                field,
                format!("{} field", field.name())
            )?;
        }
        
        self.plugin_manager.add_plugin(plugin)?;
        Ok(())
    }

    /// Add an acoustic source
    pub fn add_source(&mut self, source: Box<dyn Source>) {
        self.sources.push(source);
    }

    /// Set the recorder
    pub fn set_recorder(&mut self, recorder: Box<dyn RecorderTrait>) {
        self.recorder = Some(recorder);
    }

    /// Initialize the solver
    pub fn initialize(&mut self) -> KwaversResult<()> {
        info!("Initializing plugin-based solver");
        
        // Build field registry
        self.field_registry.build()?;
        
        // Initialize plugins
        self.plugin_manager.initialize(&self.grid, &*self.medium)?;
        
        // Boundary conditions don't need initialization in current implementation
        
        debug!("Solver initialized with {} plugins", self.plugin_manager.component_count());
        Ok(())
    }

    /// Run the simulation for a specified duration
    pub fn run_for_duration(&mut self, duration: f64) -> KwaversResult<()> {
        let steps = (duration / self.time.dt) as usize;
        self.run_for_steps(steps)
    }

    /// Run the simulation for a specified number of steps
    pub fn run_for_steps(&mut self, steps: usize) -> KwaversResult<()> {
        info!("Running simulation for {} steps", steps);
        
        for step in 0..steps {
            self.step()?;
            
            // Record if needed (every 10 steps for now)
            if step % 10 == 0 {
                if let Some(ref mut recorder) = self.recorder {
                    if let Some(data) = self.field_registry.data() {
                        recorder.record(data, step)?;
                    }
                }
            }
            
            // Log progress
            if step % 100 == 0 {
                debug!("Step {}/{}", step, steps);
            }
        }
        
        info!("Simulation completed");
        Ok(())
    }

    /// Perform a single time step
    pub fn step(&mut self) -> KwaversResult<()> {
        let t = self.current_step as f64 * self.time.dt;
        
        // Apply sources
        for source in &self.sources {
            if let Ok(mut pressure) = self.field_registry.get_field_mut(UnifiedFieldType::Pressure) {
                let mask = source.create_mask(&self.grid);
                let amplitude = source.amplitude(t);
                pressure.scaled_add(amplitude, &mask);
            }
        }
        
        // Execute plugins with direct field array access
        if let Some(fields_array) = self.field_registry.data_mut() {
            // The plugin manager needs mutable access to execute plugins
            // We need to temporarily extract it and put it back
            let mut plugin_manager = std::mem::replace(&mut self.plugin_manager, PluginManager::new());
            
            // Execute all plugins with the field array
            let result = plugin_manager.execute(
                fields_array,
                &self.grid,
                &**self.medium,  // Deref Arc<dyn Medium> to &dyn Medium
                self.time.dt,
                t
            );
            
            // Restore the plugin manager
            self.plugin_manager = plugin_manager;
            
            // Handle any errors from plugin execution
            result?;
        }
        
        // Apply boundary conditions
        if let Ok(pressure) = self.field_registry.get_field_mut(UnifiedFieldType::Pressure) {
            self.boundary.apply_acoustic(pressure, &self.grid, self.current_step);
        }
        
        self.current_step += 1;
        self.performance.next_iteration();
        
        Ok(())
    }



    /// Get performance metrics
    pub fn performance_report(&self) -> String {
        self.performance.report()
    }

    /// Get field registry reference
    pub fn field_registry(&self) -> &FieldRegistry {
        &self.field_registry
    }

    /// Get mutable field registry reference
    pub fn field_registry_mut(&mut self) -> &mut FieldRegistry {
        &mut self.field_registry
    }

    /// Get plugin manager reference
    pub fn plugin_manager(&self) -> &PluginManager {
        &self.plugin_manager
    }

    /// Get medium reference
    pub fn medium(&self) -> &Arc<dyn Medium> {
        &self.medium
    }

    /// Get time reference
    pub fn time(&self) -> &Time {
        &self.time
    }

    /// Clear all sources
    pub fn clear_sources(&mut self) {
        self.sources.clear();
    }

    /// Get field by type
    pub fn get_field(&self, field_type: UnifiedFieldType) -> Option<ndarray::Array3<f64>> {
        self.field_registry.get_field(field_type)
            .ok()
            .map(|view| view.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::homogeneous::HomogeneousMedium;
    use crate::boundary::PMLBoundary;
    use crate::source::PointSource;
    use crate::signal::SineWave;

    #[test]
    fn test_solver_creation() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0);
        let time = Time::new(0.001, 100);
        let medium = Arc::new(HomogeneousMedium::from_minimal(1500.0, 1000.0, &grid));
        let boundary = Box::new(PMLBoundary::new(Default::default()).unwrap());
        let signal = Arc::new(SineWave::new(1e6, 1.0, 0.0));
        let source = Box::new(PointSource::new(
            (5.0, 5.0, 5.0),
            signal,
        ));

        let solver = PluginBasedSolver::new(grid, time, medium, boundary, source);
        
        assert_eq!(solver.current_step, 0);
        assert_eq!(solver.plugin_manager.component_count(), 0);
    }
}