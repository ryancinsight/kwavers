//! Plugin-based solver architecture following SOLID principles
//!
//! This module provides a clean, extensible solver that:
//! - Follows Single Responsibility Principle (orchestration only)
//! - Applies Dependency Inversion (depends on abstractions)
//! - Is Open/Closed (extensible via plugins)
//! - Uses dynamic field registration (no static const indices)

use crate::grid::Grid;
use crate::KwaversResult;
use crate::boundary::Boundary;
use crate::medium::Medium;
use crate::physics::plugin::{PhysicsPlugin, PluginManager, PluginContext};
use crate::physics::field_mapping::{UnifiedFieldType, FieldAccessor, FieldAccessorMut};
use crate::recorder::RecorderTrait;
use crate::source::Source;
use crate::time::Time;
use crate::error::KwaversError;
use log::{info, debug, trace};
use ndarray::{Array3, Array4};
use std::sync::Arc;
use std::collections::{HashMap, HashSet};

/// Dynamic field registry for type-safe field management
pub struct FieldRegistry {
    /// Registered fields and their metadata
    fields: HashMap<UnifiedFieldType, FieldMetadata>,
    /// Field data storage - dynamically sized
    data: Option<Array4<f64>>,
    /// Grid dimensions for validation
    grid_dims: (usize, usize, usize),
}

#[derive(Clone)]
struct FieldMetadata {
    /// Index in the Array4
    index: usize,
    /// Field description
    description: String,
    /// Whether field is currently active
    active: bool,
}

impl FieldRegistry {
    /// Create a new field registry
    pub fn new(grid: &Grid) -> Self {
        Self {
            fields: HashMap::new(),
            data: None,
            grid_dims: (grid.nx, grid.ny, grid.nz),
        }
    }
    
    /// Register a new field dynamically
    pub fn register_field(&mut self, field_type: UnifiedFieldType, description: String) -> KwaversResult<()> {
        if self.fields.contains_key(&field_type) {
            return Ok(()); // Already registered
        }
        
        let index = self.fields.len();
        self.fields.insert(field_type, FieldMetadata {
            index,
            description,
            active: true,
        });
        
        // Reallocate data array if needed
        self.reallocate_data()?;
        
        info!("Registered field: {} at index {}", field_type, index);
        Ok(())
    }
    
    /// Register multiple fields at once
    pub fn register_fields(&mut self, fields: &[(UnifiedFieldType, String)]) -> KwaversResult<()> {
        for (field_type, description) in fields {
            self.register_field(*field_type, description.clone())?;
        }
        Ok(())
    }
    
    /// Get the current field data
    pub fn data(&self) -> Option<&Array4<f64>> {
        self.data.as_ref()
    }
    
    /// Get mutable field data
    pub fn data_mut(&mut self) -> Option<&mut Array4<f64>> {
        self.data.as_mut()
    }
    
    /// Get a specific field by type
    pub fn get_field(&self, field_type: UnifiedFieldType) -> Option<Array3<f64>> {
        let metadata = self.fields.get(&field_type)?;
        if !metadata.active {
            return None;
        }
        
        let data = self.data.as_ref()?;
        Some(data.index_axis(ndarray::Axis(0), metadata.index).to_owned())
    }
    
    /// Set a specific field
    pub fn set_field(&mut self, field_type: UnifiedFieldType, values: &Array3<f64>) -> KwaversResult<()> {
        let metadata = self.fields.get(&field_type)
            .ok_or_else(|| KwaversError::FieldNotRegistered(field_type.name().to_string()))?;
        
        if !metadata.active {
            return Err(KwaversError::FieldInactive(field_type.name().to_string()));
        }
        
        let data = self.data.as_mut()
            .ok_or_else(|| KwaversError::FieldDataNotInitialized)?;
        
        let mut field_view = data.index_axis_mut(ndarray::Axis(0), metadata.index);
        field_view.assign(values);
        
        Ok(())
    }
    
    /// Check if a field is registered
    pub fn has_field(&self, field_type: UnifiedFieldType) -> bool {
        self.fields.contains_key(&field_type)
    }
    
    /// Get list of registered fields
    pub fn registered_fields(&self) -> Vec<UnifiedFieldType> {
        self.fields.keys().cloned().collect()
    }
    
    /// Reallocate data array when fields change
    fn reallocate_data(&mut self) -> KwaversResult<()> {
        let num_fields = self.fields.len();
        if num_fields == 0 {
            self.data = None;
            return Ok(());
        }
        
        let (nx, ny, nz) = self.grid_dims;
        let mut new_data = Array4::zeros((num_fields, nx, ny, nz));
        
        // Copy existing data if present
        if let Some(old_data) = &self.data {
            let min_fields = old_data.shape()[0].min(num_fields);
            for i in 0..min_fields {
                new_data.index_axis_mut(ndarray::Axis(0), i)
                    .assign(&old_data.index_axis(ndarray::Axis(0), i));
            }
        }
        
        self.data = Some(new_data);
        Ok(())
    }
}

/// Plugin-based solver that follows SOLID principles
pub struct PluginBasedSolver {
    /// Grid configuration
    grid: Grid,
    /// Time management
    time: Time,
    /// Medium properties (shared, immutable reference)
    medium: Arc<dyn Medium>,
    /// Dynamic field registry
    field_registry: FieldRegistry,
    /// Plugin manager for physics
    plugin_manager: PluginManager,
    /// Boundary conditions (abstraction)
    boundary: Box<dyn Boundary>,
    /// Source terms (abstraction)
    source: Box<dyn Source>,
    /// Data recorder (abstraction)
    recorder: Option<Box<dyn RecorderTrait>>,
    /// Performance metrics
    metrics: PerformanceMetrics,
}

#[derive(Default)]
struct PerformanceMetrics {
    total_steps: usize,
    plugin_execution_times: HashMap<String, Vec<f64>>,
    field_update_times: Vec<f64>,
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
        let field_registry = FieldRegistry::new(&grid);
        let plugin_manager = PluginManager::new();
        
        Self {
            grid,
            time,
            medium,
            field_registry,
            plugin_manager,
            boundary,
            source,
            recorder: None,
            metrics: PerformanceMetrics::default(),
        }
    }
    
    /// Register a physics plugin
    pub fn register_plugin(&mut self, plugin: Box<dyn PhysicsPlugin>) -> KwaversResult<()> {
        // Register required fields
        for field_type in plugin.required_fields() {
            self.field_registry.register_field(
                field_type,
                format!("Required by {}", plugin.metadata().name)
            )?;
        }
        
        // Register provided fields
        for field_type in plugin.provided_fields() {
            self.field_registry.register_field(
                field_type,
                format!("Provided by {}", plugin.metadata().name)
            )?;
        }
        
        // Add to plugin manager
        self.plugin_manager.register(plugin)?;
        
        Ok(())
    }
    
    /// Set the data recorder
    pub fn set_recorder(&mut self, recorder: Box<dyn RecorderTrait>) {
        self.recorder = Some(recorder);
    }
    
    /// Initialize the simulation
    pub fn initialize(&mut self) -> KwaversResult<()> {
        info!("Initializing plugin-based solver");
        
        // Initialize all plugins
        self.plugin_manager.initialize_all(&self.grid, self.medium.as_ref())?;
        
        // Initialize boundary conditions
        self.boundary.initialize(&self.grid)?;
        
        // Initialize source
        self.source.initialize(&self.grid)?;
        
        // Initialize recorder if present
        if let Some(recorder) = &mut self.recorder {
            recorder.initialize(&self.grid)?;
        }
        
        info!("Solver initialized with {} plugins and {} fields",
              self.plugin_manager.component_count(),
              self.field_registry.registered_fields().len());
        
        Ok(())
    }
    
    /// Run the simulation
    pub fn run(&mut self) -> KwaversResult<()> {
        info!("Starting simulation for {} steps", self.time.n_steps);
        
        let dt = self.time.dt;
        let total_steps = self.time.n_steps;
        
        for step in 0..total_steps {
            let t = step as f64 * dt;
            
            // Update time
            self.time.update();
            
            // Execute one time step
            self.step(step, t)?;
            
            // Record data if needed
            if let Some(recorder) = &mut self.recorder {
                if let Some(fields) = self.field_registry.data() {
                    recorder.record(fields, step)?;
                }
            }
            
            // Log progress
            if step % 100 == 0 {
                debug!("Completed step {}/{}", step, total_steps);
            }
        }
        
        info!("Simulation completed successfully");
        self.finalize()?;
        
        Ok(())
    }
    
    /// Execute one time step
    fn step(&mut self, step: usize, t: f64) -> KwaversResult<()> {
        let dt = self.time.dt;
        
        // Get mutable field data
        let fields = self.field_registry.data_mut()
            .ok_or_else(|| KwaversError::FieldDataNotInitialized)?;
        
        // Apply source terms
        self.source.apply(fields, &self.grid, t)?;
        
        // Execute plugins in dependency order
        self.plugin_manager.execute(
            fields,
            &self.grid,
            self.medium.as_ref(),
            dt,
            step,
            self.time.n_steps,
            self.time.frequency
        )?;
        
        // Apply boundary conditions
        self.boundary.apply(fields, &self.grid)?;
        
        self.metrics.total_steps += 1;
        
        Ok(())
    }
    
    /// Finalize the simulation
    pub fn finalize(&mut self) -> KwaversResult<()> {
        info!("Finalizing simulation");
        
        // Finalize all plugins
        self.plugin_manager.finalize_all()?;
        
        // Finalize recorder
        if let Some(recorder) = &mut self.recorder {
            recorder.finalize()?;
        }
        
        // Report metrics
        self.report_metrics();
        
        Ok(())
    }
    
    /// Report performance metrics
    fn report_metrics(&self) {
        info!("=== Simulation Metrics ===");
        info!("Total steps: {}", self.metrics.total_steps);
        info!("Registered fields: {:?}", self.field_registry.registered_fields());
        info!("Active plugins: {}", self.plugin_manager.component_count());
    }
    
    /// Get a reference to a specific field
    pub fn get_field(&self, field_type: UnifiedFieldType) -> Option<Array3<f64>> {
        self.field_registry.get_field(field_type)
    }
    
    /// Set a specific field
    pub fn set_field(&mut self, field_type: UnifiedFieldType, values: &Array3<f64>) -> KwaversResult<()> {
        self.field_registry.set_field(field_type, values)
    }
    
    /// Get the grid
    pub fn grid(&self) -> &Grid {
        &self.grid
    }
    
    /// Get the medium
    pub fn medium(&self) -> &Arc<dyn Medium> {
        &self.medium
    }
}

// Error types for the field registry
impl KwaversError {
    fn field_not_registered(field: String) -> Self {
        Self::Configuration(format!("Field '{}' is not registered", field))
    }
    
    fn field_inactive(field: String) -> Self {
        Self::Configuration(format!("Field '{}' is inactive", field))
    }
    
    fn field_data_not_initialized() -> Self {
        Self::Configuration("Field data array not initialized".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;
    use crate::boundary::PMLBoundary;
    use crate::source::GaussianSource;
    
    #[test]
    fn test_field_registry() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0);
        let mut registry = FieldRegistry::new(&grid);
        
        // Register fields
        registry.register_field(UnifiedFieldType::Pressure, "Acoustic pressure".to_string()).unwrap();
        registry.register_field(UnifiedFieldType::Temperature, "Temperature field".to_string()).unwrap();
        
        assert!(registry.has_field(UnifiedFieldType::Pressure));
        assert!(registry.has_field(UnifiedFieldType::Temperature));
        assert!(!registry.has_field(UnifiedFieldType::Density));
        
        // Check data allocation
        assert!(registry.data().is_some());
        assert_eq!(registry.data().unwrap().shape(), &[2, 10, 10, 10]);
    }
    
    #[test]
    fn test_plugin_based_solver_creation() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0);
        let time = Time::new(0.001, 100, 1e6);
        let medium = Arc::new(HomogeneousMedium::new(1500.0, 1000.0, &grid, 0.0, 0.0));
        let boundary = Box::new(PMLBoundary::new(Default::default()).unwrap());
        let source = Box::new(GaussianSource::new(
            [grid.nx / 2, grid.ny / 2, grid.nz / 2],
            1.0,
            1e6,
            0.0,
        ));
        
        let solver = PluginBasedSolver::new(grid, time, medium, boundary, source);
        
        assert_eq!(solver.plugin_manager.component_count(), 0);
        assert_eq!(solver.field_registry.registered_fields().len(), 0);
    }
}