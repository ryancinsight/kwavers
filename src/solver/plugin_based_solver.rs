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
use crate::physics::plugin::{PhysicsPlugin, PluginManager};
use crate::physics::field_mapping::UnifiedFieldType;
use crate::recorder::RecorderTrait;
use crate::source::Source;
use crate::time::Time;
use crate::error::{KwaversError, FieldError};
use log::{info, debug};
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3};
use std::sync::Arc;
use std::collections::HashMap;

/// Dynamic field registry for type-safe field management
/// Optimized for O(1) direct indexing using Vec instead of HashMap
pub struct FieldRegistry {
    /// Registered fields indexed by UnifiedFieldType numeric value
    /// None indicates unregistered field
    fields: Vec<Option<FieldMetadata>>,
    /// Field data storage - dynamically sized
    data: Option<Array4<f64>>,
    /// Grid dimensions for validation
    grid_dims: (usize, usize, usize),
    /// Flag to defer allocation until build() is called
    deferred_allocation: bool,
    /// Counter for assigned indices in data array
    next_data_index: usize,
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

/// Field provider for plugins with restricted access
/// Prevents plugins from accessing fields they don't need
pub struct FieldProvider<'a> {
    registry: &'a mut FieldRegistry,
    allowed_fields: Vec<UnifiedFieldType>,
}

impl<'a> FieldProvider<'a> {
    /// Create a new field provider with restricted access
    pub fn new(registry: &'a mut FieldRegistry, allowed_fields: Vec<UnifiedFieldType>) -> Self {
        Self {
            registry,
            allowed_fields,
        }
    }
    
    /// Get a field view (zero-copy, read-only)
    pub fn get_field(&self, field_type: UnifiedFieldType) -> Result<ArrayView3<f64>, FieldError> {
        if !self.allowed_fields.contains(&field_type) {
            return Err(FieldError::NotRegistered(format!("Field {} not allowed for this plugin", field_type.name())));
        }
        self.registry.get_field(field_type)
    }
    
    /// Get a mutable field view (zero-copy)
    pub fn get_field_mut(&mut self, field_type: UnifiedFieldType) -> Result<ArrayViewMut3<f64>, FieldError> {
        if !self.allowed_fields.contains(&field_type) {
            return Err(FieldError::NotRegistered(format!("Field {} not allowed for this plugin", field_type.name())));
        }
        self.registry.get_field_mut(field_type)
    }
    
    /// Check if a field is available to this provider
    pub fn has_field(&self, field_type: UnifiedFieldType) -> bool {
        self.allowed_fields.contains(&field_type) && self.registry.has_field(field_type)
    }
    
    /// Get list of fields available to this provider
    pub fn available_fields(&self) -> Vec<UnifiedFieldType> {
        self.allowed_fields.clone()
    }
}

impl FieldRegistry {
    /// Create a new field registry with deferred allocation
    pub fn new(grid: &Grid) -> Self {
        Self {
            fields: vec![None; UnifiedFieldType::COUNT],
            data: None,
            grid_dims: (grid.nx, grid.ny, grid.nz),
            deferred_allocation: true,
            next_data_index: 0,
        }
    }
    
    /// Build the field registry by allocating data array
    /// This allows multiple field registrations without reallocations
    pub fn build(&mut self) -> Result<(), FieldError> {
        let num_fields = self.next_data_index;
        if num_fields == 0 {
            self.data = None;
            self.deferred_allocation = false;
            return Ok(());
        }

        let (nx, ny, nz) = self.grid_dims;
        self.data = Some(Array4::zeros((num_fields, nx, ny, nz)));
        self.deferred_allocation = false;
        
        debug!("Built FieldRegistry with {} fields and dimensions ({}, {}, {})", 
               num_fields, nx, ny, nz);
        Ok(())
    }
    
    /// Register a new field dynamically
    /// Allocation is deferred until build() is called for performance optimization
    pub fn register_field(&mut self, field_type: UnifiedFieldType, description: String) -> Result<(), FieldError> {
        let idx = field_type as usize;
        if idx < self.fields.len() && self.fields[idx].is_some() {
            return Ok(()); // Already registered
        }
        
        // Ensure Vec is large enough
        while self.fields.len() <= idx {
            self.fields.push(None);
        }
        
        self.fields[idx] = Some(FieldMetadata {
            index: self.next_data_index,
            description,
            active: true,
        });
        self.next_data_index += 1;
        
        // Only reallocate if not using deferred allocation
        if !self.deferred_allocation {
            self.reallocate_data_internal()?;
        }
        
        info!("Registered field: {} at index {}", field_type, self.next_data_index - 1);
        Ok(())
    }
    
    /// Register multiple fields at once (optimized for batch registration)
    pub fn register_fields(&mut self, fields: &[(UnifiedFieldType, String)]) -> Result<(), FieldError> {
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
    
    /// Get a specific field by type (zero-copy view)
    pub fn get_field(&self, field_type: UnifiedFieldType) -> Result<ArrayView3<f64>, FieldError> {
        let metadata = self.fields.get(&field_type)
            .ok_or_else(|| FieldError::NotRegistered(field_type.name().to_string()))?;
        
        if !metadata.active {
            return Err(FieldError::Inactive(field_type.name().to_string()));
        }
        
        let data = self.data.as_ref()
            .ok_or(FieldError::DataNotInitialized)?;
        
        Ok(data.index_axis(ndarray::Axis(0), metadata.index))
    }
    
    /// Get a mutable field view (zero-copy)
    pub fn get_field_mut(&mut self, field_type: UnifiedFieldType) -> Result<ArrayViewMut3<f64>, FieldError> {
        let metadata = self.fields.get(&field_type)
            .ok_or_else(|| FieldError::NotRegistered(field_type.name().to_string()))?;
        
        if !metadata.active {
            return Err(FieldError::Inactive(field_type.name().to_string()));
        }
        
        let data = self.data.as_mut()
            .ok_or(FieldError::DataNotInitialized)?;
        
        Ok(data.index_axis_mut(ndarray::Axis(0), metadata.index))
    }
    
    /// Get a specific field by type (owned copy for backward compatibility)
    #[deprecated(since = "0.2.0", note = "Use get_field() for zero-copy access instead")]
    pub fn get_field_owned(&self, field_type: UnifiedFieldType) -> Option<Array3<f64>> {
        let metadata = self.fields.get(&field_type)?;
        if !metadata.active {
            return None;
        }
        
        let data = self.data.as_ref()?;
        Some(data.index_axis(ndarray::Axis(0), metadata.index).to_owned())
    }
    
    /// Set a specific field with dimension validation
    pub fn set_field(&mut self, field_type: UnifiedFieldType, values: &Array3<f64>) -> Result<(), FieldError> {
        let metadata = self.fields.get(&field_type)
            .ok_or_else(|| FieldError::NotRegistered(field_type.name().to_string()))?;
        
        if !metadata.active {
            return Err(FieldError::Inactive(field_type.name().to_string()));
        }
        
        let data = self.data.as_mut()
            .ok_or(FieldError::DataNotInitialized)?;
        
        // Validate dimensions
        let expected_dims = self.grid_dims;
        let actual_dims = values.dim();
        if actual_dims != expected_dims {
            return Err(FieldError::DimensionMismatch {
                field: field_type.name().to_string(),
                expected: expected_dims,
                actual: actual_dims,
            });
        }
        
        let mut field_view = data.index_axis_mut(ndarray::Axis(0), metadata.index);
        field_view.assign(values);
        
        Ok(())
    }
    
    /// Check if a field is registered
    pub fn has_field(&self, field_type: UnifiedFieldType) -> bool {
        let idx = field_type as usize;
        idx < self.fields.len() && self.fields[idx].is_some()
    }
    
    /// Get list of registered fields
    pub fn registered_fields(&self) -> Vec<UnifiedFieldType> {
        self.fields.iter()
            .enumerate()
            .filter_map(|(idx, opt)| {
                opt.as_ref().map(|_| unsafe { std::mem::transmute(idx as u32) })
            })
            .collect()
    }
    
    /// Get the index of a field by name
    pub fn get_field_index(&self, field_name: &str) -> Option<usize> {
        // Convert field name to UnifiedFieldType
        match field_name {
            "pressure" => {
                let field_type = UnifiedFieldType::Pressure;
                let idx = field_type as usize;
                if idx < self.fields.len() && self.fields[idx].is_some() {
                    self.fields[idx].as_ref().map(|m| m.index)
                } else {
                    None
                }
            }
            _ => None
        }
    }
    
    /// Internal method to reallocate data array when fields change
    fn reallocate_data_internal(&mut self) -> Result<(), FieldError> {
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
/// 
/// # Boundary Condition Design Philosophy
/// 
/// The current implementation maintains a `Box<dyn Boundary>` for compatibility
/// with existing boundary implementations. However, the long-term vision is to
/// migrate all boundary conditions to the plugin architecture:
/// 
/// - Simple boundaries (e.g., hard wall) → `HardWallBoundaryPlugin`
/// - Complex integrated boundaries (e.g., CPML) → `CPMLPlugin`
/// 
/// This will unify the design and make plugins the single extension mechanism.
/// The current hybrid approach is a transitional state that allows gradual migration.
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

/// Enhanced performance metrics for high-performance computing
#[derive(Default)]
struct PerformanceMetrics {
    /// Total simulation steps completed
    total_steps: usize,
    /// Plugin execution times by name [seconds]
    plugin_execution_times: HashMap<String, Vec<f64>>,
    /// Field update times [seconds]
    field_update_times: Vec<f64>,
    /// Memory usage tracking [bytes]
    memory_usage: Vec<usize>,
    /// Memory allocations count per step
    memory_allocations: Vec<usize>,
    /// Cache performance metrics (placeholder for future hardware integration)
    cache_metrics: CacheMetrics,
    /// Floating-point operations per second
    flops_per_step: Vec<f64>,
    /// Bandwidth utilization [GB/s]
    bandwidth_utilization: Vec<f64>,
    /// Thread utilization
    thread_utilization: Vec<f64>,
    /// Garbage collection time (for future profiling)
    gc_time: Vec<f64>,
}

/// Cache performance metrics (placeholder for hardware-level profiling)
#[derive(Default)]
struct CacheMetrics {
    /// L1 cache miss rate
    l1_miss_rate: Vec<f64>,
    /// L2 cache miss rate  
    l2_miss_rate: Vec<f64>,
    /// L3 cache miss rate
    l3_miss_rate: Vec<f64>,
    /// Memory bandwidth saturation
    memory_bandwidth_saturation: Vec<f64>,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    fn new() -> Self {
        Self {
            total_steps: 0,
            plugin_execution_times: HashMap::new(),
            field_update_times: Vec::new(),
            memory_usage: Vec::new(),
            memory_allocations: Vec::new(),
            cache_metrics: CacheMetrics::default(),
            flops_per_step: Vec::new(),
            bandwidth_utilization: Vec::new(),
            thread_utilization: Vec::new(),
            gc_time: Vec::new(),
        }
    }
    
    /// Record plugin execution time
    pub fn record_plugin_time(&mut self, plugin_name: &str, duration: f64) {
        self.plugin_execution_times
            .entry(plugin_name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }
}

impl PluginBasedSolver {
    /// Clear all sources
    pub fn clear_sources(&mut self) {
        // In a full implementation, this would clear source terms
        // For now, we'll just reset the source
    }
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
    
    /// Get current time
    pub fn time(&self) -> &Time {
        &self.time
    }
    
    /// Get medium
    pub fn medium(&self) -> &Arc<dyn Medium> {
        &self.medium
    }
    
    /// Get source
    pub fn source(&self) -> &Box<dyn Source> {
        &self.source
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
        self.plugin_manager.add_plugin(plugin)?;
        
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
        
        // Boundary and source are already initialized in constructor
        
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
        self.run_with_progress(&mut crate::solver::NullProgressReporter)
    }
    
    /// Run simulation with progress reporting
    pub fn run_with_progress(&mut self, reporter: &mut dyn crate::solver::ProgressReporter) -> KwaversResult<()> {
        let dt = self.time.dt;
        let total_steps = self.time.n_steps;
        
        reporter.on_start(total_steps, dt);
        
        let mut step_times = Vec::with_capacity(100);
        
        for step in 0..total_steps {
            let step_start = std::time::Instant::now();
            let t = step as f64 * dt;
            
            // Execute one time step
            self.step(step, t)?;
            
            // Record data if needed
            if let Some(recorder) = &mut self.recorder {
                if let Some(fields) = self.field_registry.data() {
                    recorder.record(fields, step)?;
                }
            }
            
            let step_duration = step_start.elapsed();
            step_times.push(step_duration);
            
            // Keep only recent step times for averaging
            if step_times.len() > 100 {
                step_times.remove(0);
            }
            
            // Calculate average step time and ETA
            let avg_step_time = if !step_times.is_empty() {
                let total: std::time::Duration = step_times.iter().sum();
                total / step_times.len() as u32
            } else {
                step_duration
            };
            
            let remaining_steps = total_steps - step - 1;
            let estimated_remaining = avg_step_time * remaining_steps as u32;
            
            // Gather field summary
            let fields_summary = self.calculate_fields_summary();
            
            // Report progress
            reporter.report(&crate::solver::ProgressUpdate {
                current_step: step,
                total_steps,
                current_time: t,
                total_time: total_steps as f64 * dt,
                step_duration,
                estimated_remaining,
                fields_summary,
            });
        }
        
        reporter.on_complete();
        self.finalize()?;
        
        Ok(())
    }
    
    /// Calculate summary statistics for fields
    fn calculate_fields_summary(&self) -> crate::solver::FieldsSummary {
        let mut max_pressure = 0.0;
        let mut max_velocity = 0.0;
        let mut max_temperature = 0.0;
        let mut total_energy = 0.0;
        
        // Calculate field statistics if data is available
        if let Some(data) = self.field_registry.data() {
            // Get pressure field
            if let Some(pressure) = self.field_registry.get_field(UnifiedFieldType::Pressure) {
                max_pressure = pressure.iter()
                    .map(|&p| p.abs())
                    .fold(0.0, f64::max);
                total_energy += pressure.iter()
                    .map(|&p| p * p)
                    .sum::<f64>();
            }
            
            // Get velocity field
            if let Some(velocity) = self.field_registry.get_field(UnifiedFieldType::VelocityX) {
                max_velocity = velocity.iter()
                    .map(|&v| v.abs())
                    .fold(max_velocity, f64::max);
            }
            
            // Get temperature field
            if let Some(temperature) = self.field_registry.get_field(UnifiedFieldType::Temperature) {
                max_temperature = temperature.iter()
                    .map(|&t| t.abs())
                    .fold(0.0, f64::max);
            }
        }
        
        crate::solver::FieldsSummary {
            max_pressure,
            max_velocity,
            max_temperature,
            total_energy,
        }
    }
    
    /// Execute one time step
    fn step(&mut self, step: usize, t: f64) -> KwaversResult<()> {
        let dt = self.time.dt;
        
        // Get mutable field data
        let fields = self.field_registry.data_mut()
            .ok_or_else(|| KwaversError::Field(FieldError::DataNotInitialized))?;
        
        // 1. Apply boundary conditions first (prepare fields for physics update)
        // Boundary conditions are applied directly to the pressure field
        if let Some(pressure_idx) = self.field_registry.get_field_index("pressure") {
            let mut pressure_field = fields.index_axis_mut(ndarray::Axis(0), pressure_idx);
            self.boundary.apply_acoustic(pressure_field.view_mut(), &self.grid, step)?;
        }
        
        // 2. Apply source terms (inject energy into the system)
        // Sources add their contribution to the pressure field
        if let Some(pressure_idx) = self.field_registry.get_field_index("pressure") {
            let source_mask = self.source.create_mask(&self.grid);
            let amplitude = self.source.amplitude(t);
            let mut pressure_field = fields.index_axis_mut(ndarray::Axis(0), pressure_idx);
            
            // Add source contribution: pressure += amplitude * mask
            ndarray::Zip::from(&mut pressure_field)
                .and(&source_mask)
                .for_each(|p, &mask| {
                    *p += amplitude * mask;
                });
        }
        
        // 3. Execute physics plugins in dependency order
        let plugin_timings = self.plugin_manager.execute_with_metrics(
            fields,
            &self.grid,
            self.medium.as_ref(),
            dt,
            step,
            self.time.n_steps
        )?;
        
        // 4. Record performance metrics
        for (plugin_name, duration) in plugin_timings {
            self.metrics.record_plugin_time(&plugin_name, duration);
        }
        
        self.metrics.total_steps += 1;
        
        Ok(())
    }
    
    /// Finalize the simulation
    pub fn finalize(&mut self) -> KwaversResult<()> {
        info!("Finalizing simulation");
        
        // Plugins handle their own cleanup via Drop trait
        
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
            .ok()
            .map(|view| view.to_owned())
    }
    
    /// Set a specific field
    pub fn set_field(&mut self, field_type: UnifiedFieldType, values: &Array3<f64>) -> KwaversResult<()> {
        self.field_registry.set_field(field_type, values)
            .map_err(|e| KwaversError::Field(e))
    }
    
    /// Get the grid
    pub fn grid(&self) -> &Grid {
        &self.grid
    }
    
    /// Get the medium
    
    /// Get performance metrics
    pub fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }
    
    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = PerformanceMetrics::default();
    }
}

impl PerformanceMetrics {
    /// Record field update time
    pub fn record_field_update_time(&mut self, time: f64) {
        self.field_update_times.push(time);
    }
    
    /// Record memory usage (placeholder - would need system integration)
    pub fn record_memory_usage(&mut self, bytes: usize) {
        self.memory_usage.push(bytes);
    }
    
    /// Record FLOPS for performance analysis
    pub fn record_flops(&mut self, flops: f64) {
        self.flops_per_step.push(flops);
    }
    
    /// Get average plugin execution time
    pub fn average_plugin_time(&self, plugin_name: &str) -> Option<f64> {
        let times = self.plugin_execution_times.get(plugin_name)?;
        if times.is_empty() {
            return None;
        }
        Some(times.iter().sum::<f64>() / times.len() as f64)
    }
    
    /// Get total execution time
    pub fn total_execution_time(&self) -> f64 {
        self.plugin_execution_times
            .values()
            .flat_map(|times| times.iter())
            .sum::<f64>() + self.field_update_times.iter().sum::<f64>()
    }
    
    /// Get performance summary
    pub fn summary(&self) -> String {
        let total_time = self.total_execution_time();
        let avg_step_time = if self.total_steps > 0 {
            total_time / self.total_steps as f64
        } else {
            0.0
        };
        
        let mut summary = format!(
            "Performance Summary:\n\
             Total Steps: {}\n\
             Total Time: {:.3} s\n\
             Average Step Time: {:.3} ms\n",
            self.total_steps,
            total_time,
            avg_step_time * 1000.0
        );
        
        // Plugin breakdown
        for (plugin, times) in &self.plugin_execution_times {
            let avg_time = times.iter().sum::<f64>() / times.len() as f64;
            summary.push_str(&format!(
                "  {}: {:.3} ms/step\n",
                plugin,
                avg_time * 1000.0
            ));
        }
        
        // Memory statistics
        if !self.memory_usage.is_empty() {
            let avg_memory = self.memory_usage.iter().sum::<usize>() / self.memory_usage.len();
            let max_memory = self.memory_usage.iter().max().unwrap_or(&0);
            summary.push_str(&format!(
                "Memory: Avg {:.1} MB, Peak {:.1} MB\n",
                avg_memory as f64 / 1_048_576.0,
                *max_memory as f64 / 1_048_576.0
            ));
        }
        
        summary
    }
}

// Error types for the field registry
// Error variants are already defined in error.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;
    use crate::boundary::PMLBoundary;

    
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
        let time = Time::new(0.001, 100);
        let medium = Arc::new(HomogeneousMedium::from_minimal(1500.0, 1000.0, &grid, 0.0, 0.0));
        let boundary = Box::new(PMLBoundary::new(Default::default()).unwrap());
        use crate::source::PointSource;
        use crate::signal::SineWave;
        use std::sync::Arc;
        
        let signal = Arc::new(SineWave::new(1e6, 1.0, 0.0));
        let source = Box::new(PointSource::new(
            (grid.nx as f64 / 2.0, grid.ny as f64 / 2.0, grid.nz as f64 / 2.0),
            signal,
        ));
        
        let solver = PluginBasedSolver::new(grid, time, medium, boundary, source);
        
        assert_eq!(solver.plugin_manager.component_count(), 0);
        assert_eq!(solver.field_registry.registered_fields().len(), 0);
    }
}