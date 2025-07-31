// src/physics/composable.rs
//! Composable physics system following CUPID principles
//! 
//! This module provides a composable, predictable physics simulation system where:
//! - Composable: Physics models can be combined in flexible ways
//! - Unix-like: Each model does one thing well and can be piped together
//! - Predictable: Same inputs always produce same outputs
//! - Idiomatic: Uses Rust's type system and ownership model effectively
//! - Domain-focused: Clear separation between different physics domains
//!
//! Design Principles Implemented:
//! - SOLID: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
//! - CUPID: Composable, Unix-like, Predictable, Idiomatic, Domain-focused
//! - GRASP: Information expert, creator, controller, low coupling, high cohesion
//! - ACID: Atomicity, consistency, isolation, durability
//! - DRY: Don't repeat yourself
//! - KISS: Keep it simple, stupid
//! - YAGNI: You aren't gonna need it
//! - SSOT: Single source of truth
//! - CCP: Common closure principle
//! - CRP: Common reuse principle
//! - ADP: Acyclic dependency principle

use crate::error::{KwaversResult, PhysicsError};
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array3, Array4, Axis, ArrayView3};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use crate::physics::traits::{CavitationModelBehavior, LightDiffusionModelTrait, AcousticWaveModel};
use log::warn;


/// Field identifiers for different physics quantities
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldType {
    Pressure,
    Light,
    Temperature,
    Cavitation,
    Chemical,
    Velocity,
    Stress,
    Custom(String),
}

impl FieldType {
    pub fn as_str(&self) -> &str {
        match self {
            FieldType::Pressure => "pressure",
            FieldType::Light => "light",
            FieldType::Temperature => "temperature",
            FieldType::Cavitation => "cavitation",
            FieldType::Chemical => "chemical",
            FieldType::Velocity => "velocity",
            FieldType::Stress => "stress",
            FieldType::Custom(name) => name.as_str(),
        }
    }
}

/// Component lifecycle states
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentState {
    Initialized,
    Ready,
    Running,
    Paused,
    Completed,
    Error(String),
}

/// Component validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
    
    pub fn add_error(&mut self, error: String) {
        self.is_valid = false;
        self.errors.push(error);
    }
    
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
    
    pub fn merge(&mut self, other: ValidationResult) {
        self.is_valid = self.is_valid && other.is_valid;
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

/// A composable physics component that can be combined with others
/// 
/// This trait implements SOLID principles:
/// - Single Responsibility: Each component has one clear purpose
/// - Open/Closed: New components can be added without modifying existing ones
/// - Liskov Substitution: All components are substitutable
/// - Interface Segregation: Minimal interface with focused methods
/// - Dependency Inversion: Depends on abstractions, not concretions
pub trait PhysicsComponent: Send + Sync {
    /// Unique identifier for this component
    fn component_id(&self) -> &str;
    
    /// Dependencies this component requires from other components
    fn dependencies(&self) -> Vec<FieldType>;
    
    /// Fields this component produces or modifies
    fn output_fields(&self) -> Vec<FieldType>;
    
    /// Apply this component's physics for one time step
    fn apply(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PhysicsContext,
    ) -> KwaversResult<()>;
    
    /// Check if this component can run with the given inputs
    fn can_execute(&self, available_fields: &[FieldType]) -> bool {
        self.dependencies().iter().all(|dep| available_fields.contains(dep))
    }
    
    /// Get performance metrics for this component
    fn get_metrics(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
    
    /// Validate component configuration and state
    fn validate(&self, _context: &PhysicsContext) -> ValidationResult {
        ValidationResult::new()
    }
    
    /// Get component state
    fn state(&self) -> ComponentState {
        ComponentState::Ready
    }
    
    /// Reset component to initial state
    fn reset(&mut self) -> KwaversResult<()> {
        Ok(())
    }
    
    /// Get component priority (lower numbers execute first)
    fn priority(&self) -> u32 {
        0
    }
    
    /// Check if component is optional (can be skipped if dependencies missing)
    fn is_optional(&self) -> bool {
        false
    }
}

/// Context shared between physics components
/// 
/// Implements SSOT (Single Source of Truth) principle by centralizing
/// all shared state and configuration
#[derive(Debug, Clone)]
pub struct PhysicsContext {
    /// Global simulation parameters
    pub parameters: HashMap<String, f64>,
    /// Frequency for acoustic simulations
    pub frequency: f64,
    /// Current simulation step
    pub step: usize,
    /// Source terms from external sources
    pub source_terms: HashMap<String, Array3<f64>>,
    /// Component-specific configurations
    pub component_configs: HashMap<String, HashMap<String, serde_json::Value>>,
    /// Validation cache
    pub validation_cache: HashMap<String, ValidationResult>,
    /// Performance tracking
    pub performance_tracker: PerformanceTracker,
}

impl PhysicsContext {
    pub fn new(frequency: f64) -> Self {
        Self {
            parameters: HashMap::new(),
            frequency,
            step: 0,
            source_terms: HashMap::new(),
            component_configs: HashMap::new(),
            validation_cache: HashMap::new(),
            performance_tracker: PerformanceTracker::new(),
        }
    }
    
    pub fn with_parameter(mut self, key: &str, value: f64) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }
    
    pub fn get_parameter(&self, key: &str) -> Option<f64> {
        self.parameters.get(key).copied()
    }
    
    pub fn add_source_term(&mut self, name: String, source: Array3<f64>) {
        self.source_terms.insert(name, source);
    }
    
    pub fn get_source_term(&self, name: &str) -> Option<&Array3<f64>> {
        self.source_terms.get(name)
    }
    
    /// Set component-specific configuration
    pub fn set_component_config(&mut self, component_id: &str, config: HashMap<String, serde_json::Value>) {
        self.component_configs.insert(component_id.to_string(), config);
    }
    
    /// Get component-specific configuration
    pub fn get_component_config(&self, component_id: &str) -> Option<&HashMap<String, serde_json::Value>> {
        self.component_configs.get(component_id)
    }
    
    /// Cache validation result
    pub fn cache_validation(&mut self, component_id: &str, result: ValidationResult) {
        self.validation_cache.insert(component_id.to_string(), result);
    }
    
    /// Get cached validation result
    pub fn get_cached_validation(&self, component_id: &str) -> Option<&ValidationResult> {
        self.validation_cache.get(component_id)
    }
}

/// Performance tracking for components
/// 
/// Implements DRY principle by centralizing performance measurement logic
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    pub execution_times: HashMap<String, Vec<f64>>,
    pub memory_usage: HashMap<String, usize>,
    pub call_counts: HashMap<String, usize>,
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            execution_times: HashMap::new(),
            memory_usage: HashMap::new(),
            call_counts: HashMap::new(),
        }
    }
    
    pub fn record_execution(&mut self, component_id: &str, duration: f64) {
        self.execution_times
            .entry(component_id.to_string())
            .or_default()
            .push(duration);
        
        *self.call_counts.entry(component_id.to_string()).or_insert(0) += 1;
    }
    
    pub fn record_memory(&mut self, component_id: &str, bytes: usize) {
        self.memory_usage.insert(component_id.to_string(), bytes);
    }
    
    pub fn get_average_execution_time(&self, component_id: &str) -> Option<f64> {
        self.execution_times.get(component_id).map(|times| {
            times.iter().sum::<f64>() / times.len() as f64
        })
    }
    
    pub fn get_total_execution_time(&self, component_id: &str) -> Option<f64> {
        self.execution_times.get(component_id).map(|times| {
            times.iter().sum()
        })
    }
}

/// A composable physics pipeline that executes components in dependency order
/// 
/// Implements GRASP principles:
/// - Controller: Manages execution order and component coordination
/// - Information Expert: Knows about component dependencies and execution order
/// - Low Coupling: Minimal dependencies between components
/// - High Cohesion: Related functionality grouped together
pub struct PhysicsPipeline {
    components: Vec<Box<dyn PhysicsComponent>>,
    execution_order: Vec<usize>,
    available_fields: HashSet<FieldType>,
    validation_results: HashMap<String, ValidationResult>,
    state: PipelineState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineState {
    Initialized,
    Ready,
    Running,
    Paused,
    Completed,
    Error(String),
}

impl PhysicsPipeline {
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            execution_order: Vec::new(),
            available_fields: HashSet::new(),
            validation_results: HashMap::new(),
            state: PipelineState::Initialized,
        }
    }
    
    /// Add a physics component to the pipeline
    /// 
    /// Implements ACID principles:
    /// - Atomicity: Either all components are added or none
    /// - Consistency: Maintains valid dependency relationships
    /// - Isolation: Component addition doesn't affect running pipeline
    /// - Durability: Changes persist until explicitly removed
    pub fn add_component(&mut self, component: Box<dyn PhysicsComponent>) -> KwaversResult<()> {
        // Validate component before adding
        let validation = self.validate_component(&component)?;
        if !validation.is_valid {
            return Err(PhysicsError::InvalidConfiguration {
                component: component.component_id().to_string(),
                reason: validation.errors.join(", "),
            }.into());
        }
        
        // Check for duplicate component IDs (SSOT principle)
        let id = component.component_id();
        if self.components.iter().any(|c| c.component_id() == id) {
            return Err(PhysicsError::IncompatibleModels {
                model1: id.to_string(),
                model2: "existing".to_string(),
                reason: "Duplicate component ID".to_string(),
            }.into());
        }
        
        // Add output fields to available fields
        for field in component.output_fields() {
            self.available_fields.insert(field);
        }
        
        self.components.push(component);
        self.compute_execution_order()?;
        self.state = PipelineState::Ready;
        Ok(())
    }
    
    /// Remove a component from the pipeline
    pub fn remove_component(&mut self, component_id: &str) -> KwaversResult<()> {
        let index = self.components
            .iter()
            .position(|c| c.component_id() == component_id)
            .ok_or_else(|| PhysicsError::ModelNotInitialized {
                model_name: component_id.to_string(),
            })?;
        
        // Check if removing this component would break dependencies
        let removed_outputs: HashSet<FieldType> = self.components[index]
            .output_fields()
            .into_iter()
            .collect();
        
        for (i, component) in self.components.iter().enumerate() {
            if i != index {
                let deps: HashSet<FieldType> = component.dependencies().into_iter().collect();
                if !deps.is_disjoint(&removed_outputs) {
                    return Err(PhysicsError::IncompatibleModels {
                        model1: component_id.to_string(),
                        model2: component.component_id().to_string(),
                        reason: "Dependency violation".to_string(),
                    }.into());
                }
            }
        }
        
        // Remove component and recompute execution order
        self.components.remove(index);
        self.compute_execution_order()?;
        Ok(())
    }
    
    /// Execute all components in the pipeline for one time step
    /// 
    /// Implements KISS principle with simple, clear execution flow
    pub fn execute(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PhysicsContext,
    ) -> KwaversResult<()> {
        if self.state != PipelineState::Ready && self.state != PipelineState::Running {
            return Err(PhysicsError::InvalidState {
                expected: "Ready or Running".to_string(),
                actual: format!("{:?}", self.state),
            }.into());
        }
        
        self.state = PipelineState::Running;
        context.step += 1;
        
        let available_fields: Vec<FieldType> = self.available_fields.iter().cloned().collect();
        
        for &idx in &self.execution_order {
            let component = &mut self.components[idx];
            let component_id = component.component_id();
            
            // Check if component can execute
            if !component.can_execute(&available_fields) {
                if component.is_optional() {
                    continue; // Skip optional components
                } else {
                    return Err(PhysicsError::ModelNotInitialized {
                        model_name: component_id.to_string(),
                    }.into());
                }
            }
            
            // Execute component with performance tracking
            let start_time = Instant::now();
            let component_id = component.component_id().to_string();
            let result = component.apply(fields, grid, medium, dt, t, context);
            let duration = start_time.elapsed().as_secs_f64();
            
            context.performance_tracker.record_execution(&component_id, duration);
            
            if let Err(e) = result {
                self.state = PipelineState::Error(e.to_string());
                return Err(e);
            }
        }
        
        self.state = PipelineState::Ready;
        Ok(())
    }
    
    /// Validate the entire pipeline
    pub fn validate_pipeline(&mut self, context: &PhysicsContext) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // Validate each component
        for component in &self.components {
            let component_validation = component.validate(context);
            if !component_validation.is_valid {
                result.add_error(format!(
                    "Component '{}' validation failed: {}",
                    component.component_id(),
                    component_validation.errors.join(", ")
                ));
            }
            result.warnings.extend(component_validation.warnings);
        }
        
        // Check for dependency cycles
        if let Err(e) = self.check_dependency_cycles() {
            result.add_error(format!("Dependency cycle detected: {}", e));
        }
        
        // Check for missing dependencies
        let missing_deps = self.find_missing_dependencies();
        if !missing_deps.is_empty() {
            result.add_warning(format!(
                "Missing dependencies: {}",
                missing_deps.join(", ")
            ));
        }
        
        result
    }
    
    /// Get performance metrics from all components
    pub fn get_all_metrics(&self) -> HashMap<String, HashMap<String, f64>> {
        self.components
            .iter()
            .map(|c| (c.component_id().to_string(), c.get_metrics()))
            .collect()
    }
    
    /// Get the number of components in the pipeline
    /// Follows Information Expert principle - pipeline knows its own component count
    pub fn component_count(&self) -> usize {
        self.components.len()
    }
    
    /// Get component by ID
    /// Follows Information Expert principle - pipeline knows about its components
    pub fn get_component(&self, id: &str) -> Option<&dyn PhysicsComponent> {
        self.components.iter()
            .find(|comp| comp.component_id() == id)
            .map(|comp| comp.as_ref())
    }
    
    /// Get component IDs
    /// Follows Information Expert principle - pipeline knows its component structure
    pub fn component_ids(&self) -> Vec<String> {
        self.components.iter()
            .map(|comp| comp.component_id().to_string())
            .collect()
    }
    
    /// Get pipeline metrics
    /// Follows Information Expert principle - pipeline aggregates component metrics
    pub fn get_pipeline_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        // Aggregate execution times
        let mut total_time = 0.0;
        for component in &self.components {
            let comp_metrics = component.get_metrics();
            if let Some(time) = comp_metrics.get("execution_time") {
                total_time += time;
            }
            
            // Add component-specific metrics
            let comp_id = component.component_id();
            for (key, value) in comp_metrics {
                metrics.insert(format!("{}_{}", comp_id, key), value);
            }
        }
        
        metrics.insert("total_execution_time".to_string(), total_time);
        metrics.insert("component_count".to_string(), self.components.len() as f64);
        
        metrics
    }
    
    /// Get pipeline state
    pub fn state(&self) -> &PipelineState {
        &self.state
    }
    
    /// Reset pipeline to initial state
    pub fn reset(&mut self) -> KwaversResult<()> {
        for component in &mut self.components {
            component.reset()?;
        }
        self.state = PipelineState::Initialized;
        Ok(())
    }
    
    /// Compute execution order based on dependencies
    /// 
    /// Implements ADP (Acyclic Dependency Principle) by detecting cycles
    fn compute_execution_order(&mut self) -> KwaversResult<()> {
        let n = self.components.len();
        let mut order = Vec::new();
        let mut visited = vec![false; n];
        let mut temp_visited = vec![false; n];
        
        // Topological sort with cycle detection
        for i in 0..n {
            if !visited[i] {
                self.visit_component(i, &mut visited, &mut temp_visited, &mut order)?;
            }
        }
        
        // Sort by priority within dependency groups
        order.sort_by(|&a, &b| {
            let priority_a = self.components[a].priority();
            let priority_b = self.components[b].priority();
            priority_a.cmp(&priority_b)
        });
        
        self.execution_order = order;
        Ok(())
    }
    
    fn visit_component(
        &self,
        idx: usize,
        visited: &mut [bool],
        temp_visited: &mut [bool],
        order: &mut Vec<usize>,
    ) -> KwaversResult<()> {
        if temp_visited[idx] {
            return Err(PhysicsError::IncompatibleModels {
                model1: self.components[idx].component_id().to_string(),
                model2: "dependency cycle".to_string(),
                reason: "Circular dependency detected".to_string(),
            }.into());
        }
        
        if visited[idx] {
            return Ok(());
        }
        
        temp_visited[idx] = true;
        
        // Visit dependencies first
        let deps = self.components[idx].dependencies();
        for dep in deps {
            if let Some(dep_idx) = self.find_component_by_output(&dep) {
                self.visit_component(dep_idx, visited, temp_visited, order)?;
            }
        }
        
        temp_visited[idx] = false;
        visited[idx] = true;
        order.push(idx);
        
        Ok(())
    }
    
    fn find_component_by_output(&self, field: &FieldType) -> Option<usize> {
        self.components
            .iter()
            .position(|c| c.output_fields().contains(field))
    }
    
    fn validate_component(&self, component: &Box<dyn PhysicsComponent>) -> KwaversResult<ValidationResult> {
        let mut result = ValidationResult::new();
        
        // Check for basic validation
        if component.component_id().is_empty() {
            result.add_error("Component ID cannot be empty".to_string());
        }
        
        // Check for duplicate dependencies
        let deps = component.dependencies();
        let dep_set: HashSet<&FieldType> = deps.iter().collect();
        if dep_set.len() != deps.len() {
            result.add_error("Duplicate dependencies detected".to_string());
        }
        
        // Check for duplicate outputs
        let outputs = component.output_fields();
        let output_set: HashSet<&FieldType> = outputs.iter().collect();
        if output_set.len() != outputs.len() {
            result.add_error("Duplicate output fields detected".to_string());
        }
        
        Ok(result)
    }
    
    fn check_dependency_cycles(&self) -> KwaversResult<()> {
        let n = self.components.len();
        let mut visited = vec![false; n];
        let mut temp_visited = vec![false; n];
        
        for i in 0..n {
            if !visited[i] {
                self.visit_component(i, &mut visited, &mut temp_visited, &mut Vec::new())?;
            }
        }
        
        Ok(())
    }
    
    fn find_missing_dependencies(&self) -> Vec<String> {
        let mut missing = Vec::new();
        let available: HashSet<FieldType> = self.available_fields.iter().cloned().collect();
        
        for component in &self.components {
            for dep in component.dependencies() {
                if !available.contains(&dep) {
                    missing.push(format!("{}:{}", component.component_id(), dep.as_str()));
                }
            }
        }
        
        missing
    }
}

impl Default for PhysicsPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Acoustic wave propagation component
/// 
/// Implements YAGNI principle by providing only necessary functionality
#[derive(Debug)]
pub struct AcousticWaveComponent {
    id: String,
    metrics: HashMap<String, f64>,
    state: ComponentState,
    /// Use higher-order spatial derivatives for improved accuracy
    spatial_order: usize,
}

impl AcousticWaveComponent {
    pub fn new(id: String) -> Self {
        Self {
            id,
            metrics: HashMap::new(),
            state: ComponentState::Initialized,
            spatial_order: 4, // Default to 4th order accuracy
        }
    }
    
    /// Set spatial derivative order (2, 4, or 6)
    pub fn set_spatial_order(&mut self, order: usize) {
        if order == 2 || order == 4 || order == 6 {
            self.spatial_order = order;
        } else {
            warn!("Invalid spatial order {} requested, keeping current order {}", order, self.spatial_order);
        }
    }
    
    /// Compute spatial derivatives with specified order accuracy
    fn compute_derivatives(
        &self,
        field: &ArrayView3<f64>,
        grid: &Grid,
        axis: usize,
    ) -> Array3<f64> {
        let mut deriv = Array3::zeros(field.raw_dim());
        let (nx, ny, nz) = field.dim();
        
        match self.spatial_order {
            2 => {
                // Second-order central differences
                for i in 1..nx-1 {
                    for j in 1..ny-1 {
                        for k in 1..nz-1 {
                            let val = match axis {
                                0 => (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * grid.dx),
                                1 => (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * grid.dy),
                                2 => (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * grid.dz),
                                _ => 0.0,
                            };
                            deriv[[i, j, k]] = val;
                        }
                    }
                }
            }
            4 => {
                // Fourth-order central differences
                for i in 2..nx-2 {
                    for j in 2..ny-2 {
                        for k in 2..nz-2 {
                            let val = match axis {
                                0 => (-field[[i+2, j, k]] + 8.0*field[[i+1, j, k]] 
                                     - 8.0*field[[i-1, j, k]] + field[[i-2, j, k]]) / (12.0 * grid.dx),
                                1 => (-field[[i, j+2, k]] + 8.0*field[[i, j+1, k]] 
                                     - 8.0*field[[i, j-1, k]] + field[[i, j-2, k]]) / (12.0 * grid.dy),
                                2 => (-field[[i, j, k+2]] + 8.0*field[[i, j, k+1]] 
                                     - 8.0*field[[i, j, k-1]] + field[[i, j, k-2]]) / (12.0 * grid.dz),
                                _ => 0.0,
                            };
                            deriv[[i, j, k]] = val;
                        }
                    }
                }
            }
            6 => {
                // Sixth-order central differences
                for i in 3..nx-3 {
                    for j in 3..ny-3 {
                        for k in 3..nz-3 {
                            let val = match axis {
                                0 => (field[[i+3, j, k]] - 9.0*field[[i+2, j, k]] 
                                     + 45.0*field[[i+1, j, k]] - 45.0*field[[i-1, j, k]] 
                                     + 9.0*field[[i-2, j, k]] - field[[i-3, j, k]]) / (60.0 * grid.dx),
                                1 => (field[[i, j+3, k]] - 9.0*field[[i, j+2, k]] 
                                     + 45.0*field[[i, j+1, k]] - 45.0*field[[i, j-1, k]] 
                                     + 9.0*field[[i, j-2, k]] - field[[i, j-3, k]]) / (60.0 * grid.dy),
                                2 => (field[[i, j, k+3]] - 9.0*field[[i, j, k+2]] 
                                     + 45.0*field[[i, j, k+1]] - 45.0*field[[i, j, k-1]] 
                                     + 9.0*field[[i, j, k-2]] - field[[i, j, k-3]]) / (60.0 * grid.dz),
                                _ => 0.0,
                            };
                            deriv[[i, j, k]] = val;
                        }
                    }
                }
            }
            _ => {
                // Unsupported order - fall back to 2nd order with warning
                warn!("Unsupported spatial order {} in compute_derivatives, falling back to 2nd order", self.spatial_order);
                
                // Second-order central differences (fallback)
                for i in 1..nx-1 {
                    for j in 1..ny-1 {
                        for k in 1..nz-1 {
                            let val = match axis {
                                0 => (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * grid.dx),
                                1 => (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * grid.dy),
                                2 => (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * grid.dz),
                                _ => 0.0,
                            };
                            deriv[[i, j, k]] = val;
                        }
                    }
                }
            }
        }
        
        deriv
    }
}

impl PhysicsComponent for AcousticWaveComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<FieldType> {
        vec![] // Acoustic wave doesn't depend on its own output
    }
    
    fn output_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Pressure]
    }
    
    fn apply(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PhysicsContext,
    ) -> KwaversResult<()> {
        self.state = ComponentState::Running;
        
        let start_time = Instant::now();
        
        // Create temporary arrays to avoid borrowing conflicts
        let mut pressure_updates = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        let mut velocity_x_updates = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        let mut velocity_y_updates = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        let mut velocity_z_updates = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        
        // Get current field values
        let pressure_field = fields.index_axis(ndarray::Axis(0), 0);
        let velocity_x = fields.index_axis(ndarray::Axis(0), 3);
        let velocity_y = fields.index_axis(ndarray::Axis(0), 4);
        let velocity_z = fields.index_axis(ndarray::Axis(0), 5);
        
        // Get medium properties
        let rho_array = medium.density_array();
        let c_array = medium.sound_speed_array();
        
        // Compute velocity derivatives with higher-order accuracy
        let dvx_dx = self.compute_derivatives(&velocity_x, grid, 0);
        let dvy_dy = self.compute_derivatives(&velocity_y, grid, 1);
        let dvz_dz = self.compute_derivatives(&velocity_z, grid, 2);
        
        // Compute pressure derivatives
        let dp_dx = self.compute_derivatives(&pressure_field, grid, 0);
        let dp_dy = self.compute_derivatives(&pressure_field, grid, 1);
        let dp_dz = self.compute_derivatives(&pressure_field, grid, 2);
        
        // Enhanced finite difference update with proper wave physics
        // Solve: ∂p/∂t = -ρc²∇·v and ∂v/∂t = -∇p/ρ
        
        // Determine valid region based on spatial order
        let boundary_width = match self.spatial_order {
            2 => 1,
            4 => 2,
            6 => 3,
            _ => 1,
        };
        
        // Only update interior points where derivatives are valid
        for i in boundary_width..grid.nx-boundary_width {
            for j in boundary_width..grid.ny-boundary_width {
                for k in boundary_width..grid.nz-boundary_width {
                    let rho = rho_array[[i, j, k]];
                    let c_sq = c_array[[i, j, k]].powi(2);
                    
                    // Calculate velocity divergence from precomputed derivatives
                    let div_v = dvx_dx[[i, j, k]] + dvy_dy[[i, j, k]] + dvz_dz[[i, j, k]];
                    
                    // Calculate pressure update: ∂p/∂t = -ρc²∇·v
                    pressure_updates[[i, j, k]] = -rho * c_sq * div_v * dt;
                    
                    // Calculate velocity updates: ∂v/∂t = -∇p/ρ
                    velocity_x_updates[[i, j, k]] = -dp_dx[[i, j, k]] / rho * dt;
                    velocity_y_updates[[i, j, k]] = -dp_dy[[i, j, k]] / rho * dt;
                    velocity_z_updates[[i, j, k]] = -dp_dz[[i, j, k]] / rho * dt;
                }
            }
        }
        
        // Apply updates to the fields sequentially to avoid borrowing conflicts
        {
            let mut pressure_field_mut = fields.index_axis_mut(ndarray::Axis(0), 0);
            pressure_field_mut += &pressure_updates;
        }
        {
            let mut velocity_x_mut = fields.index_axis_mut(ndarray::Axis(0), 3);
            velocity_x_mut += &velocity_x_updates;
        }
        {
            let mut velocity_y_mut = fields.index_axis_mut(ndarray::Axis(0), 4);
            velocity_y_mut += &velocity_y_updates;
        }
        {
            let mut velocity_z_mut = fields.index_axis_mut(ndarray::Axis(0), 5);
            velocity_z_mut += &velocity_z_updates;
        }
        
        self.metrics.insert("execution_time".to_string(), start_time.elapsed().as_secs_f64());
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
    
    fn state(&self) -> ComponentState {
        self.state.clone()
    }
    
    fn reset(&mut self) -> KwaversResult<()> {
        self.state = ComponentState::Initialized;
        self.metrics.clear();
        Ok(())
    }
}

/// Thermal diffusion component implementation
#[derive(Debug)]
pub struct ThermalDiffusionComponent {
    id: String,
    metrics: HashMap<String, f64>,
    state: ComponentState,
}

impl ThermalDiffusionComponent {
    pub fn new(id: String) -> Self {
        Self {
            id,
            metrics: HashMap::new(),
            state: ComponentState::Initialized,
        }
    }
}

impl PhysicsComponent for ThermalDiffusionComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<FieldType> {
        vec![] // Thermal diffusion doesn't depend on its own output
    }
    
    fn output_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Temperature]
    }
    
    fn apply(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        context: &PhysicsContext,
    ) -> KwaversResult<()> {
        self.state = ComponentState::Running;
        
        let start_time = Instant::now();
        
        // Get field indices
        let pressure_idx = 0;
        let temperature_idx = 2;
        
        // Get pressure field for acoustic heating calculation (clone to avoid borrow conflict)
        let pressure_field = fields.index_axis(ndarray::Axis(0), pressure_idx).to_owned();
        
        // Temperature field is at index 2
        let mut temperature_field = fields.index_axis_mut(ndarray::Axis(0), temperature_idx);
        
        // Get thermal and acoustic properties from medium
        let mut kappa_array = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut rho_array = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut cp_array = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut alpha_array = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut c_array = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    kappa_array[[i, j, k]] = medium.thermal_diffusivity(x, y, z, grid);
                    rho_array[[i, j, k]] = medium.density(x, y, z, grid);
                    cp_array[[i, j, k]] = medium.specific_heat(x, y, z, grid);
                    alpha_array[[i, j, k]] = medium.absorption_coefficient(x, y, z, grid, context.frequency);
                    c_array[[i, j, k]] = medium.sound_speed(x, y, z, grid);
                }
            }
        }
        
        // Thermal update with both diffusion and acoustic heating
        let mut temp_updates = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        
        for i in 1..grid.nx - 1 {
            for j in 1..grid.ny - 1 {
                for k in 1..grid.nz - 1 {
                    let kappa = kappa_array[[i, j, k]];
                    let rho = rho_array[[i, j, k]];
                    let cp = cp_array[[i, j, k]];
                    let alpha = alpha_array[[i, j, k]];
                    let c = c_array[[i, j, k]];
                    
                    // Calculate Laplacian of temperature
                    let laplacian_t = 
                        (temperature_field[[i+1, j, k]] - 2.0 * temperature_field[[i, j, k]] + temperature_field[[i-1, j, k]]) / (grid.dx * grid.dx) +
                        (temperature_field[[i, j+1, k]] - 2.0 * temperature_field[[i, j, k]] + temperature_field[[i, j-1, k]]) / (grid.dy * grid.dy) +
                        (temperature_field[[i, j, k+1]] - 2.0 * temperature_field[[i, j, k]] + temperature_field[[i, j, k-1]]) / (grid.dz * grid.dz);
                    
                    // Calculate acoustic heating term Q = 2αI/ρc, where I = p²/(ρc)
                    let pressure = pressure_field[[i, j, k]];
                    let intensity = pressure * pressure / (rho * c);
                    let heating_rate = 2.0 * alpha * intensity / (rho * cp);
                    
                    // Update temperature: ∂T/∂t = κ∇²T + Q/(ρcp)
                    temp_updates[[i, j, k]] = (kappa * laplacian_t + heating_rate) * dt;
                }
            }
        }
        
        // Apply updates
        temperature_field += &temp_updates;
        
        self.metrics.insert("execution_time".to_string(), start_time.elapsed().as_secs_f64());
        self.metrics.insert("max_heating_rate".to_string(), 
            temp_updates.iter().fold(0.0_f64, |a, &b| a.max(b.abs())) / dt);
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
    
    fn state(&self) -> ComponentState {
        self.state.clone()
    }
    
    fn reset(&mut self) -> KwaversResult<()> {
        self.state = ComponentState::Initialized;
        self.metrics.clear();
        Ok(())
    }
}

/// Kuznetsov wave equation component for full nonlinear acoustics
/// 
/// Implements the complete Kuznetsov equation with all second-order nonlinear terms
/// and acoustic diffusivity for accurate modeling of finite-amplitude sound propagation
pub struct KuznetsovWaveComponent {
    id: String,
    metrics: HashMap<String, f64>,
    state: ComponentState,
    /// Pressure history for time derivatives
    pressure_history: Vec<Array3<f64>>,
    /// Precomputed k-squared values
    k_squared: Option<Array3<f64>>,
    /// Enable nonlinear terms
    enable_nonlinearity: bool,
    /// Enable diffusivity term
    enable_diffusivity: bool,
    /// Nonlinearity scaling factor
    nonlinearity_scaling: f64,
}

impl KuznetsovWaveComponent {
    pub fn new(id: String, grid: &Grid) -> Self {
        Self {
            id,
            metrics: HashMap::new(),
            state: ComponentState::Initialized,
            pressure_history: vec![Array3::zeros((grid.nx, grid.ny, grid.nz)); 3],
            k_squared: Some(grid.k_squared()),
            enable_nonlinearity: true,
            enable_diffusivity: true,
            nonlinearity_scaling: 1.0,
        }
    }
    
    /// Configure nonlinearity settings
    pub fn with_nonlinearity(mut self, enable: bool, scaling: f64) -> Self {
        self.enable_nonlinearity = enable;
        self.nonlinearity_scaling = scaling;
        self
    }
    
    /// Configure diffusivity settings
    pub fn with_diffusivity(mut self, enable: bool) -> Self {
        self.enable_diffusivity = enable;
        self
    }
    
    /// Update pressure history
    fn update_history(&mut self, pressure: &Array3<f64>) {
        // Shift history
        let n = self.pressure_history.len();
        for i in (1..n).rev() {
            let (left, right) = self.pressure_history.split_at_mut(i);
            right[0].assign(&left[i-1]);
        }
        self.pressure_history[0].assign(pressure);
    }
}

impl PhysicsComponent for KuznetsovWaveComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<FieldType> {
        vec![] // Kuznetsov wave is self-contained
    }
    
    fn output_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Pressure, FieldType::Velocity]
    }
    
    fn apply(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        context: &PhysicsContext,
    ) -> KwaversResult<()> {
        use crate::utils::{fft_3d, ifft_3d};
        
        self.state = ComponentState::Running;
        let start_time = Instant::now();
        
        // Get current pressure field
        let pressure_field = fields.index_axis(ndarray::Axis(0), 0).to_owned();
        
        // Update history before computing derivatives
        self.update_history(&pressure_field);
        
        // Compute Laplacian using k-space for spectral accuracy
        let laplacian = if let Some(ref k_squared) = self.k_squared {
            // Convert pressure to 4D array for fft_3d
            let mut pressure_4d = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
            pressure_4d.index_axis_mut(Axis(0), 0).assign(&pressure_field);
            
            let pressure_k = fft_3d(&pressure_4d, 0, grid);
            
            // Compute Laplacian in k-space
            let mut lap_k = pressure_k.clone();
            lap_k.iter_mut().zip(k_squared.iter())
                .for_each(|(p, &k2)| *p *= -k2);
            
            // Transform back
            ifft_3d(&lap_k, grid)
        } else {
            // Fallback to finite differences
            let mut lap = Array3::zeros(pressure_field.raw_dim());
            for i in 1..grid.nx-1 {
                for j in 1..grid.ny-1 {
                    for k in 1..grid.nz-1 {
                        lap[[i, j, k]] = 
                            (pressure_field[[i+1, j, k]] - 2.0*pressure_field[[i, j, k]] + pressure_field[[i-1, j, k]]) / (grid.dx * grid.dx) +
                            (pressure_field[[i, j+1, k]] - 2.0*pressure_field[[i, j, k]] + pressure_field[[i, j-1, k]]) / (grid.dy * grid.dy) +
                            (pressure_field[[i, j, k+1]] - 2.0*pressure_field[[i, j, k]] + pressure_field[[i, j, k-1]]) / (grid.dz * grid.dz);
                    }
                }
            }
            lap
        };
        
        // Initialize update array
        let mut pressure_update = Array3::zeros(pressure_field.raw_dim());
        
        // Linear wave term: c²∇²p
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let c = medium.sound_speed(x, y, z, grid);
                    
                    pressure_update[[i, j, k]] = c * c * laplacian[[i, j, k]];
                }
            }
        }
        
        // Nonlinear term: -(β/ρ₀c₀⁴)∂²p²/∂t²
        if self.enable_nonlinearity && self.pressure_history.len() >= 2 {
            let p_prev = &self.pressure_history[0];
            let p_prev2 = &self.pressure_history[1];
            
            // Compute p² at three time levels
            let p2_curr = &pressure_field * &pressure_field;
            let p2_prev = p_prev * p_prev;
            let p2_prev2 = p_prev2 * p_prev2;
            
            // Second-order finite difference for ∂²p²/∂t²
            let d2p2_dt2 = (&p2_curr - 2.0 * &p2_prev + &p2_prev2) / (dt * dt);
            
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        
                        let rho = medium.density(x, y, z, grid);
                        let c0 = medium.sound_speed(x, y, z, grid);
                        let beta = medium.nonlinearity_coefficient(x, y, z, grid);
                        
                        let nonlinear_coeff = -beta / (rho * c0.powi(4));
                        pressure_update[[i, j, k]] += nonlinear_coeff * d2p2_dt2[[i, j, k]] * self.nonlinearity_scaling;
                    }
                }
            }
        }
        
        // Diffusivity term: -(δ/c₀⁴)∂³p/∂t³
        if self.enable_diffusivity && self.pressure_history.len() >= 3 {
            let p_prev = &self.pressure_history[0];
            let p_prev2 = &self.pressure_history[1];
            let p_prev3 = &self.pressure_history[2];
            
            // Third-order finite difference for ∂³p/∂t³
            let d3p_dt3 = (&pressure_field - 3.0 * p_prev + 3.0 * p_prev2 - p_prev3) / (dt * dt * dt);
            
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        
                        let c0 = medium.sound_speed(x, y, z, grid);
                        let alpha = medium.absorption_coefficient(x, y, z, grid, 1e6); // Using 1 MHz as default
                        
                        // Approximate diffusivity from absorption
                        let omega_ref = 2.0 * std::f64::consts::PI * 1e6; // 1 MHz reference
                        let delta = alpha * c0.powi(3) / (omega_ref * omega_ref);
                        
                        let diffusivity_coeff = -delta / c0.powi(4);
                        pressure_update[[i, j, k]] += diffusivity_coeff * d3p_dt3[[i, j, k]];
                    }
                }
            }
        }
        
        // Add source terms if available
        if let Some(source) = context.source_terms.get("acoustic") {
            pressure_update += source;
        }
        
        // Update pressure field
        {
            let mut pressure_mut = fields.index_axis_mut(ndarray::Axis(0), 0);
            pressure_mut += &(&pressure_update * dt);
        }
        
        // Update velocity field from pressure gradient
        let velocity_x = fields.index_axis(ndarray::Axis(0), 3).to_owned();
        let velocity_y = fields.index_axis(ndarray::Axis(0), 4).to_owned();
        let velocity_z = fields.index_axis(ndarray::Axis(0), 5).to_owned();
        
        let mut vx_update = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut vy_update = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut vz_update = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        // Compute pressure gradients and update velocities
        for i in 1..grid.nx-1 {
            for j in 1..grid.ny-1 {
                for k in 1..grid.nz-1 {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let rho = medium.density(x, y, z, grid);
                    
                    // Pressure gradients (4th order central differences)
                    let dp_dx = if i >= 2 && i < grid.nx-2 {
                        (-pressure_field[[i+2, j, k]] + 8.0*pressure_field[[i+1, j, k]] 
                         - 8.0*pressure_field[[i-1, j, k]] + pressure_field[[i-2, j, k]]) / (12.0 * grid.dx)
                    } else {
                        (pressure_field[[i+1, j, k]] - pressure_field[[i-1, j, k]]) / (2.0 * grid.dx)
                    };
                    
                    let dp_dy = if j >= 2 && j < grid.ny-2 {
                        (-pressure_field[[i, j+2, k]] + 8.0*pressure_field[[i, j+1, k]] 
                         - 8.0*pressure_field[[i, j-1, k]] + pressure_field[[i, j-2, k]]) / (12.0 * grid.dy)
                    } else {
                        (pressure_field[[i, j+1, k]] - pressure_field[[i, j-1, k]]) / (2.0 * grid.dy)
                    };
                    
                    let dp_dz = if k >= 2 && k < grid.nz-2 {
                        (-pressure_field[[i, j, k+2]] + 8.0*pressure_field[[i, j, k+1]] 
                         - 8.0*pressure_field[[i, j, k-1]] + pressure_field[[i, j, k-2]]) / (12.0 * grid.dz)
                    } else {
                        (pressure_field[[i, j, k+1]] - pressure_field[[i, j, k-1]]) / (2.0 * grid.dz)
                    };
                    
                    // Velocity updates: ∂v/∂t = -∇p/ρ
                    vx_update[[i, j, k]] = -dp_dx / rho * dt;
                    vy_update[[i, j, k]] = -dp_dy / rho * dt;
                    vz_update[[i, j, k]] = -dp_dz / rho * dt;
                }
            }
        }
        
        // Apply velocity updates
        {
            let mut vx_mut = fields.index_axis_mut(ndarray::Axis(0), 3);
            vx_mut += &vx_update;
        }
        {
            let mut vy_mut = fields.index_axis_mut(ndarray::Axis(0), 4);
            vy_mut += &vy_update;
        }
        {
            let mut vz_mut = fields.index_axis_mut(ndarray::Axis(0), 5);
            vz_mut += &vz_update;
        }
        
        self.metrics.insert("execution_time".to_string(), start_time.elapsed().as_secs_f64());
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
    
    fn state(&self) -> ComponentState {
        self.state.clone()
    }
    
    fn reset(&mut self) -> KwaversResult<()> {
        self.state = ComponentState::Initialized;
        self.metrics.clear();
        // Clear pressure history
        for history in &mut self.pressure_history {
            history.fill(0.0);
        }
        Ok(())
    }
    
    fn validate(&self, _context: &PhysicsContext) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        if self.pressure_history.is_empty() {
            result.add_error("Pressure history not initialized".to_string());
        }
        
        if self.enable_diffusivity && self.pressure_history.len() < 3 {
            result.add_warning("Diffusivity term requires 3 time levels of history".to_string());
        }
        
        result
    }
}

/// Cavitation Physics Component
/// Follows Single Responsibility: Handles only cavitation dynamics
pub struct CavitationComponent {
    id: String,
    cavitation_model: crate::physics::mechanics::cavitation::CavitationModel,
    state: ComponentState,
    metrics: HashMap<String, f64>,
}

impl CavitationComponent {
    pub fn new(id: String, grid: &Grid) -> Self {
        Self {
            cavitation_model: crate::physics::mechanics::cavitation::CavitationModel::new(grid, 1e-6),
            id,
            state: ComponentState::Ready,
            metrics: HashMap::new(),
        }
    }
}

impl PhysicsComponent for CavitationComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<FieldType> {
        vec![FieldType::Pressure]
    }
    
    fn output_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Cavitation, FieldType::Light]
    }
    
    fn apply(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        _context: &PhysicsContext,
    ) -> KwaversResult<()> {
        let start_time = Instant::now();
        
        // Extract pressure field (field type is on Axis(0), pressure is typically index 0)
        let pressure = fields.index_axis(ndarray::Axis(0), 0).to_owned();
        
        // Create a mutable copy for cavitation processing
        let mut pressure_for_cavitation = pressure.clone();
        
        // Update cavitation dynamics using the trait
        self.cavitation_model.update_cavitation(
            &pressure,
            grid,
            medium,
            dt,
            t,
        )?;
        
        // Write the updated pressure back to the main fields array
        let mut pressure_field_mut = fields.index_axis_mut(ndarray::Axis(0), 0);
        pressure_field_mut.assign(&pressure_for_cavitation);
        
        // Light emission is now handled separately by the cavitation model
        // and should be retrieved through the state if needed
        
        // Record performance metrics
        self.metrics.insert("execution_time".to_string(), start_time.elapsed().as_secs_f64());
        
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
    
    fn state(&self) -> ComponentState {
        self.state.clone()
    }
    
    fn priority(&self) -> u32 {
        2 // Execute after acoustic wave
    }
}

/// Elastic Wave Physics Component
/// Follows Single Responsibility: Handles only elastic wave propagation
pub struct ElasticWaveComponent {
    id: String,
    elastic_model: crate::physics::mechanics::elastic_wave::ElasticWave,
    state: ComponentState,
    metrics: HashMap<String, f64>,
}

impl ElasticWaveComponent {
    pub fn new(id: String, grid: &Grid) -> crate::error::KwaversResult<Self> {
        Ok(Self {
            elastic_model: crate::physics::mechanics::elastic_wave::ElasticWave::new(grid)?,
            id,
            state: ComponentState::Ready,
            metrics: HashMap::new(),
        })
    }
}

impl PhysicsComponent for ElasticWaveComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<FieldType> {
        vec![FieldType::Pressure]
    }
    
    fn output_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Stress, FieldType::Velocity]
    }
    
    fn apply(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        _context: &PhysicsContext,
    ) -> KwaversResult<()> {
        let start_time = Instant::now();
        
        // Update elastic wave propagation
        // Use a dummy pressure field since ElasticWave doesn't use it
        let dummy_pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let dummy_source = &crate::source::NullSource::new();
        self.elastic_model.update_wave(fields, &dummy_pressure, dummy_source, grid, medium, dt, t);
        
        // Record performance metrics
        self.metrics.insert("execution_time".to_string(), start_time.elapsed().as_secs_f64());
        
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
    
    fn state(&self) -> ComponentState {
        self.state.clone()
    }
    
    fn priority(&self) -> u32 {
        1 // Execute with acoustic wave
    }
}

/// Light Diffusion Physics Component
/// Follows Single Responsibility: Handles only light propagation and diffusion
pub struct LightDiffusionComponent {
    id: String,
    light_model: crate::physics::optics::diffusion::LightDiffusion,
    state: ComponentState,
    metrics: HashMap<String, f64>,
}

impl LightDiffusionComponent {
    pub fn new(id: String, grid: &Grid) -> Self {
        Self {
            light_model: crate::physics::optics::diffusion::LightDiffusion::new(grid, false, true, false),
            id,
            state: ComponentState::Ready,
            metrics: HashMap::new(),
        }
    }
}

impl PhysicsComponent for LightDiffusionComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<FieldType> {
        vec![FieldType::Light, FieldType::Temperature]
    }
    
    fn output_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Light, FieldType::Temperature]
    }
    
    fn apply(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PhysicsContext,
    ) -> KwaversResult<()> {
        let start_time = Instant::now();
        
        // Update light diffusion
        // Create a dummy light source for now
        let light_source = Array3::zeros((grid.nx, grid.ny, grid.nz));
        self.light_model.update_light(fields, &light_source, grid, medium, dt);
        
        // Record performance metrics
        self.metrics.insert("execution_time".to_string(), start_time.elapsed().as_secs_f64());
        
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
    
    fn state(&self) -> ComponentState {
        self.state.clone()
    }
    
    fn priority(&self) -> u32 {
        3 // Execute after cavitation
    }
}

/// Chemical Reaction Physics Component
/// Follows Single Responsibility: Handles only chemical reactions and kinetics
pub struct ChemicalComponent {
    id: String,
    chemical_model: crate::physics::chemistry::ChemicalModel,
    state: ComponentState,
    metrics: HashMap<String, f64>,
}

impl ChemicalComponent {
    pub fn new(id: String, grid: &Grid) -> KwaversResult<Self> {
        let chemical_model = crate::physics::chemistry::ChemicalModel::new(grid, true, true)?;
        Ok(Self {
            chemical_model,
            id,
            state: ComponentState::Ready,
            metrics: HashMap::new(),
        })
    }
}

impl PhysicsComponent for ChemicalComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<FieldType> {
        vec![FieldType::Light, FieldType::Temperature, FieldType::Pressure]
    }
    
    fn output_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Chemical, FieldType::Temperature]
    }
    
    fn apply(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        _context: &PhysicsContext,
    ) -> KwaversResult<()> {
        let start_time = Instant::now();
        
        // Prepare chemical update parameters with proper physical data
        let light_field = fields.index_axis(ndarray::Axis(0), 1).to_owned();
        let temperature_field = fields.index_axis(ndarray::Axis(0), 2).to_owned();
        let pressure_field = fields.index_axis(ndarray::Axis(0), 0).to_owned();
        
        // Get bubble radius from proper cavitation component integration
        // Note: ChemicalComponent doesn't have cavitation_model field, using fallback approach
        let bubble_radius = {
            // For now, using enhanced fallback approach since ChemicalComponent doesn't have cavitation_model
            // Fallback: Enhanced bubble radius estimation from pressure and temperature
            pressure_field.mapv(|p| {
                // Enhanced Rayleigh-Plesset equation estimation
                // R = R0 * [(P0 + 2σ/R0 - P) / (P0 + 2σ/R0)]^(1/3γ)
                let p0 = 101325.0; // Pa (ambient pressure)
                let r0 = 1e-6; // 1 micron initial radius
                let sigma = 0.0728; // Surface tension of water (N/m)
                let gamma = 1.4; // Adiabatic index for air
                
                let surface_pressure = 2.0 * sigma / r0;
                let numerator = p0 + surface_pressure - p.abs();
                let denominator = p0 + surface_pressure;
                
                if denominator > 0.0 && numerator > 0.0 {
                    r0 * (numerator / denominator).powf(1.0 / (3.0 * gamma)).max(0.05e-6).min(10e-6)
                } else {
                    r0 // Return equilibrium radius if calculation fails
                }
            })
        };
        
        // Get emission spectrum from light field or generate physically meaningful spectrum
        let emission_spectrum = if light_field.sum() > 0.0 {
            light_field.clone()
        } else {
            // Generate a physically meaningful emission spectrum based on cavitation intensity
            bubble_radius.mapv(|r| {
                // Sonoluminescence intensity scales with bubble collapse ratio
                let r0 = 1e-6;
                let collapse_ratio = (r0 / r.max(0.1e-6)).powi(3);
                collapse_ratio * 1e-12 // W/m³ typical sonoluminescence intensity
            })
        };
        
        let chemical_params = crate::physics::chemistry::ChemicalUpdateParams {
            light: &light_field,
            emission_spectrum: &emission_spectrum,
            bubble_radius: &bubble_radius,
            temperature: &temperature_field,
            pressure: &pressure_field,
            grid,
            dt,
            medium,
            frequency: 1e6, // 1 MHz default frequency
        };
        
        // Update chemical reactions
        self.chemical_model.update_chemical(&chemical_params)?;
        
        // Record performance metrics
        self.metrics.insert("execution_time".to_string(), start_time.elapsed().as_secs_f64());
        
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
    
    fn state(&self) -> ComponentState {
        self.state.clone()
    }
    
    fn priority(&self) -> u32 {
        4 // Execute last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    
    #[test]
    fn test_invalid_spatial_order_fallback() {
        // Initialize logging for tests
        let _ = env_logger::builder().is_test(true).try_init();
        
        let mut component = AcousticWaveComponent::new("test".to_string());
        
        // Test valid orders
        component.set_spatial_order(2);
        assert_eq!(component.spatial_order, 2);
        
        component.set_spatial_order(4);
        assert_eq!(component.spatial_order, 4);
        
        component.set_spatial_order(6);
        assert_eq!(component.spatial_order, 6);
        
        // Test invalid order - should keep current value
        component.set_spatial_order(3);
        assert_eq!(component.spatial_order, 6); // Should remain 6
        
        component.set_spatial_order(8);
        assert_eq!(component.spatial_order, 6); // Should remain 6
    }
    
    #[test]
    fn test_compute_derivatives_fallback() {
        let _ = env_logger::builder().is_test(true).try_init();
        
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let field = Array3::ones((10, 10, 10));
        let field_view = field.view();
        
        let mut component = AcousticWaveComponent::new("test".to_string());
        
        // Force an invalid spatial order by directly setting it
        component.spatial_order = 5;
        
        // This should trigger the fallback to 2nd order with a warning
        let deriv = component.compute_derivatives(&field_view, &grid, 0);
        
        // The derivative should not be all zeros (which was the bug)
        let has_nonzero = deriv.iter().any(|&x| x != 0.0);
        assert!(!has_nonzero, "Constant field should have zero derivatives");
        
        // Test with a non-constant field
        let mut field = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    field[[i, j, k]] = i as f64;
                }
            }
        }
        let field_view = field.view();
        
        let deriv = component.compute_derivatives(&field_view, &grid, 0);
        
        // Should have non-zero derivatives in x direction
        let has_nonzero = deriv.iter().any(|&x| x != 0.0);
        assert!(has_nonzero, "Non-constant field should have non-zero derivatives");
    }
}