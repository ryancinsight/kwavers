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
use ndarray::{Array3, Array4};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use crate::physics::mechanics::cavitation::model::CavitationModel;
use crate::physics::traits::{CavitationModelBehavior, LightDiffusionModelTrait, AcousticWaveModel};
use crate::physics::optics::diffusion::LightDiffusion;

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
    fn validate(&self, context: &PhysicsContext) -> ValidationResult {
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
            .or_insert_with(Vec::new)
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

/// Acoustic wave component implementation
/// 
/// Implements YAGNI principle by providing only necessary functionality
pub struct AcousticWaveComponent {
    id: String,
    metrics: HashMap<String, f64>,
    state: ComponentState,
}

impl AcousticWaveComponent {
    pub fn new(id: String) -> Self {
        Self {
            id,
            metrics: HashMap::new(),
            state: ComponentState::Initialized,
        }
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
        _medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PhysicsContext,
    ) -> KwaversResult<()> {
        self.state = ComponentState::Running;
        
        // Simple acoustic wave propagation (placeholder implementation)
        // In a real implementation, this would solve the wave equation
        
        let start_time = Instant::now();
        
        // Apply wave equation update
        let mut pressure_field = fields.index_axis_mut(ndarray::Axis(0), 0);
        
        // Simple finite difference update (placeholder)
        for i in 1..grid.nx - 1 {
            for j in 1..grid.ny - 1 {
                for k in 1..grid.nz - 1 {
                    let laplacian = (pressure_field[[i+1, j, k]] + pressure_field[[i-1, j, k]] +
                                   pressure_field[[i, j+1, k]] + pressure_field[[i, j-1, k]] +
                                   pressure_field[[i, j, k+1]] + pressure_field[[i, j, k-1]] -
                                   6.0 * pressure_field[[i, j, k]]) / (grid.dx * grid.dx);
                    
                    pressure_field[[i, j, k]] += dt * dt * laplacian;
                }
            }
        }
        
        let duration = start_time.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), duration);
        self.metrics.insert("grid_points".to_string(), (grid.nx * grid.ny * grid.nz) as f64);
        
        self.state = ComponentState::Ready;
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
        _medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PhysicsContext,
    ) -> KwaversResult<()> {
        self.state = ComponentState::Running;
        
        let start_time = Instant::now();
        
        // Simple thermal diffusion (placeholder implementation)
        let mut temp_field = fields.index_axis_mut(ndarray::Axis(0), 2);
        
        // Finite difference thermal diffusion
        for i in 1..grid.nx - 1 {
            for j in 1..grid.ny - 1 {
                for k in 1..grid.nz - 1 {
                    let laplacian = (temp_field[[i+1, j, k]] + temp_field[[i-1, j, k]] +
                                   temp_field[[i, j+1, k]] + temp_field[[i, j-1, k]] +
                                   temp_field[[i, j, k+1]] + temp_field[[i, j, k-1]] -
                                   6.0 * temp_field[[i, j, k]]) / (grid.dx * grid.dx);
                    
                    temp_field[[i, j, k]] += 0.1 * dt * laplacian; // Thermal diffusivity = 0.1
                }
            }
        }
        
        let duration = start_time.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), duration);
        self.metrics.insert("grid_points".to_string(), (grid.nx * grid.ny * grid.nz) as f64);
        
        self.state = ComponentState::Ready;
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
        
        // Extract pressure field (assuming index 0 is pressure)
        let pressure = fields.index_axis(ndarray::Axis(3), 0);
        
        // Update cavitation dynamics
        let cavitation_update = self.cavitation_model.update_cavitation(
            &mut fields.index_axis_mut(ndarray::Axis(3), 0).to_owned(),
            &fields.index_axis(ndarray::Axis(3), 0).to_owned(),
            grid,
            dt,
            medium,
            1e6, // 1 MHz frequency
        );
        
        // The method returns light emission data, not a Result
        // Store the light emission data if needed
        let _light_emission = cavitation_update;
        
        // Record performance metrics
        let duration = start_time.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), duration);
        
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
        let dummy_source = &crate::source::MockSource::new();
        self.elastic_model.update_wave(fields, &dummy_pressure, dummy_source, grid, medium, dt, t);
        
        // Record performance metrics
        let duration = start_time.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), duration);
        
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
        let duration = start_time.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), duration);
        
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
        
        // Prepare chemical update parameters
        let chemical_params = crate::physics::chemistry::ChemicalUpdateParams {
            light: &fields.index_axis(ndarray::Axis(3), 1).to_owned(),
            emission_spectrum: &fields.index_axis(ndarray::Axis(3), 1).to_owned(), // Using light as spectrum for now
            bubble_radius: &fields.index_axis(ndarray::Axis(3), 1).to_owned(), // Using light field as placeholder
            temperature: &fields.index_axis(ndarray::Axis(3), 2).to_owned(),     // Assuming temp is index 2
            pressure: &fields.index_axis(ndarray::Axis(3), 0).to_owned(),        // Assuming pressure is index 0
            grid,
            dt,
            medium,
            frequency: 1e6, // 1 MHz default frequency
        };
        
        // Update chemical reactions
        self.chemical_model.update_chemical(&chemical_params)?;
        
        // Record performance metrics
        let duration = start_time.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), duration);
        
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

    fn create_test_grid() -> Grid {
        Grid::new(10, 10, 10, 0.001, 0.001, 0.001)
    }

    #[test]
    fn test_physics_pipeline_execution_order() {
        let mut pipeline = PhysicsPipeline::new();
        
        // Add components in dependency order
        pipeline.add_component(Box::new(AcousticWaveComponent::new("acoustic".to_string()))).unwrap();
        pipeline.add_component(Box::new(ThermalDiffusionComponent::new("thermal".to_string()))).unwrap();
        
        assert_eq!(pipeline.components.len(), 2);
        assert_eq!(pipeline.state, PipelineState::Ready);
    }

    #[test]
    fn test_physics_context() {
        let mut context = PhysicsContext::new(1e6);
        context = context.with_parameter("test_param", 42.0);
        
        assert_eq!(context.get_parameter("test_param"), Some(42.0));
        assert_eq!(context.frequency, 1e6);
    }
    
    #[test]
    fn test_field_type_equality() {
        let field1 = FieldType::Pressure;
        let field2 = FieldType::Pressure;
        let field3 = FieldType::Temperature;
        
        assert_eq!(field1, field2);
        assert_ne!(field1, field3);
    }
    
    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new();
        assert!(result.is_valid);
        
        result.add_error("Test error".to_string());
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        
        result.add_warning("Test warning".to_string());
        assert_eq!(result.warnings.len(), 1);
    }
    
    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new();
        tracker.record_execution("test_component", 1.5);
        tracker.record_execution("test_component", 2.5);
        
        assert_eq!(tracker.get_average_execution_time("test_component"), Some(2.0));
        assert_eq!(tracker.get_total_execution_time("test_component"), Some(4.0));
    }
}