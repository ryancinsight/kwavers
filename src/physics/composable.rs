// src/physics/composable.rs
//! Composable physics system following CUPID principles
//! 
//! This module provides a composable, predictable physics simulation system where:
//! - Composable: Physics models can be combined in flexible ways
//! - Unix-like: Each model does one thing well and can be piped together
//! - Predictable: Same inputs always produce same outputs
//! - Idiomatic: Uses Rust's type system and ownership model effectively
//! - Domain-focused: Clear separation between different physics domains

use crate::error::{KwaversResult, PhysicsError};
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array3, Array4};
use std::collections::HashMap;

/// A composable physics component that can be combined with others
pub trait PhysicsComponent: Send + Sync {
    /// Unique identifier for this component
    fn component_id(&self) -> &str;
    
    /// Dependencies this component requires from other components
    fn dependencies(&self) -> Vec<&str>;
    
    /// Fields this component produces or modifies
    fn output_fields(&self) -> Vec<&str>;
    
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
    fn can_execute(&self, available_fields: &[&str]) -> bool {
        self.dependencies().iter().all(|dep| available_fields.contains(dep))
    }
    
    /// Get performance metrics for this component
    fn get_metrics(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Context shared between physics components
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
}

impl PhysicsContext {
    pub fn new(frequency: f64) -> Self {
        Self {
            parameters: HashMap::new(),
            frequency,
            step: 0,
            source_terms: HashMap::new(),
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
}

/// A composable physics pipeline that executes components in dependency order
pub struct PhysicsPipeline {
    components: Vec<Box<dyn PhysicsComponent>>,
    execution_order: Vec<usize>,
    available_fields: Vec<String>,
}

impl PhysicsPipeline {
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            execution_order: Vec::new(),
            available_fields: Vec::new(),
        }
    }
    
    /// Add a physics component to the pipeline
    pub fn add_component(&mut self, component: Box<dyn PhysicsComponent>) -> KwaversResult<()> {
        // Check for duplicate component IDs
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
            if !self.available_fields.contains(&field.to_string()) {
                self.available_fields.push(field.to_string());
            }
        }
        
        self.components.push(component);
        self.compute_execution_order()?;
        Ok(())
    }
    
    /// Execute all components in the pipeline for one time step
    pub fn execute(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PhysicsContext,
    ) -> KwaversResult<()> {
        context.step += 1;
        
        for &idx in &self.execution_order {
            let component = &mut self.components[idx];
            
            // Check if component can execute
            let available_refs: Vec<&str> = self.available_fields.iter().map(|s| s.as_str()).collect();
            if !component.can_execute(&available_refs) {
                return Err(PhysicsError::ModelNotInitialized {
                    model_name: component.component_id().to_string(),
                }.into());
            }
            
            // Execute component
            component.apply(fields, grid, medium, dt, t, context)?;
        }
        
        Ok(())
    }
    
    /// Get performance metrics from all components
    pub fn get_all_metrics(&self) -> HashMap<String, HashMap<String, f64>> {
        self.components
            .iter()
            .map(|c| (c.component_id().to_string(), c.get_metrics()))
            .collect()
    }
    
    /// Compute execution order based on dependencies
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
        
        // Don't reverse - the order is already correct for dependency execution
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
        let dependencies = self.components[idx].dependencies();
        for dep in dependencies {
            if let Some(dep_idx) = self.components.iter().position(|c| {
                c.output_fields().contains(&dep)
            }) {

                self.visit_component(dep_idx, visited, temp_visited, order)?;
            }
        }
        
        temp_visited[idx] = false;
        visited[idx] = true;
        order.push(idx);
        
        Ok(())
    }
}

impl Default for PhysicsPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple acoustic wave component following the composable pattern
#[derive(Debug)]
pub struct AcousticWaveComponent {
    id: String,
    metrics: HashMap<String, f64>,
}

impl AcousticWaveComponent {
    pub fn new(id: String) -> Self {
        Self {
            id,
            metrics: HashMap::new(),
        }
    }
}

impl PhysicsComponent for AcousticWaveComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<&str> {
        vec![] // No dependencies for basic acoustic wave
    }
    
    fn output_fields(&self) -> Vec<&str> {
        vec!["pressure"]
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
        use std::time::Instant;
        let start = Instant::now();
        
        // Simple wave equation: d²p/dt² = c²∇²p
        // This is a placeholder - in practice, you'd use the full k-space solver
        let pressure_idx = 0; // Assuming pressure is at index 0
        let mut pressure = fields.index_axis(ndarray::Axis(0), pressure_idx).to_owned();
        
        // Apply Laplacian (simplified finite difference)
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let c_squared = 1500.0_f64.powi(2); // Simplified constant sound speed
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let laplacian = (pressure[[i+1,j,k]] + pressure[[i-1,j,k]] - 2.0*pressure[[i,j,k]]) / (grid.dx * grid.dx)
                                  + (pressure[[i,j+1,k]] + pressure[[i,j-1,k]] - 2.0*pressure[[i,j,k]]) / (grid.dy * grid.dy)
                                  + (pressure[[i,j,k+1]] + pressure[[i,j,k-1]] - 2.0*pressure[[i,j,k]]) / (grid.dz * grid.dz);
                    
                    pressure[[i,j,k]] += dt * dt * c_squared * laplacian;
                }
            }
        }
        
        // Update the field
        fields.index_axis_mut(ndarray::Axis(0), pressure_idx).assign(&pressure);
        
        // Record metrics
        let elapsed = start.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), elapsed);
        self.metrics.insert("grid_points_processed".to_string(), (nx * ny * nz) as f64);
        
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
}

/// A thermal diffusion component
#[derive(Debug)]
pub struct ThermalDiffusionComponent {
    id: String,
    metrics: HashMap<String, f64>,
}

impl ThermalDiffusionComponent {
    pub fn new(id: String) -> Self {
        Self {
            id,
            metrics: HashMap::new(),
        }
    }
}

impl PhysicsComponent for ThermalDiffusionComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<&str> {
        vec!["pressure"] // Depends on pressure for heating source
    }
    
    fn output_fields(&self) -> Vec<&str> {
        vec!["temperature"]
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
        use std::time::Instant;
        let start = Instant::now();
        
        // Heat diffusion: dT/dt = α∇²T + Q
        // where Q is the heat source from acoustic absorption
        let temperature_idx = 2; // Assuming temperature is at index 2
        let pressure_idx = 0;
        
        let pressure = fields.index_axis(ndarray::Axis(0), pressure_idx);
        let mut temperature = fields.index_axis(ndarray::Axis(0), temperature_idx).to_owned();
        
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let alpha = 1e-7; // Thermal diffusivity (simplified)
        let absorption_coeff = 0.5; // Acoustic absorption coefficient
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Thermal diffusion
                    let laplacian = (temperature[[i+1,j,k]] + temperature[[i-1,j,k]] - 2.0*temperature[[i,j,k]]) / (grid.dx * grid.dx)
                                  + (temperature[[i,j+1,k]] + temperature[[i,j-1,k]] - 2.0*temperature[[i,j,k]]) / (grid.dy * grid.dy)
                                  + (temperature[[i,j,k+1]] + temperature[[i,j,k-1]] - 2.0*temperature[[i,j,k]]) / (grid.dz * grid.dz);
                    
                    // Heat source from acoustic absorption
                    let heat_source = absorption_coeff * pressure[[i,j,k]].powi(2);
                    
                    temperature[[i,j,k]] += dt * (alpha * laplacian + heat_source);
                }
            }
        }
        
        // Update the field
        fields.index_axis_mut(ndarray::Axis(0), temperature_idx).assign(&temperature);
        
        // Record metrics
        let elapsed = start.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), elapsed);
        self.metrics.insert("max_temperature".to_string(), 
            temperature.iter().fold(0.0_f64, |a, &b| a.max(b)));
        
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_physics_pipeline_execution_order() {
        let mut pipeline = PhysicsPipeline::new();
        
        // Add components - acoustic (outputs pressure), thermal (depends on pressure)
        let acoustic = Box::new(AcousticWaveComponent::new("acoustic".to_string()));
        let thermal = Box::new(ThermalDiffusionComponent::new("thermal".to_string()));
        
        // Verify dependencies and outputs
        assert_eq!(acoustic.dependencies(), Vec::<&str>::new());
        assert_eq!(acoustic.output_fields(), vec!["pressure"]);
        assert_eq!(thermal.dependencies(), vec!["pressure"]);
        assert_eq!(thermal.output_fields(), vec!["temperature"]);
        
        pipeline.add_component(acoustic).unwrap();
        pipeline.add_component(thermal).unwrap();
        
        // Should execute acoustic first (index 0), then thermal (index 1)
        assert_eq!(pipeline.execution_order.len(), 2);
        

        
        // The order should be: acoustic (0) then thermal (1)
        assert_eq!(pipeline.components[pipeline.execution_order[0]].component_id(), "acoustic");
        assert_eq!(pipeline.components[pipeline.execution_order[1]].component_id(), "thermal");
    }
    
    #[test]
    fn test_physics_context() {
        let mut context = PhysicsContext::new(1e6)
            .with_parameter("amplitude", 1e5)
            .with_parameter("duration", 1e-3);
        
        assert_eq!(context.frequency, 1e6);
        assert_eq!(context.get_parameter("amplitude"), Some(1e5));
        assert_eq!(context.get_parameter("nonexistent"), None);
        
        let source = Array3::zeros((10, 10, 10));
        context.add_source_term("pressure_source".to_string(), source);
        assert!(context.get_source_term("pressure_source").is_some());
    }
}