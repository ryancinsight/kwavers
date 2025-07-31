// src/physics/core/system.rs
//! System processing for physics effects
//! 
//! This module implements the system part of the ECS architecture,
//! providing scheduling and execution of physics effects.

use crate::error::KwaversResult;
use crate::physics::core::{PhysicsEffect, EffectId, EffectContext, EventBus};
use crate::physics::state::PhysicsState;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Context for system execution
#[derive(Debug, Clone)]
pub struct SystemContext {
    /// Current simulation time
    pub time: f64,
    /// Time step
    pub dt: f64,
    /// Simulation step
    pub step: usize,
    /// Event bus for communication
    pub event_bus: Arc<EventBus>,
    /// Global parameters
    pub parameters: HashMap<String, f64>,
}

impl SystemContext {
    /// Create a new system context
    pub fn new(time: f64, dt: f64, step: usize) -> Self {
        Self {
            time,
            dt,
            step,
            event_bus: Arc::new(EventBus::new()),
            parameters: HashMap::new(),
        }
    }
}

/// Physics system that processes effects
pub trait PhysicsSystem: Send + Sync {
    /// Process all effects for one time step
    fn process(&mut self, state: &mut PhysicsState, context: &SystemContext) -> KwaversResult<()>;
    
    /// Get system name
    fn name(&self) -> &str;
    
    /// Get performance metrics
    fn metrics(&self) -> HashMap<String, f64>;
}

/// Scheduler for physics systems
pub trait SystemScheduler: Send + Sync {
    /// Schedule and execute systems
    fn execute(
        &mut self,
        systems: &mut [Box<dyn PhysicsSystem>],
        state: &mut PhysicsState,
        context: &SystemContext,
    ) -> KwaversResult<()>;
    
    /// Get scheduler name
    fn name(&self) -> &str;
}

/// Sequential scheduler that runs systems one after another
pub struct SequentialScheduler {
    /// Execution times for each system
    execution_times: HashMap<String, Vec<f64>>,
}

impl SequentialScheduler {
    pub fn new() -> Self {
        Self {
            execution_times: HashMap::new(),
        }
    }
}

impl SystemScheduler for SequentialScheduler {
    fn execute(
        &mut self,
        systems: &mut [Box<dyn PhysicsSystem>],
        state: &mut PhysicsState,
        context: &SystemContext,
    ) -> KwaversResult<()> {
        for system in systems.iter_mut() {
            let start = Instant::now();
            system.process(state, context)?;
            let elapsed = start.elapsed().as_secs_f64();
            
            self.execution_times
                .entry(system.name().to_string())
                .or_insert_with(Vec::new)
                .push(elapsed);
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "SequentialScheduler"
    }
}

/// Parallel scheduler that runs independent systems concurrently
pub struct ParallelScheduler {
    /// Dependency graph
    dependencies: HashMap<String, HashSet<String>>,
    /// Execution times
    execution_times: Arc<Mutex<HashMap<String, Vec<f64>>>>,
}

impl ParallelScheduler {
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            execution_times: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Add a dependency between systems
    pub fn add_dependency(&mut self, system: &str, depends_on: &str) {
        self.dependencies
            .entry(system.to_string())
            .or_insert_with(HashSet::new)
            .insert(depends_on.to_string());
    }
}

impl SystemScheduler for ParallelScheduler {
    fn execute(
        &mut self,
        systems: &mut [Box<dyn PhysicsSystem>],
        state: &mut PhysicsState,
        context: &SystemContext,
    ) -> KwaversResult<()> {
        // For now, fall back to sequential execution
        // In a real implementation, we would:
        // 1. Build execution groups based on dependencies
        // 2. Execute each group in parallel
        // 3. Synchronize between groups
        
        for system in systems.iter_mut() {
            let start = Instant::now();
            system.process(state, context)?;
            let elapsed = start.elapsed().as_secs_f64();
            
            let mut times = self.execution_times.lock().unwrap();
            times
                .entry(system.name().to_string())
                .or_insert_with(Vec::new)
                .push(elapsed);
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "ParallelScheduler"
    }
}

/// Effect-based physics system
#[derive(Debug)]
pub struct EffectSystem {
    /// Name of this system
    name: String,
    /// Physics effects managed by this system
    effects: Vec<Box<dyn PhysicsEffect>>,
    /// Execution order based on dependencies
    execution_order: Vec<usize>,
    /// Performance metrics
    metrics: HashMap<String, f64>,
}

impl EffectSystem {
    /// Create a new effect system
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            effects: Vec::new(),
            execution_order: Vec::new(),
            metrics: HashMap::new(),
        }
    }
    
    /// Add an effect to the system
    pub fn add_effect(&mut self, effect: Box<dyn PhysicsEffect>) -> KwaversResult<()> {
        self.effects.push(effect);
        self.compute_execution_order()?;
        Ok(())
    }
    
    /// Compute execution order based on dependencies
    fn compute_execution_order(&mut self) -> KwaversResult<()> {
        // Build dependency graph
        let mut graph: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        
        // Initialize
        for i in 0..self.effects.len() {
            graph.insert(i, HashSet::new());
            in_degree.insert(i, 0);
        }
        
        // Build edges based on effect dependencies
        for (i, effect) in self.effects.iter().enumerate() {
            let required = effect.required_effects();
            
            for (j, other) in self.effects.iter().enumerate() {
                if i != j && required.contains(other.id()) {
                    graph.get_mut(&j).unwrap().insert(i);
                    *in_degree.get_mut(&i).unwrap() += 1;
                }
            }
        }
        
        // Topological sort using Kahn's algorithm
        let mut queue: Vec<usize> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&idx, _)| idx)
            .collect();
        
        let mut order = Vec::new();
        
        while let Some(idx) = queue.pop() {
            order.push(idx);
            
            if let Some(neighbors) = graph.get(&idx) {
                for &neighbor in neighbors {
                    *in_degree.get_mut(&neighbor).unwrap() -= 1;
                    if in_degree[&neighbor] == 0 {
                        queue.push(neighbor);
                    }
                }
            }
        }
        
        if order.len() != self.effects.len() {
            return Err(crate::error::PhysicsError::InvalidConfiguration {
                component: "EffectSystem".to_string(),
                reason: "Circular dependency detected in effects".to_string(),
            }.into());
        }
        
        self.execution_order = order;
        Ok(())
    }
    
    /// Get effects in execution order
    pub fn effects_ordered(&self) -> Vec<&dyn PhysicsEffect> {
        self.execution_order
            .iter()
            .map(|&i| self.effects[i].as_ref())
            .collect()
    }
}

impl PhysicsSystem for EffectSystem {
    fn process(&mut self, state: &mut PhysicsState, context: &SystemContext) -> KwaversResult<()> {
        let effect_context = EffectContext::new(context.time, context.dt, context.step)
            .with_fields(state.available_fields());
        
        // Process effects in dependency order
        for &idx in &self.execution_order {
            let effect = &mut self.effects[idx];
            
            if !effect.is_enabled() {
                continue;
            }
            
            let start = Instant::now();
            
            // Handle incoming events
            let events = context.event_bus.history();
            for event in events {
                effect.handle_event(&event)?;
            }
            
            // Update effect
            effect.update(state, &effect_context)?;
            
            // Emit events
            let emitted = effect.emit_events();
            context.event_bus.publish_batch(emitted);
            
            // Track metrics
            let elapsed = start.elapsed().as_secs_f64();
            self.metrics.insert(
                format!("{}_time", effect.name()),
                elapsed,
            );
        }
        
        // Process all events
        context.event_bus.process_events()?;
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
}

/// System builder for convenient construction
pub struct SystemBuilder {
    name: String,
    effects: Vec<Box<dyn PhysicsEffect>>,
}

impl SystemBuilder {
    /// Create a new system builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            effects: Vec::new(),
        }
    }
    
    /// Add an effect
    pub fn with_effect(mut self, effect: Box<dyn PhysicsEffect>) -> Self {
        self.effects.push(effect);
        self
    }
    
    /// Build the system
    pub fn build(self) -> KwaversResult<EffectSystem> {
        let mut system = EffectSystem::new(self.name);
        for effect in self.effects {
            system.add_effect(effect)?;
        }
        Ok(system)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::composable::FieldType;
    
    // Mock effect for testing
    #[derive(Debug)]
    struct MockEffect {
        id: EffectId,
        dependencies: Vec<EffectId>,
    }
    
    impl PhysicsEffect for MockEffect {
        fn id(&self) -> &EffectId {
            &self.id
        }
        
        fn category(&self) -> crate::physics::core::EffectCategory {
            crate::physics::core::EffectCategory::Custom
        }
        
        fn required_effects(&self) -> Vec<EffectId> {
            self.dependencies.clone()
        }
        
        fn required_fields(&self) -> Vec<FieldType> {
            vec![]
        }
        
        fn provided_fields(&self) -> Vec<FieldType> {
            vec![]
        }
        
        fn update(&mut self, _state: &mut PhysicsState, _context: &EffectContext) -> KwaversResult<()> {
            Ok(())
        }
    }
    
    #[test]
    fn test_effect_system_ordering() {
        let mut system = EffectSystem::new("test");
        
        // Create effects with dependencies
        let effect1 = Box::new(MockEffect {
            id: EffectId::from("effect1"),
            dependencies: vec![],
        });
        
        let effect2 = Box::new(MockEffect {
            id: EffectId::from("effect2"),
            dependencies: vec![EffectId::from("effect1")],
        });
        
        let effect3 = Box::new(MockEffect {
            id: EffectId::from("effect3"),
            dependencies: vec![EffectId::from("effect2")],
        });
        
        // Add in reverse order to test sorting
        system.add_effect(effect3).unwrap();
        system.add_effect(effect2).unwrap();
        system.add_effect(effect1).unwrap();
        
        // Check execution order
        let ordered = system.effects_ordered();
        assert_eq!(ordered[0].id().0, "effect1");
        assert_eq!(ordered[1].id().0, "effect2");
        assert_eq!(ordered[2].id().0, "effect3");
    }
}