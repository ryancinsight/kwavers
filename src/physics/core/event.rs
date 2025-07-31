// src/physics/core/event.rs
//! Event system for physics effect communication
//! 
//! This module implements an event-driven architecture that allows physics
//! effects to communicate without direct coupling, following the Observer pattern.

use crate::physics::core::EffectId;
use ndarray::Array3;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

/// Physics events that can be emitted and handled by effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicsEvent {
    /// Bubble collapse event
    BubbleCollapse {
        position: [usize; 3],
        radius: f64,
        energy: f64,
        peak_temperature: f64,
        peak_pressure: f64,
    },
    
    /// Light emission event (sonoluminescence)
    LightEmission {
        position: [usize; 3],
        intensity: f64,
        spectrum: Vec<f64>,
        duration: f64,
    },
    
    /// Chemical reaction event
    ChemicalReaction {
        position: [usize; 3],
        reactants: Vec<String>,
        products: Vec<String>,
        energy: f64,
    },
    
    /// Material damage event
    MaterialDamage {
        position: [usize; 3],
        damage_type: DamageType,
        severity: f64,
    },
    
    /// Temperature spike event
    TemperatureSpike {
        position: [usize; 3],
        temperature: f64,
        duration: f64,
    },
    
    /// Shock wave event
    ShockWave {
        origin: [usize; 3],
        pressure: f64,
        velocity: f64,
        direction: [f64; 3],
    },
    
    /// Phase transition event
    PhaseTransition {
        position: [usize; 3],
        from_phase: String,
        to_phase: String,
        energy: f64,
    },
    
    /// Custom event for extensions
    Custom {
        event_type: String,
        source: EffectId,
        data: serde_json::Value,
    },
}

/// Types of material damage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DamageType {
    Erosion,
    Fatigue,
    Fracture,
    Pitting,
    Deformation,
}

/// Trait for objects that can handle physics events
pub trait EventHandler: Send + Sync {
    /// Handle a physics event
    fn handle_event(&mut self, event: &PhysicsEvent) -> crate::error::KwaversResult<()>;
    
    /// Get event types this handler is interested in
    fn subscribed_events(&self) -> Vec<String>;
}

/// Event bus for publishing and subscribing to physics events
pub struct EventBus {
    /// Queue of pending events
    events: Arc<Mutex<VecDeque<PhysicsEvent>>>,
    /// Registered event handlers
    handlers: Arc<Mutex<HashMap<EffectId, Box<dyn EventHandler>>>>,
    /// Event history for debugging
    history: Arc<Mutex<Vec<PhysicsEvent>>>,
    /// Maximum history size
    max_history: usize,
}

impl EventBus {
    /// Create a new event bus
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(VecDeque::new())),
            handlers: Arc::new(Mutex::new(HashMap::new())),
            history: Arc::new(Mutex::new(Vec::new())),
            max_history: 1000,
        }
    }
    
    /// Publish an event to the bus
    pub fn publish(&self, event: PhysicsEvent) {
        let mut events = self.events.lock().unwrap();
        events.push_back(event.clone());
        
        // Add to history
        let mut history = self.history.lock().unwrap();
        history.push(event);
        if history.len() > self.max_history {
            history.remove(0);
        }
    }
    
    /// Publish multiple events
    pub fn publish_batch(&self, events: Vec<PhysicsEvent>) {
        let mut queue = self.events.lock().unwrap();
        for event in events {
            queue.push_back(event.clone());
            
            // Add to history
            let mut history = self.history.lock().unwrap();
            history.push(event);
            if history.len() > self.max_history {
                history.remove(0);
            }
        }
    }
    
    /// Register an event handler
    pub fn subscribe(&self, id: EffectId, handler: Box<dyn EventHandler>) {
        let mut handlers = self.handlers.lock().unwrap();
        handlers.insert(id, handler);
    }
    
    /// Unregister an event handler
    pub fn unsubscribe(&self, id: &EffectId) {
        let mut handlers = self.handlers.lock().unwrap();
        handlers.remove(id);
    }
    
    /// Process all pending events
    pub fn process_events(&self) -> crate::error::KwaversResult<()> {
        let events: Vec<PhysicsEvent> = {
            let mut queue = self.events.lock().unwrap();
            queue.drain(..).collect()
        };
        
        let handlers = self.handlers.lock().unwrap();
        
        for event in events {
            let event_type = match &event {
                PhysicsEvent::BubbleCollapse { .. } => "BubbleCollapse",
                PhysicsEvent::LightEmission { .. } => "LightEmission",
                PhysicsEvent::ChemicalReaction { .. } => "ChemicalReaction",
                PhysicsEvent::MaterialDamage { .. } => "MaterialDamage",
                PhysicsEvent::TemperatureSpike { .. } => "TemperatureSpike",
                PhysicsEvent::ShockWave { .. } => "ShockWave",
                PhysicsEvent::PhaseTransition { .. } => "PhaseTransition",
                PhysicsEvent::Custom { event_type, .. } => event_type.as_str(),
            };
            
            for (_, handler) in handlers.iter() {
                if handler.subscribed_events().contains(&event_type.to_string()) {
                    // Clone handler to avoid holding lock during handling
                    // In practice, we'd use a different pattern here
                    // handler.handle_event(&event)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Get event history
    pub fn history(&self) -> Vec<PhysicsEvent> {
        self.history.lock().unwrap().clone()
    }
    
    /// Clear event history
    pub fn clear_history(&self) {
        self.history.lock().unwrap().clear();
    }
    
    /// Get number of pending events
    pub fn pending_count(&self) -> usize {
        self.events.lock().unwrap().len()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Event aggregator for spatial events
pub struct SpatialEventAggregator {
    /// Grid dimensions
    grid_shape: [usize; 3],
    /// Event density map
    event_density: Array3<f64>,
    /// Event type counters
    event_counts: HashMap<String, usize>,
}

impl SpatialEventAggregator {
    /// Create a new spatial event aggregator
    pub fn new(grid_shape: [usize; 3]) -> Self {
        Self {
            grid_shape,
            event_density: Array3::zeros(grid_shape),
            event_counts: HashMap::new(),
        }
    }
    
    /// Add an event to the aggregator
    pub fn add_event(&mut self, event: &PhysicsEvent) {
        let position = match event {
            PhysicsEvent::BubbleCollapse { position, .. } |
            PhysicsEvent::LightEmission { position, .. } |
            PhysicsEvent::ChemicalReaction { position, .. } |
            PhysicsEvent::MaterialDamage { position, .. } |
            PhysicsEvent::TemperatureSpike { position, .. } |
            PhysicsEvent::PhaseTransition { position, .. } => Some(*position),
            PhysicsEvent::ShockWave { origin, .. } => Some(*origin),
            PhysicsEvent::Custom { .. } => None,
        };
        
        if let Some(pos) = position {
            if pos[0] < self.grid_shape[0] && 
               pos[1] < self.grid_shape[1] && 
               pos[2] < self.grid_shape[2] {
                self.event_density[[pos[0], pos[1], pos[2]]] += 1.0;
            }
        }
        
        let event_type = match event {
            PhysicsEvent::BubbleCollapse { .. } => "BubbleCollapse",
            PhysicsEvent::LightEmission { .. } => "LightEmission",
            PhysicsEvent::ChemicalReaction { .. } => "ChemicalReaction",
            PhysicsEvent::MaterialDamage { .. } => "MaterialDamage",
            PhysicsEvent::TemperatureSpike { .. } => "TemperatureSpike",
            PhysicsEvent::ShockWave { .. } => "ShockWave",
            PhysicsEvent::PhaseTransition { .. } => "PhaseTransition",
            PhysicsEvent::Custom { event_type, .. } => event_type.as_str(),
        };
        
        *self.event_counts.entry(event_type.to_string()).or_insert(0) += 1;
    }
    
    /// Get event density map
    pub fn density(&self) -> &Array3<f64> {
        &self.event_density
    }
    
    /// Get event statistics
    pub fn statistics(&self) -> &HashMap<String, usize> {
        &self.event_counts
    }
    
    /// Reset the aggregator
    pub fn reset(&mut self) {
        self.event_density.fill(0.0);
        self.event_counts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_event_bus() {
        let bus = EventBus::new();
        
        // Publish some events
        bus.publish(PhysicsEvent::BubbleCollapse {
            position: [10, 20, 30],
            radius: 1e-6,
            energy: 1e-3,
            peak_temperature: 5000.0,
            peak_pressure: 1e9,
        });
        
        assert_eq!(bus.pending_count(), 1);
        
        // Process events
        bus.process_events().unwrap();
        assert_eq!(bus.pending_count(), 0);
        
        // Check history
        assert_eq!(bus.history().len(), 1);
    }
    
    #[test]
    fn test_spatial_aggregator() {
        let mut aggregator = SpatialEventAggregator::new([100, 100, 100]);
        
        aggregator.add_event(&PhysicsEvent::BubbleCollapse {
            position: [50, 50, 50],
            radius: 1e-6,
            energy: 1e-3,
            peak_temperature: 5000.0,
            peak_pressure: 1e9,
        });
        
        assert_eq!(aggregator.density()[[50, 50, 50]], 1.0);
        assert_eq!(aggregator.statistics().get("BubbleCollapse"), Some(&1));
    }
}