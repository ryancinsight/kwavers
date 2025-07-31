// src/physics/core/entity.rs
//! Entity-Component System for physics effects
//! 
//! This module implements a data-oriented ECS architecture that allows
//! efficient processing of physics effects on grid cells.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;

/// Unique identifier for an entity (grid cell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId {
    /// Grid indices [x, y, z]
    pub indices: [usize; 3],
    /// Unique ID for disambiguation
    pub id: u64,
}

impl EntityId {
    /// Create a new entity ID from grid indices
    pub fn from_indices(x: usize, y: usize, z: usize) -> Self {
        Self {
            indices: [x, y, z],
            id: ((x as u64) << 40) | ((y as u64) << 20) | (z as u64),
        }
    }
    
    /// Get x index
    pub fn x(&self) -> usize {
        self.indices[0]
    }
    
    /// Get y index
    pub fn y(&self) -> usize {
        self.indices[1]
    }
    
    /// Get z index
    pub fn z(&self) -> usize {
        self.indices[2]
    }
}

/// Base trait for components
pub trait Component: Any + Send + Sync + Debug {
    /// Get component type name
    fn type_name(&self) -> &'static str;
    
    /// Clone the component as a boxed trait object
    fn clone_box(&self) -> Box<dyn Component>;
    
    /// Get the component as Any for downcasting
    fn as_any(&self) -> &dyn Any;
    
    /// Get the component as mutable Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Macro to implement Component trait for a type
#[macro_export]
macro_rules! impl_component {
    ($type:ty) => {
        impl Component for $type {
            fn type_name(&self) -> &'static str {
                std::any::type_name::<$type>()
            }
            
            fn clone_box(&self) -> Box<dyn Component> {
                Box::new(self.clone())
            }
            
            fn as_any(&self) -> &dyn Any {
                self
            }
            
            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }
    };
}

/// Entity representing a grid cell with components
pub struct Entity {
    /// Unique identifier
    pub id: EntityId,
    /// Components attached to this entity
    components: HashMap<TypeId, Box<dyn Component>>,
}

impl Entity {
    /// Create a new entity
    pub fn new(id: EntityId) -> Self {
        Self {
            id,
            components: HashMap::new(),
        }
    }
    
    /// Add a component to the entity
    pub fn add_component<T: Component + 'static>(&mut self, component: T) {
        self.components.insert(TypeId::of::<T>(), Box::new(component));
    }
    
    /// Get a component by type
    pub fn get_component<T: Component + 'static>(&self) -> Option<&T> {
        self.components
            .get(&TypeId::of::<T>())
            .and_then(|c| c.as_any().downcast_ref::<T>())
    }
    
    /// Get a mutable component by type
    pub fn get_component_mut<T: Component + 'static>(&mut self) -> Option<&mut T> {
        self.components
            .get_mut(&TypeId::of::<T>())
            .and_then(|c| c.as_any_mut().downcast_mut::<T>())
    }
    
    /// Check if entity has a component
    pub fn has_component<T: Component + 'static>(&self) -> bool {
        self.components.contains_key(&TypeId::of::<T>())
    }
    
    /// Remove a component
    pub fn remove_component<T: Component + 'static>(&mut self) -> Option<Box<dyn Component>> {
        self.components.remove(&TypeId::of::<T>())
    }
    
    /// Get all component type IDs
    pub fn component_types(&self) -> Vec<TypeId> {
        self.components.keys().cloned().collect()
    }
}

/// Entity manager for the physics system
pub struct EntityManager {
    /// All entities indexed by ID
    entities: HashMap<EntityId, Entity>,
    /// Grid dimensions
    grid_shape: [usize; 3],
    /// Component type registry
    component_types: HashMap<TypeId, String>,
}

impl EntityManager {
    /// Create a new entity manager
    pub fn new(grid_shape: [usize; 3]) -> Self {
        Self {
            entities: HashMap::new(),
            grid_shape,
            component_types: HashMap::new(),
        }
    }
    
    /// Register a component type
    pub fn register_component_type<T: Component + 'static>(&mut self, name: &str) {
        self.component_types.insert(TypeId::of::<T>(), name.to_string());
    }
    
    /// Create or get an entity at grid position
    pub fn get_or_create_entity(&mut self, x: usize, y: usize, z: usize) -> &mut Entity {
        let id = EntityId::from_indices(x, y, z);
        self.entities.entry(id).or_insert_with(|| Entity::new(id))
    }
    
    /// Get an entity by ID
    pub fn get_entity(&self, id: &EntityId) -> Option<&Entity> {
        self.entities.get(id)
    }
    
    /// Get a mutable entity by ID
    pub fn get_entity_mut(&mut self, id: &EntityId) -> Option<&mut Entity> {
        self.entities.get_mut(id)
    }
    
    /// Get entities with a specific component type
    pub fn entities_with_component<T: Component + 'static>(&self) -> Vec<&Entity> {
        self.entities
            .values()
            .filter(|e| e.has_component::<T>())
            .collect()
    }
    
    /// Get mutable entities with a specific component type
    pub fn entities_with_component_mut<T: Component + 'static>(&mut self) -> Vec<&mut Entity> {
        self.entities
            .values_mut()
            .filter(|e| e.has_component::<T>())
            .collect()
    }
    
    /// Remove all entities
    pub fn clear(&mut self) {
        self.entities.clear();
    }
    
    /// Get number of entities
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
    
    /// Get statistics about component usage
    pub fn component_statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        for entity in self.entities.values() {
            for type_id in entity.component_types() {
                if let Some(name) = self.component_types.get(&type_id) {
                    *stats.entry(name.clone()).or_insert(0) += 1;
                }
            }
        }
        
        stats
    }
}

// Example components for physics effects

/// Acoustic properties component
#[derive(Debug, Clone)]
pub struct AcousticComponent {
    pub pressure: f64,
    pub velocity: [f64; 3],
    pub density: f64,
}

impl_component!(AcousticComponent);

/// Bubble dynamics component
#[derive(Debug, Clone)]
pub struct BubbleComponent {
    pub radius: f64,
    pub wall_velocity: f64,
    pub gas_pressure: f64,
    pub temperature: f64,
}

impl_component!(BubbleComponent);

/// Thermal component
#[derive(Debug, Clone)]
pub struct ThermalComponent {
    pub temperature: f64,
    pub heat_flux: [f64; 3],
    pub thermal_conductivity: f64,
}

impl_component!(ThermalComponent);

/// Optical component
#[derive(Debug, Clone)]
pub struct OpticalComponent {
    pub intensity: f64,
    pub spectrum: Vec<f64>,
    pub refractive_index: f64,
}

impl_component!(OpticalComponent);

/// Chemical component
#[derive(Debug, Clone)]
pub struct ChemicalComponent {
    pub species_concentrations: HashMap<String, f64>,
    pub reaction_rates: HashMap<String, f64>,
    pub ph: f64,
}

impl_component!(ChemicalComponent);

/// Material damage component
#[derive(Debug, Clone)]
pub struct DamageComponent {
    pub erosion_depth: f64,
    pub fatigue_cycles: u32,
    pub damage_rate: f64,
}

impl_component!(DamageComponent);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_entity_id() {
        let id1 = EntityId::from_indices(10, 20, 30);
        let id2 = EntityId::from_indices(10, 20, 30);
        let id3 = EntityId::from_indices(11, 20, 30);
        
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(id1.x(), 10);
        assert_eq!(id1.y(), 20);
        assert_eq!(id1.z(), 30);
    }
    
    #[test]
    fn test_entity_components() {
        let mut entity = Entity::new(EntityId::from_indices(0, 0, 0));
        
        let acoustic = AcousticComponent {
            pressure: 1e5,
            velocity: [0.0, 0.0, 0.0],
            density: 1000.0,
        };
        
        entity.add_component(acoustic.clone());
        
        assert!(entity.has_component::<AcousticComponent>());
        assert!(!entity.has_component::<BubbleComponent>());
        
        let retrieved = entity.get_component::<AcousticComponent>().unwrap();
        assert_eq!(retrieved.pressure, 1e5);
    }
    
    #[test]
    fn test_entity_manager() {
        let mut manager = EntityManager::new([100, 100, 100]);
        manager.register_component_type::<AcousticComponent>("acoustic");
        
        let entity = manager.get_or_create_entity(50, 50, 50);
        entity.add_component(AcousticComponent {
            pressure: 1e5,
            velocity: [0.0, 0.0, 0.0],
            density: 1000.0,
        });
        
        assert_eq!(manager.entity_count(), 1);
        assert_eq!(manager.entities_with_component::<AcousticComponent>().len(), 1);
    }
}