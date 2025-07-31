// src/physics/core/mod.rs
//! Core infrastructure for the improved physics architecture
//! 
//! This module provides the fundamental building blocks for the physics system:
//! - Base traits and types
//! - Entity-Component System
//! - Event system
//! - Common utilities

pub mod effect;
pub mod entity;
pub mod event;
pub mod system;

// Re-export core types
pub use effect::{PhysicsEffect, EffectCategory, EffectId, EffectContext, EffectState};
pub use entity::{Entity, EntityId, EntityManager, Component};
pub use event::{PhysicsEvent, EventBus, EventHandler};
pub use system::{PhysicsSystem, SystemScheduler, SystemContext};

/// Common result type for physics operations
pub type PhysicsResult<T> = Result<T, crate::error::PhysicsError>;