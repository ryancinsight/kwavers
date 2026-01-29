//! Imaging Domain Module
//!
//! This module contains imaging-related domain types.
//!
//! ## Architecture
//!
//! Provides three layers:
//! 1. **Domain Models** (ultrasound/, photoacoustic/) - Data structures for imaging concepts
//! 2. **Orchestration Interfaces** (ceus_orchestrator.rs) - Traits for orchestration logic
//! 3. **Implementations** - Physics and simulation layers implement these traits
//!
//! This ensures clinical layer depends only on domain abstractions, not on implementation details.
//!
//! ## See Also
//!
//! - `clinical::imaging` - Application-level imaging workflows and types
//! - `physics::foundations` - Physics specifications for wave equations
//! - `domain::sensor` - Sensor primitives for signal detection
//! - `domain::source` - Source primitives for wave generation

pub mod ceus_orchestrator;
pub mod photoacoustic;
pub mod ultrasound;

pub use ceus_orchestrator::{CEUSOrchestrator, CEUSOrchestrators};
