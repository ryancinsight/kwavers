//! Modular SIMD auto-detection with architecture-specific implementations
//!
//! This module follows GRASP principles by separating architecture-specific
//! concerns into dedicated modules:
//! - Information Expert: Each architecture owns its implementation knowledge
//! - Single Responsibility: Separate modules for each CPU architecture
//! - High Cohesion: Related SIMD operations grouped by architecture
//!
//! References:
//! - IEEE TSE 2022: "Understanding Memory and Thread Safety Practices"
//! - Intel SIMD Programming Guide (2023)
//! - ARM NEON Programmer's Guide

pub mod capability;
pub mod dispatcher;
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// Re-export main types
pub use capability::SimdCapability;
pub use dispatcher::SimdAuto;