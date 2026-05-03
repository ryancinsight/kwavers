//! Architecture Layer Validation and Enforcement.
//!
//! Automated validation to prevent architectural drift and layer violations
//! in the deep vertical 9-layer hierarchy.

#[cfg(test)]
mod tests;
pub mod types;
pub mod validator;

pub use types::{
    ArchitectureLayer, LayerViolation, ModuleDependency, ValidationResult, ValidationStats,
    ViolationType,
};
pub use validator::ArchitectureValidator;
