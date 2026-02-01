//! Architecture Validation and Enforcement
//!
//! This module validates and enforces the deep vertical 9-layer architecture hierarchy
//! to prevent architectural drift and ensure clean separation of concerns.
//!
//! ## Layers (Strict Downward Dependencies)
//!
//! - **Layer 0 (Core)**: Error handling, time, constants, logging
//! - **Layer 1 (Math)**: Linear algebra, FFT, SIMD, numerics
//! - **Layer 2 (Domain)**: Grid, sensors, sources, boundaries, media, signals
//! - **Layer 3 (Physics)**: Acoustics, thermal, EM, optics, chemistry
//! - **Layer 4 (Solver)**: FDTD, PSTD, SEM, BEM, FEM, inverse, coupling
//! - **Layer 5 (Simulation)**: Orchestration, factories, backends
//! - **Layer 6 (Clinical)**: Therapy, imaging, safety, monitoring, patient management
//! - **Layer 7 (Analysis)**: Signal processing, ML, visualization, validation
//! - **Layer 8 (Infrastructure)**: I/O, API, GPU, runtime, cloud, hardware

pub mod layer_validation;

pub use layer_validation::{
    ArchitectureLayer, ArchitectureValidator, LayerViolation, ValidationResult, ValidationStats,
    ViolationType,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_layer_hierarchy() {
        // Verify layer ordering
        assert!(ArchitectureLayer::Solver > ArchitectureLayer::Physics);
        assert!(ArchitectureLayer::Physics > ArchitectureLayer::Domain);
        assert!(ArchitectureLayer::Clinical > ArchitectureLayer::Simulation);
    }

    #[test]
    fn test_architecture_validator_creation() {
        let _validator = ArchitectureValidator::new();
        // Validator created successfully
    }
}
