//! Simulation Modalities Module
//!
//! This module provides high-level simulation interfaces for different imaging modalities.
//! Each modality integrates domain models, physics solvers, and reconstruction algorithms
//! into a unified simulation pipeline.
//!
//! ## Available Modalities
//!
//! - **Photoacoustic Imaging**: Combines optical excitation with acoustic detection
//!
//! ## Architecture
//!
//! Each modality module follows Clean Architecture principles:
//! - Domain layer: Core types and domain models
//! - Application layer: Simulation orchestration
//! - Infrastructure layer: Physics solvers and reconstruction algorithms
//! - Interface layer: Public API

pub mod photoacoustic;

// Re-export main types from photoacoustic module
pub use photoacoustic::{
    InitialPressure, PhotoacousticOpticalProperties, PhotoacousticParameters, PhotoacousticResult,
    PhotoacousticSimulator,
};
