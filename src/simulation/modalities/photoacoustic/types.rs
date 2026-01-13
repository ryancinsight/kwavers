//! Type Definitions for Photoacoustic Imaging Module
//!
//! This module provides type definitions and re-exports for the photoacoustic imaging module.
//! Following Domain-Driven Design principles, the canonical type definitions reside in the
//! clinical domain module, and this module re-exports them for convenient access.
//!
//! ## Design Philosophy
//!
//! - **Single Source of Truth**: Core types defined in `crate::clinical::imaging::photoacoustic`
//! - **Domain-Driven Design**: Types follow ubiquitous language from photoacoustic imaging domain
//! - **Separation of Concerns**: Domain models separated from simulation infrastructure
//!
//! ## Re-exported Types
//!
//! - [`PhotoacousticParameters`]: Simulation configuration parameters
//! - [`PhotoacousticResult`]: Complete simulation results with metadata
//! - [`InitialPressure`]: Initial pressure distribution from optical absorption
//! - [`PhotoacousticOpticalProperties`]: Tissue-specific optical properties
//! - [`OpticalPropertyData`]: Low-level optical property data structure
//!
//! ## Usage
//!
//! ```rust,no_run
//! use kwavers::simulation::modalities::photoacoustic::types::*;
//!
//! // Create default parameters
//! let params = PhotoacousticParameters::default();
//!
//! // Access wavelength configuration
//! println!("Wavelengths: {:?}", params.wavelengths);
//! ```

// Re-export types from clinical domain (SSOT)
pub use crate::clinical::imaging::photoacoustic::{
    InitialPressure, PhotoacousticOpticalProperties, PhotoacousticParameters, PhotoacousticResult,
};

// Re-export optical property data from domain layer
pub use crate::domain::medium::properties::OpticalPropertyData;
