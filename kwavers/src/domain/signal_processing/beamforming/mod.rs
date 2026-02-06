//! Beamforming Abstractions
//!
//! Defines domain-level beamforming interfaces that physics, analysis, and clinical layers
//! can implement. This ensures clean separation of concerns:
//!
//! - **Physics Layer**: Computes theoretical beamforming from acoustic field equations
//! - **Analysis Layer**: Provides signal processing algorithms (neural networks, adaptive, etc.)
//! - **Clinical Layer**: Applies beamforming for diagnosis and therapy monitoring
//!
//! The trait-based design allows multiple implementations without interdependencies.

pub mod config;
pub mod interface;
pub mod types;

pub use config::{BeamformingConfig, WindowFunction};
pub use interface::BeamformingProcessor;
pub use types::{BeamPattern, BeamformingResult};
