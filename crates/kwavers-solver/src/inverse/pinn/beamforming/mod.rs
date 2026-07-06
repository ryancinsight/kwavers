//! PINN-Based Beamforming Implementations
//!
//! This module provides concrete PINN implementations for neural beamforming,
//! bridging the solver layer's PINN capabilities with the analysis layer's
//! beamforming interface.
//!
//! ## Architecture
//!
//! ```text
//! Analysis Layer (Layer 7)
//!     ↓ depends on trait
//! PinnBeamformingProvider (interface in analysis)
//!     ↑ implemented by
//! Solver Layer (Layer 4) - this module
//!     BurnPinnBeamformingAdapter (concrete implementation)
//!     ↓ uses
//! Burn PINN Models (BurnPINN1DWave, BurnPINNConfig)
//! ```
//!
//! This design respects Clean Architecture's dependency rule:
//! - Analysis layer defines the interface
//! - Solver layer implements the interface
//! - No upward dependencies (Layer 4 never imports from Layer 7)
//!
//! ## Available Implementations
//!
//! - **Burn Backend** (`burn_adapter`): Rust-native differentiable programming
//!   with CPU execution today and provider-generic GPU execution pending the
//!   Coeus + Hephaestus migration

#[cfg(feature = "pinn")]
pub mod burn_adapter;

#[cfg(feature = "pinn")]
pub use burn_adapter::{create_burn_beamforming_provider, BurnPinnBeamformingAdapter};
