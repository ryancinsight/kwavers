//! Neural Beamforming Backend Implementations
//!
//! This module contains concrete implementations of neural beamforming providers
//! for various backends (Burn, TensorFlow, PyTorch, etc.).
//!
//! ## Architecture
//!
//! Each backend implements the `PinnBeamformingProvider` trait defined in the
//! parent neural module, allowing different machine learning frameworks to be
//! used interchangeably for neural beamforming.
//!
//! ## Available Backends
//!
//! - **Burn** (`burn_adapter`): Rust-native PINN framework with automatic differentiation
//! - Additional backends can be added without changing the public interface

#[cfg(feature = "pinn")]
pub mod burn_adapter;

#[cfg(feature = "pinn")]
pub use burn_adapter::{create_burn_beamforming_provider, BurnPinnBeamformingAdapter};
