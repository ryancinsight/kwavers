
//! Interpolation utilities for multi-grid and multi-physics coupling
//!
//! Provides conservative and non-conservative interpolation methods for
//! transferring fields between different grids, essential for:
//! - Multi-GPU domain decomposition with overlap regions
//! - Adaptive mesh refinement (AMR)
//! - Multiphysics coupling on different spatial resolutions
//! - Real-time simulation with dynamic grid adaptation

pub mod conservative;

pub use conservative::{ConservationMode, ConservativeInterpolator};
