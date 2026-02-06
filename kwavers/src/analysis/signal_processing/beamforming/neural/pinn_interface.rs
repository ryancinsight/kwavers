//! PINN-Beamforming Interface (re-export shim)
//!
//! The canonical definitions now live in [`crate::solver::interface::pinn_beamforming`].
//! This module re-exports everything for backward compatibility with analysis-layer consumers.

pub use crate::solver::interface::pinn_beamforming::*;
