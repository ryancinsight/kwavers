//! Heterogeneous media handling with Gibbs phenomenon mitigation
//!
//! This module implements state-of-the-art techniques for handling sharp interfaces
//! in heterogeneous media, preventing spurious oscillations (Gibbs phenomenon).
//!
//! # References
//! - Tabei et al. (2002). "A k-space method for coupled first-order acoustic propagation equations." JASA
//! - Pinton et al. (2009). "A heterogeneous nonlinear attenuating full-wave model of ultrasound." IEEE UFFC

pub mod config;
pub mod handler;
pub mod interface_detection;
pub mod pressure_velocity_split;
pub mod smoothing;

pub use config::{HeterogeneousConfig, SmoothingMethod};
pub use handler::HeterogeneousHandler;
pub use pressure_velocity_split::PressureVelocitySplit;
