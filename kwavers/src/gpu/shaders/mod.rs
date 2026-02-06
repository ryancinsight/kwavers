//! GPU compute shaders for physics simulations
//!
//! Modular WGSL shaders following SLAP principle

pub mod absorption;
pub mod fdtd;
pub mod kspace;
pub mod neural_network;
pub mod nonlinear;

// Re-export shader sources
pub use absorption::ABSORPTION_SHADER;
pub use fdtd::FDTD_PRESSURE_SHADER;
pub use kspace::KSPACE_PROPAGATE_SHADER;
pub use neural_network::NEURAL_NETWORK_SHADER;
pub use nonlinear::NONLINEAR_PROPAGATION_SHADER;
