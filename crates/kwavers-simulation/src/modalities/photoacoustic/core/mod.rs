//! Core Photoacoustic Simulator Module
//!
//! Orchestrates the complete photoacoustic imaging simulation pipeline:
//! optical fluence computation, initial pressure generation, acoustic wave propagation,
//! and image reconstruction.

pub mod acoustic;
pub mod simulator;

pub use simulator::PhotoacousticSimulator;
