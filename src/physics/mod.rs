//! Physics module for k-Wave simulation
//!
//! This module contains the core physics implementations for acoustic wave simulation,
//! including material properties, wave equations, and numerical constants.

pub mod constants;
pub mod mechanics;
pub mod chemistry;
pub mod optics;
pub mod thermal;
pub mod wave_propagation;
pub mod traits;
pub mod plugin;
pub mod state;
pub mod field_indices;
pub mod field_mapping;
pub mod bubble_dynamics;
pub mod sonoluminescence_detector;
pub mod validation;
pub mod imaging;