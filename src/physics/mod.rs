//! Physics module for k-Wave simulation
//!
//! This module contains the core physics implementations for acoustic wave simulation,
//! including material properties, wave equations, and numerical constants.

pub mod bubble_dynamics;
pub mod cavitation_control;
pub mod chemistry;
pub mod constants;
pub mod field_indices;
pub mod field_mapping;
pub mod imaging;
pub mod mechanics;
pub mod optics;
pub mod plugin;
pub mod skull;
pub mod sonoluminescence_detector;
pub mod state;
pub mod therapy;
pub mod thermal;
pub mod traits;
pub mod transcranial;
pub mod transducer;
pub mod validation;
pub mod wave_propagation;
