//! Domain Layer
//!
//! This module contains the fundamental domain configurations and types that are shared
//! across the entire system. It sits at the bottom of the dependency graph (above core).

pub mod boundary;
pub mod core;
pub mod field;
pub mod grid;
pub mod imaging;
pub mod math;
pub mod medium;
pub mod plugin;
pub mod sensor;
pub mod signal;
pub mod source;
pub mod therapy;
