//! Bubble field management
//!
//! Manages collections of bubbles in the simulation domain

pub mod cloud;
pub mod core;
pub mod distributions;

pub use crate::domain::field::BubbleStateFields;
pub use cloud::BubbleCloud;
pub use core::{BubbleField, BubbleFieldStats};
pub use distributions::{SizeDistribution, SpatialDistribution};

#[cfg(test)]
mod tests;
