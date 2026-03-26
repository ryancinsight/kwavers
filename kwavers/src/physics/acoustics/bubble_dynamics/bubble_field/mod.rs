//! Bubble field management
//!
//! Manages collections of bubbles in the simulation domain

pub mod cloud;
pub mod core;
pub mod distributions;

pub use cloud::BubbleCloud;
pub use core::{BubbleField, BubbleFieldStats};
pub use distributions::{SizeDistribution, SpatialDistribution};
pub use crate::domain::field::BubbleStateFields;

#[cfg(test)]
mod tests;
