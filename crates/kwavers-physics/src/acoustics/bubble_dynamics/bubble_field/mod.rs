//! Bubble field management
//!
//! Manages collections of bubbles in the simulation domain

pub mod cloud;
pub mod core;
pub mod distributions;

pub use kwavers_field::BubbleStateFields;
pub use cloud::BubbleCloud;
pub use core::{BubbleField, BubbleFieldStats};
pub use distributions::{BubbleFieldSizeDistribution, SpatialDistribution};

#[cfg(test)]
mod tests;
