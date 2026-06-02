//! Spherical-cap element layouts for focused bowl sources.
//!
//! This module owns reusable focused-cap geometry. Clinical adapters may choose
//! abdominal, transcranial, or other placement policy, but element placement on
//! a focused spherical bowl is a source-domain concern.

mod layout;

#[cfg(test)]
mod tests;

pub use layout::{SphericalCapConfig, SphericalCapElement, SphericalCapLayout};
