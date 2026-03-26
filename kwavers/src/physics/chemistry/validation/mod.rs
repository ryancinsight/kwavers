//! Chemistry kinetics validation against literature
//!
//! Validates reaction rate constants and thermodynamic parameters
//! against peer-reviewed sources.

pub mod arrhenius;
pub mod kinetics_database;
pub mod literature;

pub use arrhenius::ArrheniusValidator;
pub use kinetics_database::{ValidatedKinetics, ValidationResult};
pub use literature::LiteratureValue;

#[cfg(test)]
mod tests;
