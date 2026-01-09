//! Clinical module
//!
//! This module provides application-level workflows for clinical imaging and therapy.
//! It uses the `physics` and `solver` modules to implement clinical scenarios.

pub mod imaging;
pub mod therapy;

pub use imaging::*;
pub use therapy::*;
