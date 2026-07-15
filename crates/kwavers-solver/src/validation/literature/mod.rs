//! Literature validation: types, validator, and tests.

#[cfg(test)]
mod tests;
pub mod types;
pub mod validator;

pub use types::{
    pinton_2009, treeby_2010, LiteratureValidationCase, LiteratureValidationResult,
    ValidationMetric,
};
pub use validator::LiteratureValidator;
