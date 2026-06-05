//! Literature validation: types, validator, and tests.

pub mod types;
pub mod validator;
#[cfg(test)]
mod tests;

pub use types::{treeby_2010, pinton_2009, LiteratureValidationResult, LiteratureValidationCase, ValidationMetric};
pub use validator::LiteratureValidator;
