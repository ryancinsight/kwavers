//! Composite error types

use crate::error::KwaversError;
use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeError {
    pub errors: Vec<String>,
}

impl fmt::Display for CompositeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Multiple errors: {:?}", self.errors)
    }
}

impl StdError for CompositeError {}

/// A collection of multiple validation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiError {
    pub errors: Vec<String>,
}

impl MultiError {
    /// Create a new empty MultiError
    #[must_use]
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Add an error to the collection
    pub fn add(&mut self, error: KwaversError) {
        self.errors.push(error.to_string());
    }

    /// Check if there are any errors
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get the number of errors
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Convert to a Result, returning the MultiError if any errors exist
    /// # Errors
    /// - Returns [`KwaversError::MultipleErrors`] if the precondition for a MultipleErrors-class constraint is violated.
    ///
    pub fn into_result(self) -> Result<(), KwaversError> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(KwaversError::MultipleErrors(self))
        }
    }
}

impl fmt::Display for MultiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Multiple validation errors:")?;
        for (i, error) in self.errors.iter().enumerate() {
            writeln!(f, "  {}: {}", i + 1, error)?;
        }
        Ok(())
    }
}

impl StdError for MultiError {}

impl Default for MultiError {
    fn default() -> Self {
        Self::new()
    }
}
