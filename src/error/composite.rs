//! Composite error types

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;
use crate::error::KwaversError;

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
    pub errors: Vec<KwaversError>,
}

impl MultiError {
    /// Create a new empty MultiError
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Add an error to the collection
    pub fn add(&mut self, error: KwaversError) {
        self.errors.push(error);
    }

    /// Check if there are any errors
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get the number of errors
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Convert to a Result, returning the MultiError if any errors exist
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
