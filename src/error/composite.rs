//! Composite error types

use std::error::Error as StdError;
use std::fmt;
use serde::{Deserialize, Serialize};

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
