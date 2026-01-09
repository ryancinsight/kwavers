//! Error context utilities

use std::fmt;

#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub file: String,
    pub line: u32,
    pub function: String,
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.function)
    }
}
