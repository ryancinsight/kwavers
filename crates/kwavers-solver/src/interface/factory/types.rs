//! Factory configuration and error types.

use crate::config::SolverType;

/// Factory configuration with performance targets
#[derive(Debug, Clone)]
pub struct FactoryConfiguration {
    /// Maximum memory budget (bytes)
    pub memory_budget: usize,
    /// Required solver features
    pub required_features: Vec<String>,
    /// Target performance ratio (vs reference)
    pub performance_target: f64,
    /// Enable solver auto-selection
    pub enable_auto_selection: bool,
}

impl Default for FactoryConfiguration {
    fn default() -> Self {
        Self {
            memory_budget: usize::MAX,
            required_features: Vec::new(),
            performance_target: 1.0,
            enable_auto_selection: true,
        }
    }
}

/// Factory error types with structured diagnostics
#[derive(Debug)]
pub enum FactoryError {
    /// Requested solver type not available
    SolverTypeNotSupported(SolverType),
    /// Configuration invalid or incomplete
    InvalidConfiguration(String),
    /// Resource constraints violated (memory, etc.)
    ResourceExceeded { requested: usize, available: usize },
    /// Factory not initialized properly
    NotInitialized,
    /// Internal error during solver construction
    Internal(String),
}

impl std::fmt::Display for FactoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SolverTypeNotSupported(t) => {
                write!(f, "Solver type not supported: {:?}", t)
            }
            Self::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            Self::ResourceExceeded {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Resource exceeded: requested {} bytes, available {} bytes",
                    requested, available
                )
            }
            Self::NotInitialized => write!(f, "Factory not initialized"),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for FactoryError {}

/// Conversion from FactoryError to KwaversError
impl From<FactoryError> for kwavers_core::error::KwaversError {
    fn from(err: FactoryError) -> Self {
        Self::InternalError(err.to_string())
    }
}
