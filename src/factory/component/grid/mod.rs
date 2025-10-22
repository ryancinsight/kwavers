//! Grid component factory - Deep hierarchical organization
//!
//! Follows Information Expert principle with domain-focused decomposition:
//! - Configuration: Grid parameter validation and setup
//! - Creator: Actual grid instantiation logic
//! - Validator: Specialized validation rules

pub mod config;
pub mod creator;
pub mod validator;

// Re-export main types for backward compatibility
pub use config::GridConfig;
pub use creator::GridCreator;
pub use validator::GridValidator;

/// Main grid factory interface - maintains existing API
#[derive(Debug)]
pub struct GridFactory;

impl GridFactory {
    /// Create grid from configuration
    /// Delegates to specialized components following Information Expert
    pub fn create_grid(config: &GridConfig) -> crate::error::KwaversResult<crate::grid::Grid> {
        // Validate through specialized validator
        GridValidator::validate(config)?;

        // Create through specialized creator
        GridCreator::create(config)
    }
}
