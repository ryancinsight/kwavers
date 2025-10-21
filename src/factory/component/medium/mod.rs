//! Medium component factory - Deep hierarchical organization
//!
//! Domain-driven decomposition following SOLID principles:
//! - Types: Medium type definitions and configurations
//! - Validation: Specialized validation rules for medium properties  
//! - Builder: Complex medium construction logic

pub mod builder;
pub mod types;
pub mod validation;

// Re-export main types
pub use builder::MediumBuilder;
pub use types::{MediumConfig, MediumType};
pub use validation::MediumValidator;

/// Main medium factory interface
#[derive(Debug)]
pub struct MediumFactory;

impl MediumFactory {
    /// Create medium from configuration
    pub fn create_medium(
        config: &MediumConfig,
        grid: &crate::grid::Grid,
    ) -> crate::error::KwaversResult<Box<dyn crate::medium::Medium>> {
        // Validate through specialized validator
        MediumValidator::validate(config)?;

        // Build through specialized builder
        MediumBuilder::build(config, grid)
    }
}
