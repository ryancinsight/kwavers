//! Heterogeneous medium with spatially-varying properties
//!
//! **Deep Hierarchical Architecture**: Following Rust Book Ch.7 per senior engineering standards
//! **Evidence-Based Refactoring**: 479-line monolith → focused submodules per GRASP principles

// Core structure definition
pub mod core;

// Specialized trait implementations  
pub mod traits;

// Interpolation utilities
pub mod interpolation;

// Factory patterns for medium creation
pub mod factory;

// Legacy modules (maintained for compatibility)
pub mod constants;
pub mod tissue;

// Backward compatibility - use new modular structure
pub use core::HeterogeneousMedium;
pub use factory::TissueFactory;

// Maintain existing API compatibility
impl HeterogeneousMedium {
    /// Create a heterogeneous tissue medium (legacy API)
    ///
    /// **Migration Path**: Use `TissueFactory::create_tissue_medium()` for new code
    pub fn tissue(grid: &crate::grid::Grid) -> Self {
        TissueFactory::create_tissue_medium(grid)
    }
}
