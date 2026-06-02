//! CPML configuration and validation

mod cpml_config;
mod per_dimension;
#[cfg(test)]
mod tests;

pub use cpml_config::CPMLConfig;
pub use per_dimension::{PerDimensionAlpha, PerDimensionPML};
