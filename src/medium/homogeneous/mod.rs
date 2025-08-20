//! Homogeneous medium module - uniform properties throughout spatial domain

mod float_key;
mod absorption_cache;
mod core;

pub use core::HomogeneousMedium;

// Re-export for backward compatibility (will be removed)
pub(crate) use float_key::FloatKey;
pub(crate) use absorption_cache::AbsorptionCache;