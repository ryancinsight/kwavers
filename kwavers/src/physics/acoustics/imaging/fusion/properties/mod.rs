//! Tissue property extraction from fused multi-modal data.
//!
//! This module provides methods for deriving tissue properties from fused
//! multi-modal imaging data, including tissue classification, oxygenation
//! estimation, and mechanical property characterization.

pub mod classification;
pub mod extractor;
pub mod mechanical;
pub mod oxygenation;

pub use classification::{classify_tissue_types, detect_regions_of_interest};
pub use extractor::extract_tissue_properties;
pub use mechanical::{compute_composite_stiffness, compute_tissue_density};
pub use oxygenation::compute_oxygenation_index;

#[cfg(test)]
mod tests;
