//! CT Image Loader — NIFTI Format Support
//!
//! CT images are fundamental domain concepts used in both diagnostic and therapeutic
//! ultrasound applications. CT loading belongs in the domain layer where all solvers
//! can access it.
//!
//! ## Reference
//!
//! Marquet et al. (2009) "Non-invasive transcranial ultrasound therapy based on a 3D CT scan"

pub mod loader;
#[cfg(test)]
mod tests;
pub mod types;

pub use loader::CTImageLoader;
pub use types::CTMetadata;
