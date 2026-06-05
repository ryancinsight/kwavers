//! DICOM image loader.

mod dicom_ritk;
pub mod loader;
#[cfg(test)]
mod tests;
pub mod types;

pub use loader::DicomImageLoader;
pub use types::{DicomMetadata, DicomModality};
