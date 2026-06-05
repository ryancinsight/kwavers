//! Drug Payload and Release Kinetics
//!
//! Implementation of drug encapsulation and release mechanisms for therapeutic
//! microbubbles used in targeted drug delivery applications.

pub mod loading_mode;
pub mod payload;
#[cfg(test)]
mod tests;

pub use loading_mode::DrugLoadingMode;
pub use payload::DrugPayload;
