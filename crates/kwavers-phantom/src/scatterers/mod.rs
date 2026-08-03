//! Point scatterers and synthetic RF synthesis.

mod types;
pub use types::{PointScatterer, ScattererCloud, RfSynthesisConfig, TransmitWavefront};

#[cfg(test)]
mod tests;
