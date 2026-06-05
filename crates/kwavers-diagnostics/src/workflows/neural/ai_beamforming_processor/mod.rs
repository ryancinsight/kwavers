//! Neural Beamforming Processor — SRP submodules.

pub mod processor;
#[cfg(test)]
mod tests;
pub mod trait_engine;

pub use processor::AIEnhancedBeamformingProcessor;
pub use trait_engine::PinnInferenceEngine;
