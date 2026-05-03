//! Core distributed neural beamforming processor — SRP submodules.

pub mod processor;
#[cfg(test)]
mod tests;

#[cfg(feature = "pinn")]
pub use processor::{DistributedNeuralBeamformingProcessor, FaultToleranceState};
