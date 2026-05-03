//! Cloud PINN service orchestrator.

pub mod orchestrator;
#[cfg(test)]
mod tests;

pub use orchestrator::CloudPINNService;
