//! 3D SAFT — SRP submodules.

pub mod config;
pub mod processor;
#[cfg(test)]
mod tests;

pub use config::SaftConfig;
pub use processor::SaftProcessor;
