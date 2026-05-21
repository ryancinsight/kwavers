//! Cloud Deployment Configuration
//!
//! Configuration types for cloud PINN deployments with validation,
//! defaults, and builder patterns.

#[cfg(test)]
mod tests;
pub mod types;

pub use types::{AlertThresholds, AutoScalingConfig, CloudMonitoringConfig, DeploymentConfig};
