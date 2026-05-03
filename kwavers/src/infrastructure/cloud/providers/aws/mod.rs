//! AWS Cloud Provider Implementation.
//!
//! Implements cloud deployment operations for Amazon Web Services (AWS)
//! using SageMaker, ELB, and Application Auto Scaling.

pub mod deploy;
pub mod scale;
pub mod terminate;
#[cfg(test)]
mod tests;

#[cfg(all(feature = "pinn", feature = "cloud-aws"))]
pub use deploy::deploy_to_aws;
#[cfg(all(feature = "pinn", feature = "cloud-aws"))]
pub use scale::scale_aws_deployment;
#[cfg(all(feature = "pinn", feature = "cloud-aws"))]
pub use terminate::terminate_aws_deployment;
