//! Cloud Provider Implementations
//!
//! This module contains provider-specific implementations for cloud deployment
//! operations, organized by platform (AWS, GCP, Azure).
//!
//! # Architecture
//!
//! Each provider module implements the full deployment lifecycle:
//! - Model deployment to provider-specific ML platform
//! - Auto-scaling operations
//! - Deployment termination and cleanup
//!
//! Providers are implemented as separate modules to maintain clear separation
//! of concerns and allow conditional compilation based on feature flags.
//!
//! # Design Pattern
//!
//! Uses the Strategy pattern with provider-specific implementations:
//! - Common interface: Deploy, scale, terminate operations
//! - Provider-specific logic: AWS SageMaker, GCP Vertex AI, Azure ML
//! - Dependency inversion: Core service depends on abstractions, not implementations
//!
//! # Literature References
//!
//! - Gamma, E., et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software.
//!   Addison-Wesley. ISBN: 978-0201633610
//! - Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design.
//!   Prentice Hall. ISBN: 978-0134494166

#[cfg(all(feature = "pinn", feature = "api"))]
pub mod aws;

#[cfg(feature = "pinn")]
pub mod azure;

#[cfg(feature = "pinn")]
pub mod gcp;

// Re-export provider implementations for internal use
#[cfg(all(feature = "pinn", feature = "api"))]
pub use aws::*;

#[cfg(feature = "pinn")]
pub use azure::*;

#[cfg(feature = "pinn")]
pub use gcp::*;
