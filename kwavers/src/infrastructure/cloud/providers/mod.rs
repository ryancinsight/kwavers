// ! Cloud Provider Implementations
//!
//! This module contains provider-specific implementations for cloud deployment
//! operations for ultrasound simulation model deployment.
//!
//! # Current Status
//!
//! **Implemented:**
//! - AWS SageMaker (complete, production-ready)
//!
//! **Not yet implemented (removed incomplete stubs):**
//! - Azure ML (incomplete implementation removed)
//! - GCP Vertex AI (incomplete implementation removed)
//!
//! **Not yet implemented (multi-cloud extension points):**
//!
//! - Azure ML: workspace integration, Blob Storage, Functions deployment,
//!   auto-scaling, monitoring.
//! - GCP Vertex AI: model endpoint deployment, Cloud Storage, Cloud Functions,
//!   GKE-based Kubernetes deployment.
//!
//! # Architecture
//!
//! Each provider module implements the full deployment lifecycle:
//! - Model deployment to provider-specific ML platform
//! - Auto-scaling operations based on traffic/load
//! - Deployment monitoring and health checks
//! - Deployment termination and cleanup
//!
//! Providers use the Strategy pattern with provider-specific implementations:
//! - Common interface: Deploy, scale, monitor, terminate operations
//! - Provider-specific logic: AWS SageMaker, (future: GCP Vertex AI, Azure ML)
//! - Dependency inversion: Core service depends on abstractions
//!
//! # Literature References
//!
//! - Gamma, E., et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software.
//!   Addison-Wesley. ISBN: 978-0201633610
//! - Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design.
//!   Prentice Hall. ISBN: 978-0134494166

// AWS SageMaker provider (production-ready)
#[cfg(all(feature = "pinn", feature = "cloud-aws"))]
pub mod aws;

// Explicit re-exports of AWS provider functions
#[cfg(all(feature = "pinn", feature = "cloud-aws"))]
pub use aws::{deploy_to_aws, scale_aws_deployment, terminate_aws_deployment};
