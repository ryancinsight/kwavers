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
//! **Not Implemented (Removed for Code Quality):**
//! - Azure ML (incomplete, removed - see TODO_AUDIT below)
//! - GCP Vertex AI (incomplete, removed - see TODO_AUDIT below)
//!
//! TODO_AUDIT: P1 - Azure ML Provider - Implement complete Azure Machine Learning deployment
//! DEPENDS ON: infra/cloud/providers/azure.rs (removed), Azure SDK integration
//! MISSING: Azure ML workspace integration, Azure Blob Storage for model artifacts
//! MISSING: Azure Functions serverless deployment, auto-scaling policies
//! MISSING: Azure Load Balancer configuration, monitoring and logging
//! SEVERITY: HIGH (blocks multi-cloud deployment strategy)
//! REFERENCES: Azure ML Documentation (https://docs.microsoft.com/en-us/azure/machine-learning/)
//! REFERENCES: Lakshmanan et al. (2020) Machine Learning Design Patterns, O'Reilly
//!
//! TODO_AUDIT: P1 - GCP Vertex AI Provider - Implement complete Google Cloud Platform deployment
//! DEPENDS ON: infra/cloud/providers/gcp.rs (removed), GCP SDK integration
//! MISSING: Vertex AI model deployment, Google Cloud Storage integration
//! MISSING: Cloud Functions deployment, auto-scaling based on load metrics
//! MISSING: Cloud Load Balancing, Kubernetes deployment support
//! SEVERITY: HIGH (blocks multi-cloud deployment strategy)
//! REFERENCES: GCP Vertex AI Documentation (https://cloud.google.com/vertex-ai/docs)
//! REFERENCES: Sato, D. (2019) Machine Learning Operations, O'Reilly
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
#[cfg(all(feature = "pinn", feature = "api"))]
pub mod aws;

// Re-export AWS provider implementation
#[cfg(all(feature = "pinn", feature = "api"))]
pub use aws::*;
