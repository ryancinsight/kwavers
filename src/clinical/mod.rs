//! Clinical Systems and Workflows
//!
//! This module provides clinical-grade systems for medical ultrasound imaging,
//! including real-time workflows, diagnostic decision support, and regulatory
//! compliance frameworks.
//!
//! ## Clinical Applications
//!
//! - **Diagnostic Imaging**: Complete examination workflows
//! - **Interventional Guidance**: Real-time procedure support
//! - **Treatment Planning**: AI-enhanced therapeutic guidance
//! - **Quality Assurance**: Automated clinical validation
//!
//! ## Regulatory Compliance
//!
//! - **FDA 510(k)**: Medical device regulatory requirements
//! - **IEC 60601-2-37**: Ultrasound safety standards
//! - **Clinical Trials**: GCP-compliant data collection
//! - **Quality Management**: ISO 13485 compliance frameworks

pub mod workflows;

// Re-export main clinical components
pub use workflows::{
    ClinicalWorkflowOrchestrator, ClinicalWorkflowConfig, ClinicalApplication,
    WorkflowPriority, QualityPreference, ClinicalProtocol, WorkflowState,
    ClinicalExaminationResult, DiagnosticRecommendation, DiagnosticUrgency,
    PerformanceMetrics,
};
