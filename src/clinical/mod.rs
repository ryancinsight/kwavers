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
pub mod swe_3d_workflows;
pub mod therapy_integration;

// Re-export main clinical components
pub use workflows::{
    ClinicalWorkflowOrchestrator, ClinicalWorkflowConfig, ClinicalApplication,
    WorkflowPriority, QualityPreference, ClinicalProtocol, WorkflowState,
    ClinicalExaminationResult, DiagnosticRecommendation, DiagnosticUrgency,
    PerformanceMetrics,
};

// Re-export 3D SWE clinical workflow components
pub use swe_3d_workflows::{
    VolumetricROI, ElasticityMap3D, ElasticityMap2D, VolumetricStatistics,
    ClinicalDecisionSupport, TissueReference, LiverFibrosisStage, FibrosisStage,
    BreastLesionClassification, ClassificationConfidence, MultiPlanarReconstruction,
    SliceOrientation, SlicePositions,
};

// Re-export therapy integration components
pub use therapy_integration::{
    TherapyIntegrationOrchestrator, TherapySessionConfig, TherapySessionState,
    AcousticField, AcousticWaveSolver,
};
