//! Real-Time Clinical Workflow Pipelines
//!
//! This module provides integrated clinical workflows that orchestrate multi-modal
//! imaging acquisition, real-time processing, fusion, and AI-enhanced diagnostics.
//! Designed for clinical environments requiring fast, reliable, and comprehensive
//! diagnostic capabilities.
//!
//! ## Workflow Architecture
//!
//! - **Acquisition Pipeline**: Coordinated multi-modal data acquisition
//! - **Real-Time Processing**: GPU-accelerated parallel processing streams
//! - **Intelligent Fusion**: Adaptive multi-modal fusion with quality optimization
//! - **Clinical Decision Support**: AI-enhanced diagnostic recommendations
//! - **Quality Assurance**: Automated quality checks and artifact detection

pub mod analysis;
pub mod blood_oxygenation;
pub mod config;
pub mod orchestrator;
pub mod results;
pub mod simulation;
pub mod state;

// Re-export core types to maintain API compatibility
pub use config::{
    ClinicalApplication, ClinicalProtocol, ClinicalWorkflowConfig, QualityPreference,
    WorkflowPriority,
};
pub use orchestrator::ClinicalWorkflowOrchestrator;
pub use results::{
    ClinicalExaminationResult, DiagnosticRecommendation, DiagnosticUrgency, PerformanceMetrics,
};
pub use state::WorkflowState;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clinical_workflow_creation() {
        let config = ClinicalWorkflowConfig::default();
        let workflow = ClinicalWorkflowOrchestrator::new(config);
        assert!(workflow.is_ok());

        let workflow = workflow.unwrap();
        match workflow.get_state() {
            WorkflowState::Initializing => {}
            _ => panic!("Expected Initializing state"),
        }
    }

    #[test]
    fn test_workflow_execution() {
        let config = ClinicalWorkflowConfig {
            real_time_enabled: false, // Disable real-time for testing
            ..Default::default()
        };
        let mut workflow = ClinicalWorkflowOrchestrator::new(config).unwrap();

        let result = workflow.execute_examination("patient_001");
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.patient_id, "patient_001");
        assert!(result.confidence_score >= 0.0 && result.confidence_score <= 100.0);
        assert!(!result.diagnostic_recommendations.is_empty());
    }

    #[test]
    fn test_realtime_performance_check() {
        let config = ClinicalWorkflowConfig {
            max_latency_ms: 1000,
            real_time_enabled: true,
            ..Default::default()
        };
        let workflow = ClinicalWorkflowOrchestrator::new(config).unwrap();

        // Should pass since no execution has occurred yet
        assert!(workflow.check_realtime_performance());
    }

    #[test]
    fn test_diagnostic_recommendations() {
        let workflow = ClinicalWorkflowOrchestrator::new(ClinicalWorkflowConfig::default());
        // Note: This would need proper setup for testing diagnostic recommendations
        // For now, just test that workflow creation succeeds
        assert!(workflow.is_ok());
    }
}
