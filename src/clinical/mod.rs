//! Clinical module
//!
//! This module provides application-level workflows for clinical imaging and therapy.
//! It uses the `physics` and `solver` modules to implement clinical scenarios.

pub mod imaging;
pub mod safety; // IEC 60601-2-37 compliance framework
pub mod therapy;

pub use imaging::{
    ClinicalApplication, ClinicalExaminationResult, ClinicalProtocol, ClinicalWorkflowConfig,
    ClinicalWorkflowOrchestrator, DiagnosticRecommendation, DiagnosticUrgency, PerformanceMetrics,
    QualityPreference, WorkflowPriority, WorkflowState,
};
pub use safety::{
    AuditEntry, ComplianceReport, ComplianceResult, ComplianceValidator, DoseController, Interlock,
    InterlockSystem, SafetyAuditLogger, SafetyEventType, SafetyLevel, SafetyLimits, SafetyMonitor,
    SafetyViolation, SystemConfiguration, TreatmentRecord,
};
pub use therapy::{
    AcousticField, AcousticTherapyParams, AcousticWaveSolver, PatientParameters, RiskOrgan,
    SafetyLimits as TherapySafetyLimits, SafetyMetrics, SafetyStatus, TargetVolume,
    TherapyIntegrationOrchestrator, TherapyModality, TherapySessionConfig, TherapySessionState,
    TissuePropertyMap, TissueType,
};
