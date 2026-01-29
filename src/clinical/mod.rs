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
    mechanical_index::{
        MechanicalIndexCalculator, MechanicalIndexResult, SafetyStatus as MISafetyStatus,
        TissueType as MITissueType,
    },
    AuditEntry, ComplianceReport, ComplianceResult, ComplianceValidator, DoseController, Interlock,
    InterlockSystem, SafetyAuditLogger, SafetyEventType, SafetyLevel, SafetyLimits, SafetyMonitor,
    SafetyViolation, SystemConfiguration, TreatmentRecord,
};
pub use therapy::{TherapyMechanism, TherapyModality, TherapyParameters, TreatmentMetrics};

// FIXME: These types are referenced but not yet implemented in the therapy module
// They need to be added during architectural refactoring:
// - AcousticField
// - AcousticTherapyParams
// - AcousticWaveSolver
// - PatientParameters
// - RiskOrgan
// - SafetyLimits (therapy-specific)
// - SafetyMetrics
// - SafetyStatus
// - TargetVolume
// - TherapyIntegrationOrchestrator
// - TherapySessionConfig
// - TherapySessionState
// - TissuePropertyMap
// - TissueType
