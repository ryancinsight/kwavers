//! Clinical module
//!
//! This module provides application-level workflows for clinical imaging and therapy.
//! It uses the `physics` and `solver` modules to implement clinical scenarios.

pub mod imaging;
pub mod patient_management; // Electronic health record and clinical workflow management
pub mod regulatory_documentation; // FDA 510(k) submission and compliance
pub mod safety; // IEC 60601-2-37 compliance framework
pub mod therapy;

pub use imaging::{
    ClinicalApplication, ClinicalExaminationResult, ClinicalProtocol, ClinicalWorkflowConfig,
    ClinicalWorkflowOrchestrator, DiagnosticRecommendation, DiagnosticUrgency, PerformanceMetrics,
    QualityPreference, WorkflowPriority, WorkflowState,
};
pub use patient_management::{
    ClinicalEncounter, ClinicalNote, ConsentRecord, ConsentType, EncounterId, EncounterType,
    MedicalHistoryEntry, MedicationRecord, PatientDemographics, PatientId, PatientManagementSystem,
    PatientMedicalProfile, TreatmentPlan, TreatmentStatus, VitalSigns,
};
pub use regulatory_documentation::{
    ClinicalEvidence, DeviceClass, DeviceDescription, PerformanceTest, PredicateDevice, RiskRecord,
    SubmissionDocument,
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
