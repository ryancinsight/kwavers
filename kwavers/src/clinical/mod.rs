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

// NOTE: Therapy integration framework types are available but not re-exported here
// due to ongoing architectural refactoring. Access them directly via:
//
//   use kwavers::clinical::therapy::therapy_integration::{
//       TherapyIntegrationOrchestrator, TherapySessionConfig, AcousticTherapyParams,
//       SafetyLimits, PatientParameters, TissuePropertyMap, TargetVolume, etc.
//   };
//
// The therapy_integration module provides a comprehensive clinical therapy framework
// with support for HIFU, histotripsy, lithotripsy, and other modalities. It includes:
// - TherapyIntegrationOrchestrator: Main therapy orchestration
// - AcousticField, AcousticWaveSolver: Acoustic simulation infrastructure
// - SafetyMetrics, SafetyStatus, SafetyController: Real-time safety monitoring
// - TissuePropertyMap, TissueType: Patient-specific tissue modeling
//
// Status: Implementation complete, undergoing final integration testing (Sprint 214)
