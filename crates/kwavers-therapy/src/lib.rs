#![doc = include_str!("../README.md")]

mod parallel;
pub mod patient_management; // Electronic health record and clinical workflow management
pub mod regulatory; // FDA 510(k) submission and compliance
pub mod safety; // IEC 60601-2-37 compliance framework
pub mod therapy;

pub use patient_management::{
    ClinicalEncounter, ClinicalNote, ConsentRecord, ConsentType, EncounterId, EncounterType,
    MedicalHistoryEntry, MedicationRecord, PatientDemographics, PatientId, PatientManagementSystem,
    PatientMedicalProfile, PatientTreatmentPlan, TreatmentStatus, VitalSigns,
};
pub use regulatory::{
    ClinicalEvidence, DeviceClass, DeviceDescription, PerformanceTest, PredicateDevice, RiskRecord,
    SubmissionDocument,
};
pub use safety::{
    mechanical_index::{
        MechanicalIndexCalculator, MechanicalIndexResult, MechanicalIndexSafetyStatus,
        MechanicalIndexTissueType,
    },
    AuditEntry, AuditSafetyEventType, ClinicalSafetyLevel, ClinicalSafetyLimits,
    ClinicalSafetyMonitor, ComplianceResult, ComplianceValidator, DoseController, Interlock,
    InterlockSystem, SafetyAuditLogger, SafetyComplianceReport, SafetyViolation,
    SystemConfiguration, TreatmentRecord,
};
pub use therapy::ClinicalTherapyParameters;
