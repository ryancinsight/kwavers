//! Clinical therapy and care-delivery layer for kwavers.
//!
//! Application-level therapeutic workflows (HIFU/LIFU/histotripsy/lithotripsy
//! planning, theranostic image-guided guidance, dose and safety monitoring) plus
//! the clinical-care modules they depend on (IEC 60601-2-37 safety, FDA 510(k)
//! regulatory, patient management). Orchestrates the lower layers
//! (`physics`/`solver`/`simulation`/`analysis`); contains no numerical kernels.
//!
//! Extracted from the former `clinical` module (ADR 011); the sibling diagnostic
//! imaging workflows live in the `kwavers-diagnostics` crate.

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
