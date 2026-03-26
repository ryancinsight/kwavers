pub mod consent;
pub mod demographics;
pub mod encounter;
pub mod profile;
pub mod system;
pub mod treatment;

pub use consent::{ConsentRecord, ConsentType};
pub use demographics::{PatientDemographics, PatientId};
pub use encounter::{ClinicalEncounter, ClinicalNote, EncounterId, EncounterType, VitalSigns};
pub use profile::{MedicalHistoryEntry, MedicationRecord, PatientMedicalProfile};
pub use system::PatientManagementSystem;
pub use treatment::{TreatmentPlan, TreatmentStatus};
