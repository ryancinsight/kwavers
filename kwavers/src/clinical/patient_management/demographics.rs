use crate::core::error::{KwaversError, KwaversResult};

/// Patient identifier (anonymized for HIPAA compliance in production)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PatientId(String);

impl PatientId {
    /// Create a new patient ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a new anonymous patient ID
    pub fn generate_anonymous() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(format!("PAT_{:08}", id))
    }

    /// Get the underlying ID string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Patient demographics and basic information
#[derive(Debug, Clone)]
pub struct PatientDemographics {
    /// Patient identifier
    pub patient_id: PatientId,
    /// Patient full name (encrypted in production)
    pub name: String,
    /// Date of birth as ISO 8601 string
    pub date_of_birth: String,
    /// Biological sex: M, F, Other
    pub sex: char,
    /// Weight in kilograms
    pub weight_kg: f64,
    /// Height in centimeters
    pub height_cm: f64,
    /// Primary contact phone
    pub contact_phone: String,
    /// Primary email
    pub contact_email: String,
    /// Emergency contact name
    pub emergency_contact: String,
    /// Emergency contact phone
    pub emergency_contact_phone: String,
    /// Medical record number
    pub medical_record_number: String,
}

impl PatientDemographics {
    /// Calculate BMI from weight and height
    pub fn calculate_bmi(&self) -> f64 {
        let height_m = self.height_cm / 100.0;
        self.weight_kg / (height_m * height_m)
    }

    /// Validate demographics data
    pub fn validate(&self) -> KwaversResult<()> {
        if self.name.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Patient name cannot be empty".to_string(),
            ));
        }

        if self.weight_kg <= 0.0 || self.weight_kg > 500.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid patient weight".to_string(),
            ));
        }

        if self.height_cm <= 50.0 || self.height_cm > 300.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid patient height".to_string(),
            ));
        }

        if !matches!(self.sex, 'M' | 'F' | 'O') {
            return Err(KwaversError::InvalidInput(
                "Invalid biological sex value".to_string(),
            ));
        }

        Ok(())
    }
}
