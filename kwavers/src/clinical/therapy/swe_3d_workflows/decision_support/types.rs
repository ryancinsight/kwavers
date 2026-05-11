//! Clinical classification result types for 3D SWE decision support.

/// Tissue reference ranges for clinical comparison.
#[derive(Debug, Clone)]
pub struct TissueReference {
    /// Mean Young's modulus (kPa).
    pub mean_modulus: f64,
    /// Standard deviation (kPa).
    pub std_modulus: f64,
    /// Minimum expected modulus (kPa).
    pub min_modulus: f64,
    /// Maximum expected modulus (kPa).
    pub max_modulus: f64,
}

/// Liver fibrosis classification result.
#[derive(Debug, Clone)]
pub struct LiverFibrosisStage {
    /// METAVIR fibrosis stage.
    pub stage: FibrosisStage,
    /// Mean stiffness in kPa.
    pub mean_stiffness_kpa: f64,
    /// Classification confidence.
    pub confidence: ClassificationConfidence,
    /// Quality score (0–1).
    pub quality_score: f64,
}

/// METAVIR fibrosis stages.
#[derive(Debug, Clone, Copy)]
pub enum FibrosisStage {
    /// No fibrosis (F0) or mild fibrosis (F1).
    F0F1,
    /// Moderate fibrosis (F2).
    F2,
    /// Severe fibrosis (F3).
    F3,
    /// Cirrhosis (F4).
    F4,
}

/// Breast lesion classification result.
#[derive(Debug, Clone)]
pub struct BreastLesionClassification {
    /// BI-RADS category (2–5).
    pub birads_category: u8,
    /// Estimated probability of malignancy (0–1).
    pub estimated_malignancy_probability: f64,
    /// Mean stiffness in kPa.
    pub mean_stiffness_kpa: f64,
    /// Classification confidence.
    pub confidence: ClassificationConfidence,
    /// Quality score (0–1).
    pub quality_score: f64,
}

/// Classification confidence levels.
#[derive(Debug, Clone, Copy)]
pub enum ClassificationConfidence {
    /// High confidence (>80% accuracy expected).
    High,
    /// Medium confidence (60–80% accuracy expected).
    Medium,
    /// Low confidence (<60% accuracy expected).
    Low,
}
