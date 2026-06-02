//! `AnalysisConservationLaw`, `ConservedQuantity`, and `ConservationResult` types.

/// Type of conservation law to verify.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalysisConservationLaw {
    /// Mass conservation: ∫ρ dV = constant.
    Mass,
    /// Momentum conservation: ∫ρu dV = constant (x, y, z components).
    Momentum,
    /// Energy conservation: ∫(KE + PE + TE) dV = constant.
    Energy,
    /// Charge conservation (for EM fields).
    Charge,
}

impl std::fmt::Display for AnalysisConservationLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mass => write!(f, "Mass"),
            Self::Momentum => write!(f, "Momentum"),
            Self::Energy => write!(f, "Energy"),
            Self::Charge => write!(f, "Charge"),
        }
    }
}

/// Conservation quantity at a single timestep.
#[derive(Debug, Clone)]
pub struct ConservedQuantity {
    /// Name of the quantity.
    pub name: String,
    /// Integral value (∫ quantity dV).
    pub integral: f64,
    /// Relative magnitude for normalization.
    pub magnitude: f64,
    /// Timestamp when measured.
    pub time: f64,
}

/// Result of conservation verification.
#[derive(Debug, Clone)]
pub struct ConservationResult {
    /// Conservation law verified.
    pub law: AnalysisConservationLaw,
    /// Initial quantity value.
    pub initial_value: f64,
    /// Current quantity value.
    pub current_value: f64,
    /// Absolute change.
    pub absolute_change: f64,
    /// Relative error (dimensionless).
    pub relative_error: f64,
    /// Tolerance used for verification.
    pub tolerance: f64,
    /// Whether conservation is satisfied.
    pub passed: bool,
    /// Error message if failed.
    pub error_message: Option<String>,
}
