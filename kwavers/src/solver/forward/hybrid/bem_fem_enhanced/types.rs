//! Types for enhanced BEM-FEM coupling results and quality metrics.

/// Interface quality metrics.
#[derive(Debug, Clone)]
pub struct InterfaceQuality {
    /// Number of interface elements.
    pub num_elements: usize,

    /// Average element size.
    pub avg_element_size: f64,

    /// Estimated interface error.
    pub estimated_error: f64,

    /// Condition number of coupling matrix.
    pub condition_number: Option<f64>,

    /// Maximum local error.
    pub max_local_error: f64,

    /// Spurious resonance detection (true if detected).
    pub spurious_resonance_detected: bool,
}

/// Single refinement step information.
#[derive(Debug, Clone)]
pub struct RefinementStep {
    /// Refinement level (0 = initial mesh).
    pub level: usize,

    /// Number of elements at this level.
    pub num_elements: usize,

    /// Estimated error.
    pub estimated_error: f64,

    /// Number of refined elements.
    pub num_refined_elements: usize,
}

/// Validation result for enhanced BEM-FEM coupling.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Test frequency (Hz).
    pub frequency: f64,

    /// Spurious resonance detection.
    pub spurious_resonance_detected: bool,

    /// Was Burton-Miller formulation used.
    pub burton_miller_used: bool,

    /// Estimated interface error.
    pub interface_error: f64,

    /// Number of adaptive refinement levels used.
    pub refinement_levels: usize,

    /// Validation computation time (seconds).
    pub validation_time: f64,

    /// Interface quality metrics.
    pub interface_quality: InterfaceQuality,
}

impl ValidationResult {
    /// Check if validation passed (no spurious resonances, error below threshold).
    #[must_use] 
    pub fn passed(&self, error_threshold: f64) -> bool {
        !self.spurious_resonance_detected && self.interface_error < error_threshold
    }

    /// Get summary as string.
    #[must_use] 
    pub fn summary(&self) -> String {
        format!(
            "Enhanced BEM-FEM Validation @ {:.1} Hz:\n  \
             Burton-Miller: {}\n  \
             Spurious Resonance: {}\n  \
             Interface Error: {:.2e}\n  \
             Refinement Levels: {}\n  \
             Computation Time: {:.3}s",
            self.frequency,
            self.burton_miller_used,
            self.spurious_resonance_detected,
            self.interface_error,
            self.refinement_levels,
            self.validation_time
        )
    }
}
