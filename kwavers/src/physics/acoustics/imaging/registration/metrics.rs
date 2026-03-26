
/// Quality metrics for temporal synchronization
#[derive(Debug, Clone)]
pub struct TemporalQualityMetrics {
    /// Root mean square timing error \[seconds\]
    pub rms_timing_error: f64,
    /// Maximum timing deviation \[seconds\]
    pub max_timing_deviation: f64,
    /// Phase lock stability factor [0-1]
    pub phase_lock_stability: f64,
    /// Synchronization success rate [0-1]
    pub sync_success_rate: f64,
}

/// Comprehensive quality metrics for registration accuracy
#[derive(Debug, Clone)]
pub struct RegistrationQualityMetrics {
    /// Fiducial registration error \[mm\]
    pub fre: Option<f64>,
    /// Target registration error \[mm\]
    pub tre: Option<f64>,
    /// Mutual information between registered images
    pub mutual_information: f64,
    /// Correlation coefficient between registered images
    pub correlation_coefficient: f64,
    /// Normalized cross-correlation
    pub normalized_cross_correlation: f64,
    /// Registration convergence flag
    pub converged: bool,
    /// Number of iterations for optimization
    pub iterations: usize,
    /// Final cost function value
    pub final_cost: f64,
}
