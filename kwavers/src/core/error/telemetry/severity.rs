use crate::core::error::KwaversError;

/// Error severity levels for alerting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorSeverity {
    /// Critical: immediate intervention required.
    Critical,
    /// High: service degradation, attention required.
    High,
    /// Medium: degraded but contained.
    Medium,
    /// Low: informational or deferred remediation.
    Low,
}

impl ErrorSeverity {
    /// String representation used by metrics exports.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
        }
    }

    /// Default alert threshold in errors per minute.
    #[must_use]
    pub const fn threshold(&self) -> f64 {
        match self {
            Self::Critical => 10.0,
            Self::High => 5.0,
            Self::Medium => 1.0,
            Self::Low => 0.5,
        }
    }

    #[must_use]
    pub const fn all() -> [Self; 4] {
        [Self::Critical, Self::High, Self::Medium, Self::Low]
    }
}

impl From<&KwaversError> for ErrorSeverity {
    fn from(error: &KwaversError) -> Self {
        match error {
            KwaversError::ResourceLimitExceeded { .. } => Self::Critical,
            KwaversError::System(_) => Self::Critical,
            KwaversError::GpuError(_) => Self::Critical,
            KwaversError::ConcurrencyError { .. } => Self::Critical,
            KwaversError::InternalError(_) => Self::Critical,

            KwaversError::Validation(_) => Self::High,
            KwaversError::Numerical(_) => Self::High,
            KwaversError::Physics(_) => Self::High,
            KwaversError::Grid(_) => Self::High,
            KwaversError::Medium(_) => Self::High,
            KwaversError::Field(_) => Self::High,
            KwaversError::Shape(_) => Self::High,
            KwaversError::DimensionMismatch(_) => Self::High,
            KwaversError::InvalidInput(_) => Self::High,
            KwaversError::MultipleErrors(_) => Self::High,
            KwaversError::Other(_) => Self::High,

            KwaversError::Data(_) => Self::Medium,
            KwaversError::Config(_) => Self::Medium,
            KwaversError::Io(_) => Self::Medium,
            #[cfg(feature = "nifti")]
            KwaversError::Nifti(_) => Self::Medium,

            KwaversError::NotImplemented(_) => Self::Low,
            KwaversError::FeatureNotAvailable(_) => Self::Low,
            KwaversError::PerformanceError(_) => Self::Low,
            KwaversError::Visualization { .. } => Self::Low,
        }
    }
}

#[must_use]
pub(crate) fn error_type_name(error: &KwaversError) -> &'static str {
    match error {
        KwaversError::Grid(_) => "grid",
        KwaversError::Medium(_) => "medium",
        KwaversError::Physics(_) => "physics",
        KwaversError::Data(_) => "data",
        KwaversError::Config(_) => "config",
        KwaversError::Numerical(_) => "numerical",
        KwaversError::Field(_) => "field",
        KwaversError::System(_) => "system",
        KwaversError::Validation(_) => "validation",
        KwaversError::InternalError(_) => "internal",
        KwaversError::DimensionMismatch(_) => "dimension_mismatch",
        KwaversError::FeatureNotAvailable(_) => "feature_not_available",
        KwaversError::PerformanceError(_) => "performance",
        KwaversError::Io(_) => "io",
        #[cfg(feature = "nifti")]
        KwaversError::Nifti(_) => "nifti",
        KwaversError::NotImplemented(_) => "not_implemented",
        KwaversError::GpuError(_) => "gpu",
        KwaversError::ResourceLimitExceeded { .. } => "resource_limit",
        KwaversError::InvalidInput(_) => "invalid_input",
        KwaversError::MultipleErrors(_) => "multiple_errors",
        KwaversError::ConcurrencyError { .. } => "concurrency",
        KwaversError::Visualization { .. } => "visualization",
        KwaversError::Shape(_) => "shape",
        KwaversError::Other(_) => "other",
    }
}

#[must_use]
pub(crate) const fn all_error_type_names() -> &'static [&'static str] {
    &[
        "grid",
        "medium",
        "physics",
        "data",
        "config",
        "numerical",
        "field",
        "system",
        "validation",
        "internal",
        "dimension_mismatch",
        "feature_not_available",
        "performance",
        "io",
        #[cfg(feature = "nifti")]
        "nifti",
        "not_implemented",
        "gpu",
        "resource_limit",
        "invalid_input",
        "multiple_errors",
        "concurrency",
        "visualization",
        "shape",
        "other",
    ]
}
