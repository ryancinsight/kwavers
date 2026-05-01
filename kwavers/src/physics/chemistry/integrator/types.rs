//! Public result and error types for radical integration.

/// Statistics returned by a successful `RadicalIntegrator::integrate` call.
#[derive(Debug, Clone)]
pub struct IntegrationStats {
    /// Accepted steps where the solution advanced.
    pub steps_accepted: usize,
    /// Rejected steps repeated with a smaller step size.
    pub steps_rejected: usize,
    /// Time reached by the integrator.
    pub final_time: f64,
}

/// Chemistry integration error.
#[derive(Debug, Clone, PartialEq)]
pub enum IntegratorError {
    /// Step size collapsed below `h_min`; the system is too stiff for explicit integration.
    StepSizeTooSmall { h: f64, t: f64 },
}

impl std::fmt::Display for IntegratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepSizeTooSmall { h, t } => {
                write!(f, "step size {h:.3e} < h_min at t = {t:.3e} s")
            }
        }
    }
}
