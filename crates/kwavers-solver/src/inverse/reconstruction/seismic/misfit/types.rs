//! `MisfitType` enum and `MisfitFunction` dispatcher.

use kwavers_core::error::KwaversResult;
use leto::Array2;

/// Type of misfit function for FWI
#[derive(Debug, Clone, Copy)]
pub enum MisfitType {
    /// L2 norm (least squares)
    L2Norm,
    /// L1 norm (robust to outliers)
    L1Norm,
    /// Envelope misfit (for cycle-skipping mitigation)
    Envelope,
    /// Phase-only misfit
    Phase,
    /// Normalized cross-correlation
    Correlation,
    /// Wasserstein distance (optimal transport)
    Wasserstein,
}

/// Misfit function calculator
#[derive(Debug)]
pub struct MisfitFunction {
    pub(super) misfit_type: MisfitType,
}

impl MisfitFunction {
    /// Create a new misfit function calculator
    #[must_use]
    pub fn new(misfit_type: MisfitType) -> Self {
        Self { misfit_type }
    }

    /// Compute adjoint source from residual (direct interface for L1/L2 norms).
    ///
    /// For envelope and phase misfits, use `compute_adjoint_source` instead
    /// for proper Hilbert transform-based adjoint computation per Fichtner et al. (2008).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn adjoint_source(&self, residual: &Array2<f64>) -> Array2<f64> {
        match self.misfit_type {
            MisfitType::L2Norm => residual.clone(),
            MisfitType::L1Norm => residual.mapv(f64::signum),
            MisfitType::Envelope
            | MisfitType::Phase
            | MisfitType::Correlation
            | MisfitType::Wasserstein => residual.clone(),
        }
    }

    /// Compute misfit between observed and synthetic data.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<f64> {
        match self.misfit_type {
            MisfitType::L2Norm => self.l2_misfit(observed, synthetic),
            MisfitType::L1Norm => self.l1_misfit(observed, synthetic),
            MisfitType::Envelope => self.envelope_misfit(observed, synthetic),
            MisfitType::Phase => self.phase_misfit(observed, synthetic),
            MisfitType::Correlation => self.correlation_misfit(observed, synthetic),
            MisfitType::Wasserstein => self.wasserstein_misfit(observed, synthetic),
        }
    }

    /// Compute adjoint source for gradient calculation.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        match self.misfit_type {
            MisfitType::L2Norm => Ok(synthetic - observed),
            MisfitType::L1Norm => self.l1_adjoint_source(observed, synthetic),
            MisfitType::Envelope => self.envelope_adjoint_source(observed, synthetic),
            MisfitType::Phase => self.phase_adjoint_source(observed, synthetic),
            MisfitType::Correlation => self.correlation_adjoint_source(observed, synthetic),
            MisfitType::Wasserstein => self.wasserstein_adjoint_source(observed, synthetic),
        }
    }
}
