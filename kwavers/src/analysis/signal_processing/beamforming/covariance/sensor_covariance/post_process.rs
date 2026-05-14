use crate::core::error::{KwaversError, KwaversResult};

/// Covariance post-processing strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum CovariancePostProcess {
    None,
    /// Apply shrinkage toward a scaled identity target:
    /// `R' = (1 - α) R + α μ I`, where `μ = tr(R)/M` and `α ∈ [0,1]`.
    ShrinkageToIdentity {
        alpha: f64,
    },
    /// Apply spatial smoothing (subarray averaging) for coherent source decorrelation.
    SpatialSmoothing {
        subarray_size: usize,
    },
    /// Apply shrinkage then spatial smoothing.
    ShrinkageThenSpatialSmoothing {
        alpha: f64,
        subarray_size: usize,
    },
}

impl Default for CovariancePostProcess {
    fn default() -> Self {
        Self::ShrinkageToIdentity { alpha: 0.05 }
    }
}

impl CovariancePostProcess {
    /// Validate.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        match self {
            Self::None => Ok(()),
            Self::ShrinkageToIdentity { alpha } => {
                if !alpha.is_finite() || *alpha < 0.0 || *alpha > 1.0 {
                    return Err(KwaversError::InvalidInput(
                        "CovariancePostProcess::ShrinkageToIdentity: alpha must be finite and in [0,1]".to_owned(),
                    ));
                }
                Ok(())
            }
            Self::SpatialSmoothing { subarray_size } => {
                if *subarray_size == 0 {
                    return Err(KwaversError::InvalidInput(
                        "CovariancePostProcess::SpatialSmoothing: subarray_size must be >= 1"
                            .to_owned(),
                    ));
                }
                Ok(())
            }
            Self::ShrinkageThenSpatialSmoothing {
                alpha,
                subarray_size,
            } => {
                if !alpha.is_finite() || *alpha < 0.0 || *alpha > 1.0 {
                    return Err(KwaversError::InvalidInput(
                        "CovariancePostProcess::ShrinkageThenSpatialSmoothing: alpha must be finite and in [0,1]".to_owned(),
                    ));
                }
                if *subarray_size == 0 {
                    return Err(KwaversError::InvalidInput(
                        "CovariancePostProcess::ShrinkageThenSpatialSmoothing: subarray_size must be >= 1".to_owned(),
                    ));
                }
                Ok(())
            }
        }
    }
}
