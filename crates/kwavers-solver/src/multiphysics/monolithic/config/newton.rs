use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};

/// Newton-Krylov method configuration.
#[derive(Debug, Clone)]
pub struct NewtonKrylovConfig {
    /// Maximum Newton iterations.
    pub max_newton_iterations: usize,

    /// Newton tolerance: `||F(u)|| < tolerance`.
    pub newton_tolerance: f64,

    /// Maximum backtracking trial step `alpha_max` in `(0, 1]`.
    ///
    /// Adaptive Newton line search evaluates candidate steps
    /// `alpha_k = alpha_max / 2^k`. Values outside `(0, 1]` are rejected
    /// before residual evaluation because they either disable progress
    /// (`alpha <= 0`) or allow steps larger than the Newton correction
    /// (`alpha > 1`).
    pub line_search_parameter: f64,

    /// Enable residual-checked adaptive step size.
    pub adaptive_step_size: bool,

    /// Emit Newton/GMRES diagnostic log messages.
    pub verbose: bool,
}

impl Default for NewtonKrylovConfig {
    fn default() -> Self {
        Self {
            max_newton_iterations: 20,
            newton_tolerance: 1e-6,
            line_search_parameter: 1.0,
            adaptive_step_size: true,
            verbose: false,
        }
    }
}

impl NewtonKrylovConfig {
    /// Validate Newton-Krylov numerical controls.
    ///
    /// The checks are scalar and run once per coupled step. They prevent
    /// impossible loop bounds, nonfinite convergence predicates, and invalid
    /// line-search step domains before any full-state residual work is done.
    ///
    /// # Errors
    /// - Returns [`crate::KwaversError::Validation`] for invalid iteration count,
    ///   Newton tolerance, or line-search alpha bound.
    pub(in crate::multiphysics::monolithic) fn validate(&self) -> KwaversResult<()> {
        if self.max_newton_iterations == 0 {
            return Err(KwaversError::Validation(
                ValidationError::InvalidParameter {
                    parameter: "NewtonKrylovConfig::max_newton_iterations".to_owned(),
                    reason: "must be greater than zero".to_owned(),
                },
            ));
        }

        if !self.newton_tolerance.is_finite() || self.newton_tolerance <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "NewtonKrylovConfig::newton_tolerance".to_owned(),
                value: self.newton_tolerance,
                reason: "must be finite and positive".to_owned(),
            }));
        }

        let alpha = self.line_search_parameter;
        if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "NewtonKrylovConfig::line_search_parameter".to_owned(),
                value: alpha,
                reason: "must be finite and in (0, 1]".to_owned(),
            }));
        }

        Ok(())
    }
}
