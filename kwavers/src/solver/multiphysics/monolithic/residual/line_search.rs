use super::super::coupler::MonolithicCoupler;
use super::super::residual_metric::norm_squared;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::field::UnifiedFieldType;
use ndarray::Array3;

impl MonolithicCoupler {
    /// Find an adaptive Newton step size that reduces the residual norm.
    ///
    /// Candidate states are exactly `u + alpha * du` with
    /// `alpha_k = alpha_max / 2^k`.  The solver evaluates five candidates and
    /// accepts the first one satisfying `||F(u + alpha*du)||² < 0.81 ||F(u)||²`.
    /// The squared comparison is algebraically equivalent to the norm
    /// comparison for nonnegative residual norms and avoids one square root per
    /// candidate. If all candidates fail, the method returns the final evaluated
    /// alpha so the caller never applies an untested Newton step.
    ///
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if `alpha_max` is outside `(0, 1]`.
    /// - Propagates any residual-evaluation error from candidate states.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::solver::multiphysics::monolithic) fn line_search(
        &mut self,
        u: &Array3<f64>,
        du: &Array3<f64>,
        f: &Array3<f64>,
        u_prev: &Array3<f64>,
        dt: f64,
        dims: (usize, usize, usize),
        field_order: &[UnifiedFieldType],
    ) -> KwaversResult<f64> {
        let f_norm_squared = norm_squared(f);
        let sufficient_decrease_squared = 0.9 * 0.9 * f_norm_squared;
        let max_alpha = self.newton_config.line_search_parameter;
        if !max_alpha.is_finite() || max_alpha <= 0.0 || max_alpha > 1.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "NewtonKrylovConfig::line_search_parameter".to_owned(),
                value: max_alpha,
                reason: "must be finite and in (0, 1]".to_owned(),
            }));
        }
        let mut trial_state = self
            .line_search_state_scratch
            .take()
            .filter(|scratch| scratch.dim() == u.dim())
            .unwrap_or_else(|| Array3::zeros(u.dim()));

        const BACKTRACK_TRIALS: usize = 5;
        let mut last_alpha = max_alpha;
        for k in 0..BACKTRACK_TRIALS {
            let alpha = max_alpha * 2.0_f64.powi(-(k as i32));
            last_alpha = alpha;
            trial_state.assign(u);
            trial_state.zip_mut_with(du, |candidate, &delta| {
                *candidate += alpha * delta;
            });
            let f_new = match self.compute_residual(&trial_state, u_prev, dt, dims, field_order) {
                Ok(residual) => residual,
                Err(error) => {
                    self.line_search_state_scratch = Some(trial_state);
                    return Err(error);
                }
            };
            let f_new_norm_squared = norm_squared(&f_new);

            if f_new_norm_squared < sufficient_decrease_squared {
                self.line_search_state_scratch = Some(trial_state);
                return Ok(alpha);
            }
        }

        self.line_search_state_scratch = Some(trial_state);
        Ok(last_alpha)
    }
}
