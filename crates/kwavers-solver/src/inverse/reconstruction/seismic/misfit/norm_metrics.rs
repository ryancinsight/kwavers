//! L2, L1, and cross-correlation misfit metrics.

use super::types::MisfitFunction;
use kwavers_core::error::KwaversResult;
use leto::Array2;

impl MisfitFunction {
    /// L2 norm misfit: 0.5 * ||d_obs − d_syn||²
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn l2_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let sum_sq: f64 = synthetic
            .iter()
            .zip(observed.iter())
            .map(|(&s, &o)| {
                let d = s - o;
                d * d
            })
            .sum();
        Ok(0.5 * sum_sq)
    }

    /// L1 norm misfit: ||d_obs − d_syn||₁
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn l1_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let sum_abs: f64 = synthetic
            .iter()
            .zip(observed.iter())
            .map(|(&s, &o)| (s - o).abs())
            .sum();
        Ok(sum_abs)
    }

    /// L1 adjoint source: sign(d_syn − d_obs)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn l1_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let [m, n] = synthetic.shape();
        let mut adjoint = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                adjoint[[i, j]] = (synthetic[[i, j]] - observed[[i, j]]).signum();
            }
        }
        Ok(adjoint)
    }

    /// Normalized cross-correlation misfit.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn correlation_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let mut misfit = 0.0;

        for i in 0..observed.shape()[0] {
            let obs_trace = observed
                .index_axis::<1>(0, i)
                .expect("invariant: observed trace index in range");
            let syn_trace = synthetic
                .index_axis::<1>(0, i)
                .expect("invariant: synthetic trace index in range");

            let obs_norm = obs_trace.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let syn_norm = syn_trace.iter().map(|&x| x * x).sum::<f64>().sqrt();

            if obs_norm > 1e-10 && syn_norm > 1e-10 {
                let correlation = leto_ops::dot(&obs_trace, &syn_trace)
                    .expect("invariant: trace cross-correlation conforms")
                    / (obs_norm * syn_norm);
                misfit += 1.0 - correlation;
            }
        }

        Ok(misfit)
    }

    /// Correlation adjoint source — Fréchet derivative of J = 1 − C w.r.t. d_syn.
    ///
    /// J = 1 − C where C = (d_obs · d_syn) / (‖d_obs‖ ‖d_syn‖).
    ///
    /// ∂J/∂d_syn`J` = −∂C/∂d_syn`J`
    ///               = −obs`J`/(‖obs‖ ‖syn‖) + C · syn`J`/‖syn‖²
    ///
    /// This is the positive gradient of the misfit J; the optimizer then applies
    /// a negative step (compute_direction returns −gradient) to minimize J and
    /// therefore maximise the normalised cross-correlation.
    ///
    /// Previous code returned +∂C/∂d_syn (wrong sign), which drove the adjoint
    /// wavefield in the direction that *maximises* J (minimises correlation).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn correlation_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let mut adjoint = Array2::<f64>::zeros(synthetic.shape());

        for i in 0..observed.shape()[0] {
            let obs_trace = observed
                .index_axis::<1>(0, i)
                .expect("invariant: observed trace index in range");
            let syn_trace = synthetic
                .index_axis::<1>(0, i)
                .expect("invariant: synthetic trace index in range");

            let obs_norm = obs_trace.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let syn_norm = syn_trace.iter().map(|&x| x * x).sum::<f64>().sqrt();

            if obs_norm > 1e-10 && syn_norm > 1e-10 {
                let correlation = leto_ops::dot(&obs_trace, &syn_trace)
                    .expect("invariant: trace cross-correlation conforms")
                    / (obs_norm * syn_norm);

                for j in 0..adjoint.shape()[1] {
                    // ∂J/∂d_syn = −∂C/∂d_syn  (J = 1 − C)
                    adjoint[[i, j]] = -obs_trace[j] / (obs_norm * syn_norm)
                        + correlation * syn_trace[j] / (syn_norm * syn_norm);
                }
            }
        }

        Ok(adjoint)
    }
}
