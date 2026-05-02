//! L2, L1, and cross-correlation misfit metrics.

use super::types::MisfitFunction;
use crate::core::error::KwaversResult;
use ndarray::Array2;

impl MisfitFunction {
    /// L2 norm misfit: 0.5 * ||d_obs − d_syn||²
    pub(super) fn l2_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let diff = synthetic - observed;
        Ok(0.5 * diff.mapv(|x| x * x).sum())
    }

    /// L1 norm misfit: ||d_obs − d_syn||₁
    pub(super) fn l1_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let diff = synthetic - observed;
        Ok(diff.mapv(f64::abs).sum())
    }

    /// L1 adjoint source: sign(d_syn − d_obs)
    pub(super) fn l1_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let diff = synthetic - observed;
        Ok(diff.mapv(f64::signum))
    }

    /// Normalized cross-correlation misfit.
    pub(super) fn correlation_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let mut misfit = 0.0;

        for i in 0..observed.shape()[0] {
            let obs_trace = observed.row(i);
            let syn_trace = synthetic.row(i);

            let obs_norm = obs_trace.mapv(|x| x * x).sum().sqrt();
            let syn_norm = syn_trace.mapv(|x| x * x).sum().sqrt();

            if obs_norm > 1e-10 && syn_norm > 1e-10 {
                let correlation = obs_trace.dot(&syn_trace) / (obs_norm * syn_norm);
                misfit += 1.0 - correlation;
            }
        }

        Ok(misfit)
    }

    /// Correlation adjoint source (Fréchet derivative of normalized cross-correlation).
    pub(super) fn correlation_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let mut adjoint = Array2::zeros(synthetic.dim());

        for i in 0..observed.shape()[0] {
            let obs_trace = observed.row(i);
            let syn_trace = synthetic.row(i);

            let obs_norm = obs_trace.mapv(|x| x * x).sum().sqrt();
            let syn_norm = syn_trace.mapv(|x| x * x).sum().sqrt();

            if obs_norm > 1e-10 && syn_norm > 1e-10 {
                let correlation = obs_trace.dot(&syn_trace) / (obs_norm * syn_norm);

                for j in 0..adjoint.shape()[1] {
                    adjoint[[i, j]] = obs_trace[j] / (obs_norm * syn_norm)
                        - correlation * syn_trace[j] / (syn_norm * syn_norm);
                }
            }
        }

        Ok(adjoint)
    }
}
