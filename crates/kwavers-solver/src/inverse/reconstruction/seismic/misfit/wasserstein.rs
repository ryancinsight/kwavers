//! Wasserstein distance misfit and its adjoint source.
//!
//! Computes W₁ (1-Wasserstein) distance between seismograms as probability distributions.
//! Uses the 1D optimal transport solution: W₁(μ,ν) = L1 distance between CDFs.
//!
//! # References
//!
//! - Villani (2003): "Topics in Optimal Transportation"
//! - Engquist & Froese (2014): "Application of Wasserstein metric to seismic signals"
//! - Métivier et al. (2016): "Measuring the misfit between seismograms using optimal transport"

use super::types::MisfitFunction;
use kwavers_core::error::KwaversResult;
use leto::Array2;

impl MisfitFunction {
    /// Wasserstein distance misfit (1-Wasserstein via CDF L1 distance).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn wasserstein_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let mut total_distance = 0.0;
        let ntraces = observed.shape()[0];

        for i in 0..ntraces {
            let obs_trace = observed.index_axis(0, i).unwrap().to_owned();
            let syn_trace = synthetic.index_axis(0, i).unwrap().to_owned();

            // Shift to non-negative and normalize to probability distributions
            let obs_min = obs_trace.iter().fold(f64::INFINITY, |min, &x| min.min(x));
            let syn_min = syn_trace.iter().fold(f64::INFINITY, |min, &x| min.min(x));

            let obs_shifted = obs_trace.mapv(|x| (x - obs_min).max(0.0));
            let syn_shifted = syn_trace.mapv(|x| (x - syn_min).max(0.0));

            let obs_sum = obs_shifted.sum();
            let syn_sum = syn_shifted.sum();

            if obs_sum < 1e-10 || syn_sum < 1e-10 {
                continue;
            }

            let obs_prob = obs_shifted.mapv(|x| x / obs_sum);
            let syn_prob = syn_shifted.mapv(|x| x / syn_sum);

            let n = (obs_prob.shape()[0] * obs_prob.shape()[1] * obs_prob.shape()[2]);
            let (obs_cdf, syn_cdf) = cumulative_distributions(&obs_prob, &syn_prob, n);

            // 1-Wasserstein = L1 distance between CDFs
            let trace_distance: f64 =
                (0..n).map(|j| (obs_cdf[j] - syn_cdf[j]).abs()).sum::<f64>() / n as f64;

            total_distance += trace_distance;
        }

        Ok(total_distance / ntraces as f64)
    }

    /// Wasserstein adjoint source (optimal transport gradient).
    ///
    /// For the 1D case: adjoint = sign(F_syn − F_obs) weighted by probability mass.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn wasserstein_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let [ntraces, nsamples] = observed.shape();
        let mut adjoint = Array2::zeros((ntraces, nsamples));

        for i in 0..ntraces {
            let obs_trace = observed.index_axis(0, i).unwrap().to_owned();
            let syn_trace = synthetic.index_axis(0, i).unwrap().to_owned();

            let obs_min = obs_trace.iter().fold(f64::INFINITY, |min, &x| min.min(x));
            let syn_min = syn_trace.iter().fold(f64::INFINITY, |min, &x| min.min(x));

            let obs_shifted = obs_trace.mapv(|x| (x - obs_min).max(0.0));
            let syn_shifted = syn_trace.mapv(|x| (x - syn_min).max(0.0));

            let obs_sum = obs_shifted.sum();
            let syn_sum = syn_shifted.sum();

            if obs_sum < 1e-10 || syn_sum < 1e-10 {
                continue;
            }

            let obs_prob = obs_shifted.mapv(|x| x / obs_sum);
            let syn_prob = syn_shifted.mapv(|x| x / syn_sum);

            let (obs_cdf, syn_cdf) = cumulative_distributions(&obs_prob, &syn_prob, nsamples);

            for j in 0..nsamples {
                let cdf_diff = syn_cdf[j] - obs_cdf[j];
                let weight = syn_prob[j].max(1e-10);
                adjoint[[i, j]] = cdf_diff.signum() * weight;
            }
        }

        Ok(adjoint)
    }
}

fn cumulative_distributions(
    obs_prob: &leto::Array1<f64>,
    syn_prob: &leto::Array1<f64>,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut obs_cdf = Vec::with_capacity(n);
    let mut syn_cdf = Vec::with_capacity(n);
    let mut obs_cumsum = 0.0;
    let mut syn_cumsum = 0.0;
    for j in 0..n {
        obs_cumsum += obs_prob[j];
        syn_cumsum += syn_prob[j];
        obs_cdf.push(obs_cumsum);
        syn_cdf.push(syn_cumsum);
    }
    (obs_cdf, syn_cdf)
}
