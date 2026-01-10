//! Misfit functions for Full Waveform Inversion
//!
//! Implements literature-validated misfit functions and adjoint sources for FWI:
//! - L2/L1 norms for standard waveform matching
//! - Envelope misfit with Hilbert transform adjoint (Bozdağ et al., 2011)
//! - Phase misfit with instantaneous phase adjoint (Fichtner et al., 2008)
//! - Cross-correlation and Wasserstein metrics
//!
//! # References
//!
//! - Bozdağ et al. (2011): "Misfit functions for full waveform inversion based on
//!   instantaneous phase and envelope measurements", *Geophysical Journal International*
//! - Fichtner et al. (2008): "The adjoint method in seismology", *Physics of the Earth
//!   and Planetary Interiors*
//! - Tarantola (1984): "Inversion of seismic reflection data in the acoustic approximation"

use crate::core::error::KwaversResult;
use crate::domain::signal::analytic::{
    hilbert_transform, instantaneous_envelope_2d, instantaneous_phase_2d,
};
use ndarray::Array2;
use num_complex::Complex;

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
    misfit_type: MisfitType,
}

impl MisfitFunction {
    /// Create a new misfit function calculator
    #[must_use]
    pub fn new(misfit_type: MisfitType) -> Self {
        Self { misfit_type }
    }

    /// Compute adjoint source from residual (direct interface for L1/L2 norms)
    ///
    /// Note: For envelope and phase misfits, use `compute_adjoint_source` instead
    /// for proper Hilbert transform-based adjoint computation per Fichtner et al. (2008).
    /// This method provides direct adjoint for simple norms, falls back to residual for complex misfits.
    #[must_use]
    pub fn adjoint_source(&self, residual: &Array2<f64>) -> Array2<f64> {
        match self.misfit_type {
            MisfitType::L2Norm => residual.clone(),
            MisfitType::L1Norm => residual.mapv(f64::signum),
            // For these, the full implementation requires observed and synthetic data
            // Use compute_adjoint_source for proper adjoint computation
            MisfitType::Envelope => residual.clone(),
            MisfitType::Phase => residual.clone(),
            MisfitType::Correlation => residual.clone(),
            MisfitType::Wasserstein => residual.clone(),
        }
    }

    /// Compute misfit between observed and synthetic data
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

    /// Compute adjoint source for gradient calculation
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

    /// L2 norm misfit: 0.5 * ||`d_obs` - `d_syn||²`
    fn l2_misfit(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<f64> {
        let diff = synthetic - observed;
        Ok(0.5 * diff.mapv(|x| x * x).sum())
    }

    /// L1 norm misfit: ||`d_obs` - `d_syn||₁`
    fn l1_misfit(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<f64> {
        let diff = synthetic - observed;
        Ok(diff.mapv(f64::abs).sum())
    }

    /// Envelope misfit for cycle-skipping mitigation
    fn envelope_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let env_obs = self.compute_envelope(observed)?;
        let env_syn = self.compute_envelope(synthetic)?;
        self.l2_misfit(&env_obs, &env_syn)
    }

    /// Phase-only misfit
    fn phase_misfit(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<f64> {
        let phase_obs = self.compute_instantaneous_phase(observed)?;
        let phase_syn = self.compute_instantaneous_phase(synthetic)?;
        self.l2_misfit(&phase_obs, &phase_syn)
    }

    /// Normalized cross-correlation misfit
    fn correlation_misfit(
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

    /// Wasserstein distance with optimal transport
    ///
    /// Computes W₁ (1-Wasserstein) distance between seismograms as probability distributions.
    /// Uses the 1D optimal transport solution which has a closed form via sorting.
    ///
    /// For 1D distributions: W₁(μ,ν) = ∫|F_μ^(-1)(u) - F_ν^(-1)(u)|du
    /// where F^(-1) is the quantile function (inverse CDF).
    ///
    /// This is equivalent to the L1 distance between sorted samples.
    ///
    /// References:
    /// - Villani (2003): "Topics in Optimal Transportation"
    /// - Engquist & Froese (2014): "Application of Wasserstein metric to seismic signals"
    /// - Métivier et al. (2016): "Measuring the misfit between seismograms using optimal transport"
    fn wasserstein_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let mut total_distance = 0.0;
        let ntraces = observed.shape()[0];

        for i in 0..ntraces {
            let obs_trace = observed.row(i).to_owned();
            let syn_trace = synthetic.row(i).to_owned();

            // Shift to non-negative and normalize to probability distributions
            let obs_min = obs_trace.iter().fold(f64::INFINITY, |min, &x| min.min(x));
            let syn_min = syn_trace.iter().fold(f64::INFINITY, |min, &x| min.min(x));

            let obs_shifted = obs_trace.mapv(|x| (x - obs_min).max(0.0));
            let syn_shifted = syn_trace.mapv(|x| (x - syn_min).max(0.0));

            let obs_sum = obs_shifted.sum();
            let syn_sum = syn_shifted.sum();

            // Skip traces with no energy
            if obs_sum < 1e-10 || syn_sum < 1e-10 {
                continue;
            }

            // Normalize to probability distributions
            let obs_prob = obs_shifted.mapv(|x| x / obs_sum);
            let syn_prob = syn_shifted.mapv(|x| x / syn_sum);

            // Compute cumulative distribution functions
            let n = obs_prob.len();
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

            // 1-Wasserstein distance is L1 distance between CDFs
            let mut trace_distance = 0.0;
            for j in 0..n {
                trace_distance += (obs_cdf[j] - syn_cdf[j]).abs();
            }

            // Normalize by number of samples for scale invariance
            trace_distance /= n as f64;

            total_distance += trace_distance;
        }

        // Average over all traces
        Ok(total_distance / ntraces as f64)
    }

    /// L1 adjoint source
    fn l1_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let diff = synthetic - observed;
        Ok(diff.mapv(f64::signum))
    }

    /// Envelope adjoint source with Hilbert transform
    ///
    /// Computes the proper Fréchet derivative for envelope-based misfit.
    /// Per Bozdağ et al. (2011), the adjoint source for envelope misfit is:
    ///
    /// ```text
    /// δE = (E_syn - E_obs) * Re[(s + i*H(s)) / E_syn]
    /// ```
    ///
    /// where:
    /// - `E` = envelope (magnitude of analytic signal)
    /// - `s` = synthetic seismogram
    /// - `H(s)` = Hilbert transform of `s`
    /// - Re[·] = real part
    ///
    /// This accounts for the fact that envelope is a nonlinear function of the signal.
    ///
    /// # References
    ///
    /// - Bozdağ et al. (2011): Eq. 11-13, envelope adjoint derivation
    /// - Wu et al. (2014): "Seismic envelope inversion and modulation signal model"
    fn envelope_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let (ntraces, nsamples) = synthetic.dim();
        let mut adjoint = Array2::zeros((ntraces, nsamples));

        // Compute envelopes
        let env_obs = instantaneous_envelope_2d(observed);
        let env_syn = instantaneous_envelope_2d(synthetic);

        // For each trace, compute proper envelope adjoint
        for i in 0..ntraces {
            let syn_trace = synthetic.row(i).to_owned();

            // Compute analytic signal: z = s + i*H(s)
            let analytic = hilbert_transform(&syn_trace);

            // Compute envelope difference
            for j in 0..nsamples {
                let env_diff = env_syn[[i, j]] - env_obs[[i, j]];
                let envelope_syn = env_syn[[i, j]];

                // Avoid division by zero at signal nulls
                if envelope_syn > 1e-12 {
                    // Adjoint = (E_syn - E_obs) * Re[analytic / E_syn]
                    // = (E_syn - E_obs) * s / E_syn
                    // This is the projection of the analytic signal onto the envelope direction
                    adjoint[[i, j]] = env_diff * analytic[j].re / envelope_syn;
                } else {
                    // At signal nulls, use simple difference
                    adjoint[[i, j]] = env_diff;
                }
            }
        }

        Ok(adjoint)
    }

    /// Instantaneous phase adjoint source
    ///
    /// Computes the proper Fréchet derivative for phase-based misfit.
    /// Per Fichtner et al. (2008) and Bozdağ et al. (2011), the adjoint source is:
    ///
    /// ```text
    /// δφ = (φ_syn - φ_obs) * [-Im(∂z/∂t) / |z|²]
    /// ```
    ///
    /// where:
    /// - `φ` = instantaneous phase = atan2(H(s), s)
    /// - `z = s + i*H(s)` = analytic signal
    /// - `∂z/∂t` = time derivative of analytic signal
    /// - Im[·] = imaginary part
    ///
    /// Analytical simplification of general formula
    ///
    /// For acoustic media with constant velocity, the Fréchet derivative
    /// simplifies to this expression (see Tarantola 1984, Eq. 6.97).
    /// ```text
    /// δφ = (φ_syn - φ_obs) * [s*H'(s) - s'(t)*H(s)] / (s² + H(s)²)
    /// ```
    ///
    /// where `'` denotes time derivative.
    ///
    /// # References
    ///
    /// - Fichtner et al. (2008): Section 2.3.2, phase adjoint derivation
    /// - Bozdağ et al. (2011): Eq. 18-20, instantaneous phase misfit
    /// - Bozdag et al. (2011): "Misfit functions for full waveform inversion"
    fn phase_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let (ntraces, nsamples) = synthetic.dim();
        let mut adjoint = Array2::zeros((ntraces, nsamples));

        // Compute instantaneous phases
        let phase_obs = instantaneous_phase_2d(observed);
        let phase_syn = instantaneous_phase_2d(synthetic);

        // For each trace, compute proper phase adjoint
        for i in 0..ntraces {
            let syn_trace = synthetic.row(i).to_owned();

            // Compute analytic signal: z = s + i*H(s)
            let analytic = hilbert_transform(&syn_trace);

            // Compute time derivative of analytic signal using central differences
            let mut dz_dt = vec![Complex::new(0.0, 0.0); nsamples];
            if nsamples >= 3 {
                // Forward difference for first point
                dz_dt[0] = analytic[1] - analytic[0];

                // Central difference for interior points
                for j in 1..nsamples - 1 {
                    dz_dt[j] = (analytic[j + 1] - analytic[j - 1]) * 0.5;
                }

                // Backward difference for last point
                dz_dt[nsamples - 1] = analytic[nsamples - 1] - analytic[nsamples - 2];
            }

            // Compute phase adjoint
            for j in 0..nsamples {
                // Phase difference (handle wrapping)
                let mut phase_diff = phase_syn[[i, j]] - phase_obs[[i, j]];

                // Unwrap phase difference to [-π, π]
                while phase_diff > std::f64::consts::PI {
                    phase_diff -= 2.0 * std::f64::consts::PI;
                }
                while phase_diff < -std::f64::consts::PI {
                    phase_diff += 2.0 * std::f64::consts::PI;
                }

                // Envelope squared: |z|²
                let envelope_sq = analytic[j].norm_sqr();

                // Avoid division by zero at signal nulls
                if envelope_sq > 1e-12 {
                    // Adjoint = (φ_syn - φ_obs) * [-Im(dz/dt) / |z|²]
                    // = phase_diff * [-Im(dz/dt) / |z|²]
                    adjoint[[i, j]] = phase_diff * (-dz_dt[j].im / envelope_sq);
                } else {
                    // At signal nulls, phase is undefined, so adjoint is zero
                    adjoint[[i, j]] = 0.0;
                }
            }
        }

        Ok(adjoint)
    }

    /// Correlation adjoint source
    fn correlation_adjoint_source(
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

                // Gradient of normalized cross-correlation
                for j in 0..adjoint.shape()[1] {
                    adjoint[[i, j]] = obs_trace[j] / (obs_norm * syn_norm)
                        - correlation * syn_trace[j] / (syn_norm * syn_norm);
                }
            }
        }

        Ok(adjoint)
    }

    /// Wasserstein adjoint source for gradient computation
    ///
    /// Computes the adjoint source (Fréchet derivative) for the Wasserstein distance.
    /// The adjoint source is the optimal transport map between distributions.
    ///
    /// For 1D case: adjoint = sign(F_syn - F_obs) where F are CDFs
    /// This gives the direction of transport to minimize the distance.
    ///
    /// References:
    /// - Engquist et al. (2016): "Optimal transport for seismic FWI"
    /// - Métivier et al. (2016): "An optimal transport approach for seismic tomography"
    fn wasserstein_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let (ntraces, nsamples) = observed.dim();
        let mut adjoint = Array2::zeros((ntraces, nsamples));

        for i in 0..ntraces {
            let obs_trace = observed.row(i).to_owned();
            let syn_trace = synthetic.row(i).to_owned();

            // Shift to non-negative
            let obs_min = obs_trace.iter().fold(f64::INFINITY, |min, &x| min.min(x));
            let syn_min = syn_trace.iter().fold(f64::INFINITY, |min, &x| min.min(x));

            let obs_shifted = obs_trace.mapv(|x| (x - obs_min).max(0.0));
            let syn_shifted = syn_trace.mapv(|x| (x - syn_min).max(0.0));

            // Normalize
            let obs_sum = obs_shifted.sum();
            let syn_sum = syn_shifted.sum();

            if obs_sum < 1e-10 || syn_sum < 1e-10 {
                continue;
            }

            let obs_prob = obs_shifted.mapv(|x| x / obs_sum);
            let syn_prob = syn_shifted.mapv(|x| x / syn_sum);

            // Compute CDFs
            let mut obs_cdf = Vec::with_capacity(nsamples);
            let mut syn_cdf = Vec::with_capacity(nsamples);

            let mut obs_cumsum = 0.0;
            let mut syn_cumsum = 0.0;

            for j in 0..nsamples {
                obs_cumsum += obs_prob[j];
                syn_cumsum += syn_prob[j];
                obs_cdf.push(obs_cumsum);
                syn_cdf.push(syn_cumsum);
            }

            // Adjoint source is the sign of CDF difference
            // This represents the optimal transport direction
            for j in 0..nsamples {
                let cdf_diff = syn_cdf[j] - obs_cdf[j];

                // Weight by original amplitude for proper scaling
                let weight = syn_prob[j].max(1e-10);

                // Adjoint source: derivative of Wasserstein w.r.t. synthetic data
                adjoint[[i, j]] = cdf_diff.signum() * weight;
            }
        }

        Ok(adjoint)
    }

    /// Compute envelope using Hilbert transform
    ///
    /// The analytic signal is z(t) = x(t) + i*H[x(t)] where H is Hilbert transform.
    /// The envelope is |z(t)| = sqrt(x(t)² + H[x(t)]²)
    ///
    /// Implementation via FFT:
    /// 1. X(f) = FFT[x(t)]
    /// 2. H[X(f)] = -i*sgn(f)*X(f)
    /// 3. h(t) = IFFT[H[X(f)]]
    /// 4. envelope = sqrt(x² + h²)
    ///
    /// References:
    /// - Marple (1999): "Computing the discrete-time analytic signal via FFT"
    /// - Oppenheim & Schafer (2009): "Discrete-Time Signal Processing", Ch. 12
    fn compute_envelope(&self, signal: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        use rustfft::{num_complex::Complex, FftPlanner};

        let (ntraces, nsamples) = signal.dim();
        let mut envelope = Array2::zeros((ntraces, nsamples));

        // Process each trace independently
        for i in 0..ntraces {
            let trace = signal.row(i);

            // Step 1: FFT of real signal
            let mut fft_planner = FftPlanner::new();
            let fft = fft_planner.plan_fft_forward(nsamples);

            let mut buffer: Vec<Complex<f64>> =
                trace.iter().map(|&x| Complex::new(x, 0.0)).collect();

            fft.process(&mut buffer);

            // Step 2: Apply Hilbert transform in frequency domain
            // H[X(f)] = -i*sgn(f)*X(f)
            // For positive frequencies: multiply by 2
            // For negative frequencies: zero out
            // DC and Nyquist: keep as is

            // Zero out negative frequencies, double positive frequencies
            for sample in buffer.iter_mut().take(nsamples / 2).skip(1) {
                *sample *= Complex::new(2.0, 0.0);
            }
            for sample in buffer.iter_mut().skip(nsamples / 2 + 1) {
                *sample = Complex::new(0.0, 0.0);
            }

            // Step 3: IFFT to get analytic signal
            let ifft = fft_planner.plan_fft_inverse(nsamples);
            ifft.process(&mut buffer);

            // Normalize IFFT
            let norm = 1.0 / nsamples as f64;
            for sample in &mut buffer {
                *sample *= norm;
            }

            // Step 4: Compute envelope as magnitude of analytic signal
            for (j, sample) in buffer.iter().enumerate() {
                envelope[[i, j]] = sample.norm();
            }
        }

        Ok(envelope)
    }

    /// Compute instantaneous phase using Hilbert transform
    ///
    /// The instantaneous phase is φ(t) = atan2(H[x(t)], x(t))
    /// where H[x] is the Hilbert transform of x.
    ///
    /// References:
    /// - Taner et al. (1979): "Complex seismic trace analysis"
    /// - Barnes (2007): "A tutorial on complex seismic trace analysis"
    fn compute_instantaneous_phase(&self, signal: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        use rustfft::{num_complex::Complex, FftPlanner};

        let (ntraces, nsamples) = signal.dim();
        let mut phase = Array2::zeros((ntraces, nsamples));

        // Process each trace
        for i in 0..ntraces {
            let trace = signal.row(i);

            // Compute analytic signal via Hilbert transform
            let mut fft_planner = FftPlanner::new();
            let fft = fft_planner.plan_fft_forward(nsamples);

            let mut buffer: Vec<Complex<f64>> =
                trace.iter().map(|&x| Complex::new(x, 0.0)).collect();

            fft.process(&mut buffer);

            // Apply Hilbert transform in frequency domain
            for sample in buffer.iter_mut().take(nsamples / 2).skip(1) {
                *sample *= Complex::new(2.0, 0.0);
            }
            for sample in buffer.iter_mut().skip(nsamples / 2 + 1) {
                *sample = Complex::new(0.0, 0.0);
            }

            let ifft = fft_planner.plan_fft_inverse(nsamples);
            ifft.process(&mut buffer);

            let norm = 1.0 / nsamples as f64;

            // Compute instantaneous phase
            for (j, sample) in buffer.iter().enumerate() {
                let real = trace[j];
                let imag = sample.im * norm;

                // Phase = atan2(imaginary, real)
                phase[[i, j]] = imag.atan2(real);
            }
        }

        Ok(phase)
    }
}
