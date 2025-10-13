//! Misfit functions for Full Waveform Inversion

use crate::error::KwaversResult;
use ndarray::Array2;

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

    /// Compute adjoint source from residual
    #[must_use]
    pub fn adjoint_source(&self, residual: &Array2<f64>) -> Array2<f64> {
        match self.misfit_type {
            MisfitType::L2Norm => residual.clone(),
            MisfitType::L1Norm => residual.mapv(f64::signum),
            MisfitType::Envelope => residual.clone(), // Simplified
            MisfitType::Phase => residual.clone(),    // Simplified
            MisfitType::Correlation => residual.clone(), // Simplified
            MisfitType::Wasserstein => residual.clone(), // Simplified
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

    /// Envelope adjoint source
    fn envelope_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        // Simplified: use L2 adjoint for envelope
        // Full implementation would require Hilbert transform adjoint
        Ok(synthetic - observed)
    }

    /// Phase adjoint source
    fn phase_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        // Simplified: use L2 adjoint for phase
        // Full implementation would require instantaneous phase adjoint
        Ok(synthetic - observed)
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
            
            let mut buffer: Vec<Complex<f64>> = trace
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect();
            
            fft.process(&mut buffer);
            
            // Step 2: Apply Hilbert transform in frequency domain
            // H[X(f)] = -i*sgn(f)*X(f)
            // For positive frequencies: multiply by 2
            // For negative frequencies: zero out
            // DC and Nyquist: keep as is
            
            // Zero out negative frequencies, double positive frequencies
            for k in 1..nsamples / 2 {
                buffer[k] *= Complex::new(2.0, 0.0);
            }
            for k in (nsamples / 2 + 1)..nsamples {
                buffer[k] = Complex::new(0.0, 0.0);
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
            
            let mut buffer: Vec<Complex<f64>> = trace
                .iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect();
            
            fft.process(&mut buffer);
            
            // Apply Hilbert transform in frequency domain
            for k in 1..nsamples / 2 {
                buffer[k] *= Complex::new(2.0, 0.0);
            }
            for k in (nsamples / 2 + 1)..nsamples {
                buffer[k] = Complex::new(0.0, 0.0);
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
