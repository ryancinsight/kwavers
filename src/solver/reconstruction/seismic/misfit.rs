//! Misfit functions for Full Waveform Inversion

use ndarray::{Array2, Zip};
use crate::error::KwaversResult;

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
pub struct MisfitFunction {
    misfit_type: MisfitType,
}

impl MisfitFunction {
    /// Create a new misfit function calculator
    pub fn new(misfit_type: MisfitType) -> Self {
        Self { misfit_type }
    }
    
    /// Compute misfit between observed and synthetic data
    pub fn compute(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
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
    
    /// L2 norm misfit: 0.5 * ||d_obs - d_syn||²
    fn l2_misfit(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<f64> {
        let diff = synthetic - observed;
        Ok(0.5 * diff.mapv(|x| x * x).sum())
    }
    
    /// L1 norm misfit: ||d_obs - d_syn||₁
    fn l1_misfit(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<f64> {
        let diff = synthetic - observed;
        Ok(diff.mapv(|x| x.abs()).sum())
    }
    
    /// Envelope misfit for cycle-skipping mitigation
    fn envelope_misfit(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<f64> {
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
    fn correlation_misfit(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<f64> {
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
    
    /// Wasserstein distance (simplified 1D implementation)
    fn wasserstein_misfit(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<f64> {
        // Simplified 1D Wasserstein distance for each trace
        let mut total_distance = 0.0;
        
        for i in 0..observed.shape()[0] {
            let obs_trace = observed.row(i).to_owned();
            let syn_trace = synthetic.row(i).to_owned();
            
            // Normalize to probability distributions
            let obs_sum = obs_trace.mapv(|x| x.abs()).sum();
            let syn_sum = syn_trace.mapv(|x| x.abs()).sum();
            
            if obs_sum > 1e-10 && syn_sum > 1e-10 {
                let obs_norm = obs_trace.mapv(|x| x.abs() / obs_sum);
                let syn_norm = syn_trace.mapv(|x| x.abs() / syn_sum);
                
                // Compute cumulative distributions
                let mut obs_cdf = obs_norm.clone();
                let mut syn_cdf = syn_norm.clone();
                
                for j in 1..obs_cdf.len() {
                    obs_cdf[j] += obs_cdf[j-1];
                    syn_cdf[j] += syn_cdf[j-1];
                }
                
                // Wasserstein-1 distance
                let distance = Zip::from(&obs_cdf)
                    .and(&syn_cdf)
                    .map_collect(|&o, &s| (o - s).abs())
                    .sum();
                
                total_distance += distance;
            }
        }
        
        Ok(total_distance)
    }
    
    /// L1 adjoint source
    fn l1_adjoint_source(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let diff = synthetic - observed;
        Ok(diff.mapv(|x| x.signum()))
    }
    
    /// Envelope adjoint source
    fn envelope_adjoint_source(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        // Simplified: use L2 adjoint for envelope
        // Full implementation would require Hilbert transform adjoint
        Ok(synthetic - observed)
    }
    
    /// Phase adjoint source
    fn phase_adjoint_source(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        // Simplified: use L2 adjoint for phase
        // Full implementation would require instantaneous phase adjoint
        Ok(synthetic - observed)
    }
    
    /// Correlation adjoint source
    fn correlation_adjoint_source(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<Array2<f64>> {
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
    
    /// Wasserstein adjoint source (simplified)
    fn wasserstein_adjoint_source(&self, observed: &Array2<f64>, synthetic: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        // Simplified implementation
        // Full implementation would require solving optimal transport problem
        Ok(synthetic - observed)
    }
    
    /// Compute envelope using Hilbert transform (simplified)
    fn compute_envelope(&self, signal: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        // Simplified: just return absolute value
        // Full implementation would use Hilbert transform
        Ok(signal.mapv(|x| x.abs()))
    }
    
    /// Compute instantaneous phase (simplified)
    fn compute_instantaneous_phase(&self, signal: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        // Simplified: return zero phase
        // Full implementation would use Hilbert transform
        Ok(Array2::zeros(signal.dim()))
    }
}