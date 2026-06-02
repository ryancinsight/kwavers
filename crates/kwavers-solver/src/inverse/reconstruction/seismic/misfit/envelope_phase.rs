//! Envelope and instantaneous-phase misfit metrics.
//!
//! # References
//!
//! - Bozdağ et al. (2011): "Misfit functions for full waveform inversion based on
//!   instantaneous phase and envelope measurements", *Geophys. J. Int.*
//! - Fichtner et al. (2008): "The adjoint method in seismology", *Phys. Earth Planet. Inter.*
//! - Taner et al. (1979): "Complex seismic trace analysis"

use super::types::MisfitFunction;
use kwavers_core::error::KwaversResult;
use kwavers_domain::signal::analytic::{
    hilbert_transform, instantaneous_envelope_2d, instantaneous_phase_2d,
};
use kwavers_math::fft::{fft_1d_array, ifft_1d_complex, Complex64};
use ndarray::Array2;

impl MisfitFunction {
    /// Envelope misfit: 0.5 * ||E_syn − E_obs||² (cycle-skipping mitigation).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn envelope_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let env_obs = self.compute_envelope(observed)?;
        let env_syn = self.compute_envelope(synthetic)?;
        self.l2_misfit_arrays(&env_obs, &env_syn)
    }

    /// Instantaneous phase misfit.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn phase_misfit(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<f64> {
        let phase_obs = self.compute_instantaneous_phase(observed)?;
        let phase_syn = self.compute_instantaneous_phase(synthetic)?;
        self.l2_misfit_arrays(&phase_obs, &phase_syn)
    }

    /// Envelope adjoint source per Bozdağ et al. (2011), Eq. 11–13.
    ///
    /// ```text
    /// δE = (E_syn − E_obs) · Re[analytic / E_syn]
    ///    = (E_syn − E_obs) · s / E_syn
    /// ```
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn envelope_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let (ntraces, nsamples) = synthetic.dim();
        let mut adjoint = Array2::zeros((ntraces, nsamples));

        let env_obs = instantaneous_envelope_2d(observed);
        let env_syn = instantaneous_envelope_2d(synthetic);

        for i in 0..ntraces {
            let syn_trace = synthetic.row(i).to_owned();
            let analytic = hilbert_transform(&syn_trace);

            for j in 0..nsamples {
                let env_diff = env_syn[[i, j]] - env_obs[[i, j]];
                let envelope_syn = env_syn[[i, j]];

                adjoint[[i, j]] = if envelope_syn > 1e-12 {
                    env_diff * analytic[j].re / envelope_syn
                } else {
                    env_diff
                };
            }
        }

        Ok(adjoint)
    }

    /// Phase adjoint source per Fichtner et al. (2008) §2.3.2 and Bozdağ et al. (2011) Eq. 18–20.
    ///
    /// ```text
    /// δφ = (φ_syn − φ_obs) · [−Im(dz/dt) / |z|²]
    /// ```
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn phase_adjoint_source(
        &self,
        observed: &Array2<f64>,
        synthetic: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let (ntraces, nsamples) = synthetic.dim();
        let mut adjoint = Array2::zeros((ntraces, nsamples));

        let phase_obs = instantaneous_phase_2d(observed);
        let phase_syn = instantaneous_phase_2d(synthetic);

        for i in 0..ntraces {
            let syn_trace = synthetic.row(i).to_owned();
            let analytic = hilbert_transform(&syn_trace);

            // Time derivative of analytic signal (central differences)
            let mut dz_dt = vec![Complex64::new(0.0, 0.0); nsamples];
            if nsamples >= 3 {
                dz_dt[0] = analytic[1] - analytic[0];
                for j in 1..nsamples - 1 {
                    dz_dt[j] = (analytic[j + 1] - analytic[j - 1]) * 0.5;
                }
                dz_dt[nsamples - 1] = analytic[nsamples - 1] - analytic[nsamples - 2];
            }

            for j in 0..nsamples {
                let phase_diff =
                    kwavers_math::signal::wrap_to_pi(phase_syn[[i, j]] - phase_obs[[i, j]]);

                let envelope_sq = analytic[j].norm_sqr();
                adjoint[[i, j]] = if envelope_sq > 1e-12 {
                    phase_diff * (-dz_dt[j].im / envelope_sq)
                } else {
                    0.0
                };
            }
        }

        Ok(adjoint)
    }

    /// Compute envelope via Hilbert transform (Marple 1999).
    ///
    /// The analytic signal is z(t) = x(t) + i·H[x(t)]; envelope = |z(t)|.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_envelope(&self, signal: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (ntraces, nsamples) = signal.dim();
        let mut envelope = Array2::zeros((ntraces, nsamples));

        for i in 0..ntraces {
            let trace = signal.row(i);
            let mut buffer = fft_1d_array(&trace.to_owned());

            // Double positive frequencies, zero negative frequencies
            for sample in buffer.iter_mut().take(nsamples / 2).skip(1) {
                *sample *= Complex64::new(2.0, 0.0);
            }
            for sample in buffer.iter_mut().skip(nsamples / 2 + 1) {
                *sample = Complex64::new(0.0, 0.0);
            }

            let analytic_signal = ifft_1d_complex(&buffer);
            for (j, sample) in analytic_signal.iter().enumerate() {
                envelope[[i, j]] = sample.norm();
            }
        }

        Ok(envelope)
    }

    /// Compute instantaneous phase via Hilbert transform (Taner et al. 1979).
    ///
    /// φ(t) = atan2(H[x(t)], x(t))
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_instantaneous_phase(
        &self,
        signal: &Array2<f64>,
    ) -> KwaversResult<Array2<f64>> {
        let (ntraces, nsamples) = signal.dim();
        let mut phase = Array2::zeros((ntraces, nsamples));

        for i in 0..ntraces {
            let trace = signal.row(i);
            let mut buffer = fft_1d_array(&trace.to_owned());

            for sample in buffer.iter_mut().take(nsamples / 2).skip(1) {
                *sample *= Complex64::new(2.0, 0.0);
            }
            for sample in buffer.iter_mut().skip(nsamples / 2 + 1) {
                *sample = Complex64::new(0.0, 0.0);
            }

            let analytic_signal = ifft_1d_complex(&buffer);
            for (j, sample) in analytic_signal.iter().enumerate() {
                let real = trace[j];
                let imag = sample.im;
                phase[[i, j]] = imag.atan2(real);
            }
        }

        Ok(phase)
    }

    /// L2 misfit between two pre-computed arrays.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn l2_misfit_arrays(&self, a: &Array2<f64>, b: &Array2<f64>) -> KwaversResult<f64> {
        let diff = b - a;
        Ok(0.5 * diff.mapv(|x| x * x).sum())
    }
}
