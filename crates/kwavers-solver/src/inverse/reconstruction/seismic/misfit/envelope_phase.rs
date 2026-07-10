//! Envelope and instantaneous-phase misfit metrics.
//!
//! # References
//!
//! - Bozdağ et al. (2011): "Misfit functions for full waveform inversion based on
//!   instantaneous phase and envelope measurements", *Geophys. J. Int.*
//! - Fichtner et al. (2008): "The adjoint method in seismology", *Phys. Earth Planet. Inter.*
//! - Taner et al. (1979): "Complex seismic trace analysis"

use super::types::MisfitFunction;
use apollo::{fft_1d_leto, ifft_1d_complex, Complex64 as ApolloComplex64};
use kwavers_core::error::KwaversResult;
use kwavers_signal::analytic::hilbert_transform;
use leto::Array1 as LetoArray1;
use leto::Array2;
use kwavers_math::fft::Complex64;

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
        let [ntraces, nsamples] = synthetic.shape();
        let mut adjoint = Array2::zeros((ntraces, nsamples));

        let env_obs = self.compute_envelope(observed)?;
        let env_syn = self.compute_envelope(synthetic)?;

        for i in 0..ntraces {
            let syn_trace = synthetic.index_axis::<1>(0, i).unwrap().to_contiguous();
            let analytic = hilbert_transform(
                &LetoArray1::from_vec([nsamples], syn_trace.iter().copied().collect())
                    .expect("envelope trace length must match its Leto shape"),
            );

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
        let [ntraces, nsamples] = synthetic.shape();
        let mut adjoint = Array2::zeros((ntraces, nsamples));

        let phase_obs = self.compute_instantaneous_phase(observed)?;
        let phase_syn = self.compute_instantaneous_phase(synthetic)?;

        for i in 0..ntraces {
            let syn_trace = synthetic.index_axis::<1>(0, i).unwrap().to_contiguous();
            let analytic = hilbert_transform(
                &LetoArray1::from_vec([nsamples], syn_trace.iter().copied().collect())
                    .expect("phase trace length must match its Leto shape"),
            );

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
        let [ntraces, nsamples] = signal.shape();
        let mut envelope = Array2::zeros((ntraces, nsamples));

        for i in 0..ntraces {
            let trace = signal.index_axis::<1>(0, i).unwrap();
            let trace_buffer = LetoArray1::from_shape_vec([nsamples], trace.iter().cloned().collect::<Vec<_>>())
                .expect("seismic envelope trace length must match its Leto shape");
            let mut buffer = fft_1d_leto(trace_buffer.view());

            // Double positive frequencies, zero negative frequencies
            let spectrum = buffer
                .as_slice_mut()
                .expect("Apollo 1-D FFT output must be contiguous");
            for sample in spectrum.iter_mut().take(nsamples / 2).skip(1) {
                *sample *= ApolloComplex64::new(2.0, 0.0);
            }
            for sample in spectrum.iter_mut().skip(nsamples / 2 + 1) {
                *sample = ApolloComplex64::new(0.0, 0.0);
            }

            let complex_buffer = LetoArray1::from_shape_vec([nsamples], buffer.into_vec())
                .expect("seismic envelope spectrum length must match its Leto shape");
            let analytic_signal = ifft_1d_complex(&complex_buffer);
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
        let [ntraces, nsamples] = signal.shape();
        let mut phase = Array2::zeros((ntraces, nsamples));

        for i in 0..ntraces {
            let trace = signal.index_axis::<1>(0, i).unwrap();
            let trace_buffer = LetoArray1::from_shape_vec([nsamples], trace.iter().cloned().collect::<Vec<_>>())
                .expect("seismic phase trace length must match its Leto shape");
            let mut buffer = fft_1d_leto(trace_buffer.view());

            let spectrum = buffer
                .as_slice_mut()
                .expect("Apollo 1-D FFT output must be contiguous");
            for sample in spectrum.iter_mut().take(nsamples / 2).skip(1) {
                *sample *= ApolloComplex64::new(2.0, 0.0);
            }
            for sample in spectrum.iter_mut().skip(nsamples / 2 + 1) {
                *sample = ApolloComplex64::new(0.0, 0.0);
            }

            let complex_buffer = LetoArray1::from_shape_vec([nsamples], buffer.into_vec())
                .expect("seismic phase spectrum length must match its Leto shape");
            let analytic_signal = ifft_1d_complex(&complex_buffer);
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
        Ok(0.5 * diff.mapv(|x| x * x).iter().sum::<f64>())
    }
}
