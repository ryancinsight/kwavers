//! Frequency-dependent absorption sub-step operator for the KZK equation.
//!
//! # KZK Absorption Term
//!
//! The absorption sub-problem within the Strang-split KZK solver is:
//!
//! ```text
//! ∂²P̂/∂z = −α(|f|) · P̂(f, z)
//! ```
//!
//! where P̂(f, z) is the Fourier transform of p(τ, z) over retarded time τ,
//! and α(f) = α₀ · |f|^y is the power-law attenuation coefficient in Np/m.
//!
//! # Theorem (spectral exactness)
//!
//! **Statement.** The sub-operator ∂P̂/∂z = −α(f) P̂ has the exact solution
//!   P̂(f, z+Δz) = P̂(f, z) · exp(−α(f) · Δz)
//!
//! Each frequency component is attenuated independently; there is no
//! inter-frequency coupling.  The sub-step is solved exactly (not
//! approximated) within each Strang-split increment.
//!
//! **Proof.** Direct integration of the linear ODE dP̂/dz = −α P̂ with
//! constant coefficient α(f) gives P̂(z) = P̂(0) · exp(−α z).  QED.
//!
//! # Why Spectral (Not Single-Frequency) Absorption Matters
//!
//! A nonlinear pulsed beam contains energy at the fundamental f₀ and all
//! harmonics n·f₀.  Higher harmonics are attenuated more strongly because
//! α(n·f₀) = α₀ · (n·f₀)^y > α₀ · f₀^y.  Applying a single scalar
//! exp(−α(f₀)·Δz) to the entire waveform under-attenuates harmonics and
//! produces unphysically large harmonic amplitudes after many steps.
//!
//! # References
//!
//! - Szabo TL (1994). "Time domain wave equations for lossy media obeying a
//!   frequency power law." J. Acoust. Soc. Am. 96(1), 491–500.
//!   DOI: 10.1121/1.410434
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press. §3.5.
//! - Aanonsen SI et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768.

use super::KZKConfig;
use crate::math::fft::{fft_1d_complex_inplace, ifft_1d_complex_inplace, Complex64};
use ndarray::{s, Array1, Array3};

/// Power-law absorption operator for the KZK equation.
///
/// Implements per-frequency attenuation via DFT → multiply by
/// exp(−α(f)·Δz) for each bin → IDFT.
#[derive(Debug)]
pub struct AbsorptionOperator {
    /// Attenuation coefficient at 1 Hz in Np/(m·Hz^y).
    ///
    /// Converted from the clinical unit dB/(cm·MHz^y) during construction.
    /// The conversion is:  α₀_[Np/(m·Hz^y)] = α₀_[dB/(cm·MHz^y)]
    ///                                          × (100 / 8.686) / (1e6)^y
    ///
    /// Note: the 1/8.686 converts dB → Np (1 Np = 8.686 dB),
    ///       the ×100 converts cm⁻¹ → m⁻¹,
    ///       the /(1e6)^y absorbs the MHz^y into Hz^y.
    alpha0_np_per_m_per_hz_y: f64,
    /// Power-law exponent y (tissue ≈ 1.1, water ≈ 2.0)
    power: f64,
    /// Grid/medium configuration
    config: KZKConfig,
}

impl AbsorptionOperator {
    /// Construct the absorption operator, converting α₀ from clinical to SI units.
    ///
    /// # Arguments
    ///
    /// `config.alpha0`  — attenuation coefficient in dB/(cm·MHz^y)
    /// `config.alpha_power` — power-law exponent y
    ///
    /// # Unit conversion
    ///
    /// ```text
    /// α₀ [Np/(m·Hz^y)] = α₀ [dB/(cm·MHz^y)] × 100 / 8.686 / (1e6)^y
    /// ```
    #[must_use]
    pub fn new(config: &KZKConfig) -> Self {
        // Convert dB/(cm·MHz^y) → Np/(m·Hz^y)
        // Factor breakdown:
        //   × 100       : cm⁻¹ → m⁻¹
        //   / 8.686     : dB → Np (ln(10)/20 ≈ 1/8.686)
        //   / (1e6)^y   : MHz^y → Hz^y  (absorb into Hz^y base)
        let alpha0_np = config.alpha0 * 100.0 / 8.686 / (1.0e6_f64).powf(config.alpha_power);

        Self {
            alpha0_np_per_m_per_hz_y: alpha0_np,
            power: config.alpha_power,
            config: config.clone(),
        }
    }

    /// Apply spectrally-resolved power-law absorption for one axial step `step_size` [m].
    ///
    /// ## Algorithm
    ///
    /// For each spatial point (i, j):
    ///
    /// 1. Copy retarded-time complex waveform into working buffer: `w[t] = p[i, j, t]`.
    /// 2. Forward 1D DFT in-place (no normalisation): `w ← FFT(w)`.
    /// 3. For each frequency bin k:
    ///    ```text
    ///    pos_k = k  if k ≤ nt/2,  else  nt − k      (fold negative freqs)
    ///    f_k   = pos_k / (nt · Δτ)                    [Hz]
    ///    H_k   = exp(−α₀ · f_k^y · Δz)               [dimensionless]
    ///    w[k] *= H_k
    ///    ```
    ///    DC bin (k=0) is left at 1.0 (no attenuation at zero frequency).
    /// 4. Inverse 1D DFT in-place with 1/N normalisation: `w ← IFFT(w)`.
    /// 5. Write complex waveform back: `p[i, j, t] = w[t]`.
    ///
    /// ## Complex-field convention
    ///
    /// The pressure array is stored as `Array3<Complex64>` to support the
    /// complex-field diffraction operator.  Absorption is a linear operator:
    ///   `A[p_r + i·p_i] = A[p_r] + i·A[p_i]`
    ///
    /// Operating on the complex waveform directly (without separating real and
    /// imaginary parts) correctly applies the same spectral attenuation to both
    /// components in a single pass.
    ///
    /// ## Theorem (spectral exactness)
    ///
    /// Each frequency bin is an eigenmode of the absorption operator
    /// ∂/∂z → −α(f).  The exact propagator exp(−α(f)·Δz) applied
    /// independently per bin is the exact (not approximate) solution of the
    /// absorption sub-step ODE.  The only approximation comes from the DFT
    /// discretisation of the continuous spectrum, which is O(1/nt²) for smooth
    /// waveforms.
    ///
    /// ## References
    ///
    /// - Szabo TL (1994). J. Acoust. Soc. Am. 96(1), 491–500. eq. (2),(4).
    /// - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics §3.5.
    pub fn apply(&mut self, pressure: &mut Array3<Complex64>, step_size: f64) {
        let nt = self.config.nt;
        let dt = self.config.dt;
        // Frequency resolution: Δf = 1 / (nt · Δτ)  [Hz]
        let df = 1.0 / (nt as f64 * dt);

        // Pre-compute attenuation mask H[k] for k = 0..nt.
        // This avoids recomputing powf() inside the spatial loop.
        let mut h_mask = vec![1.0_f64; nt];
        for (k, mask_elem) in h_mask.iter_mut().enumerate().skip(1) {
            // Fold negative-frequency bins: DFT bin k > nt/2 corresponds to
            // the physical frequency f = (nt − k) / (nt · Δτ).
            let pos_k = if k <= nt / 2 { k } else { nt - k };
            let freq_hz = pos_k as f64 * df; // Hz
                                             // α(f) = α₀ · f^y  in Np/m
            let alpha = self.alpha0_np_per_m_per_hz_y * freq_hz.powf(self.power);
            *mask_elem = (-alpha * step_size).exp();
        }
        // k = 0 (DC) stays 1.0 — acoustic waves carry zero DC component.

        // Allocate one working buffer; reused across all (i,j).
        let mut waveform = Array1::<Complex64>::zeros(nt);

        for i in 0..self.config.nx {
            for j in 0..self.config.ny {
                // 1. Copy complex waveform into working buffer.
                waveform.assign(&pressure.slice(s![i, j, ..]));

                // 2. Forward DFT in-place (no normalisation).
                fft_1d_complex_inplace(&mut waveform);

                // 3. Apply per-frequency attenuation mask.
                for k in 0..nt {
                    waveform[k] *= h_mask[k];
                }

                // 4. Inverse DFT in-place (includes 1/nt normalisation).
                ifft_1d_complex_inplace(&mut waveform);

                // 5. Write complex waveform back.
                pressure.slice_mut(s![i, j, ..]).assign(&waveform);
            }
        }
    }

    /// Return the plane-wave attenuation coefficient α(f) in Np/m.
    ///
    /// ## Law
    ///
    /// ```text
    /// α(f) = α₀ · |f|^y
    /// ```
    ///
    /// where α₀ has units Np/(m·Hz^y) (stored after unit conversion from the
    /// clinical dB/(cm·MHz^y) input).
    ///
    /// ## Reference
    ///
    /// Szabo TL (1994). J. Acoust. Soc. Am. 96(1), 491–500. eq. (2).
    /// Cobbold RSC (2007). Foundations of Biomedical Ultrasound. §3.2.
    #[must_use]
    pub fn get_absorption(&self, frequency_hz: f64) -> f64 {
        self.alpha0_np_per_m_per_hz_y * frequency_hz.powf(self.power)
    }

    /// Return the 1/e amplitude penetration depth at frequency `frequency_hz` [Hz].
    ///
    /// ```text
    /// d(f) = 1 / α(f)   [m]
    /// ```
    #[must_use]
    pub fn penetration_depth(&self, frequency_hz: f64) -> f64 {
        let alpha = self.get_absorption(frequency_hz);
        if alpha > 0.0 {
            1.0 / alpha
        } else {
            f64::INFINITY
        }
    }
}
