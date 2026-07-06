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
//! # Pre-allocated scratch buffers
//!
//! Strang splitting calls `apply` four times per z-step: twice with
//! `step_size = dz/2` and twice with (effectively) `step_size = dz` for the
//! full nonlinear pass.  The attenuation mask `H[k] = exp(−α(f_k)·Δz)` depends
//! only on `step_size`; both variants are pre-computed in `new()` and stored
//! as `h_mask_half` (dz/2) and `h_mask_full` (dz).  The per-call `Vec`
//! allocation and `powf` re-computation are eliminated on the hot path.
//!
//! Each scheduled slab owns a local `waveform` (`Array1<Complex64>`, length
//! nt) scratch buffer reused across the slab's spatial-point iterations,
//! replacing one 16 KB allocation per (i,j) pair.
//!
//! # References
//!
//! - Szabo TL (1994). "Time domain wave equations for lossy media obeying a
//!   frequency power law." J. Acoust. Soc. Am. 96(1), 491–500.
//!   DOI: 10.1121/1.410434
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press. §3.5.
//! - Aanonsen SI et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768.

use super::KZKConfig;
use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
use kwavers_core::constants::numerical::{CM_TO_M, MHZ_TO_HZ};
use kwavers_math::fft::{fft_1d_complex_inplace, ifft_1d_complex_inplace, Complex64};
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use ndarray::{Array1, Array3};

/// Power-law absorption operator for the KZK equation.
///
/// Implements per-frequency attenuation via DFT → multiply by
/// exp(−α(f)·Δz) for each bin → IDFT.
///
/// `h_mask_half` and `h_mask_full` are pre-computed in `new()` for the two
/// step-size variants used by Strang splitting (dz/2 and dz respectively),
/// eliminating per-call `powf` recomputation and `Vec` allocation.
/// The parallel `apply()` allocates one scratch buffer of `nt × 16` bytes per
/// scheduled slab; no shared mutable scratch is needed.
#[derive(Debug)]
pub struct KzkAbsorptionOperator {
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
    /// Pre-computed attenuation mask `H[k] = exp(−α(f_k) · dz/2)`.
    ///
    /// Length nt.  Applied when `step_size ≈ config.dz / 2`.
    h_mask_half: Vec<f64>,
    /// Pre-computed attenuation mask `H[k] = exp(−α(f_k) · dz)`.
    ///
    /// Length nt.  Applied when `step_size ≈ config.dz`.
    h_mask_full: Vec<f64>,
}

impl KzkAbsorptionOperator {
    /// Construct the absorption operator, converting α₀ from clinical to SI units
    /// and pre-computing both attenuation mask variants.
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
        // Convert dB/(cm·MHz^y) → Np/(m·Hz^y) using SSOT conversion constants:
        //   / CM_TO_M     : cm⁻¹ → m⁻¹             (CM_TO_M = 0.01, divide)
        //   / NP_TO_DB    : dB → Np                (NP_TO_DB = 20/ln10 ≈ 8.686)
        //   / MHZ_TO_HZ^y : MHz^y → Hz^y           (MHZ_TO_HZ = 1e6)
        let alpha0_np = config.alpha0 / CM_TO_M / NP_TO_DB / MHZ_TO_HZ.powf(config.alpha_power);

        // Pre-compute h_mask for both step-size variants used in Strang splitting.
        // Theorem (mask independence): H[k] = exp(-α(f_k) · Δz) depends only on
        // the attenuation profile and step_size, not on the field values.  Pre-
        // computing both variants (dz/2 and dz) eliminates nt powf+exp evaluations
        // per call and a Vec<f64> allocation per call.
        let h_mask_half = Self::build_mask(
            alpha0_np,
            config.alpha_power,
            config.nt,
            config.dt,
            config.dz * 0.5,
        );
        let h_mask_full = Self::build_mask(
            alpha0_np,
            config.alpha_power,
            config.nt,
            config.dt,
            config.dz,
        );

        Self {
            alpha0_np_per_m_per_hz_y: alpha0_np,
            power: config.alpha_power,
            h_mask_half,
            h_mask_full,
            config: config.clone(),
        }
    }

    /// Build the per-frequency attenuation mask H[k] = exp(−α(f_k) · step_size).
    ///
    /// ## Algorithm
    ///
    /// For each DFT bin k = 0..nt:
    /// ```text
    /// pos_k = k  if k ≤ nt/2,  else  nt − k      (fold negative freqs)
    /// f_k   = pos_k / (nt · Δτ)                    (Hz)
    /// H_k   = exp(−α₀ · f_k^y · step_size)         (dimensionless)
    /// ```
    /// DC bin (k=0) stays 1.0 (no attenuation at zero frequency).
    fn build_mask(alpha0_np: f64, power: f64, nt: usize, dt: f64, step_size: f64) -> Vec<f64> {
        let df = 1.0 / (nt as f64 * dt); // Δf = 1/(nt·Δτ)
        let mut mask = vec![1.0_f64; nt];
        for (k, elem) in mask.iter_mut().enumerate().skip(1) {
            let pos_k = if k <= nt / 2 { k } else { nt - k };
            let freq_hz = pos_k as f64 * df;
            let alpha = alpha0_np * freq_hz.powf(power);
            *elem = (-alpha * step_size).exp();
        }
        // k = 0 (DC) stays 1.0.
        mask
    }

    /// Apply spectrally-resolved power-law absorption for one axial step `step_size` (m).
    ///
    /// ## Algorithm
    ///
    /// For each spatial point (i, j):
    ///
    /// 1. Copy retarded-time complex waveform into `self.waveform`: `w[t] = p[i, j, t]`.
    /// 2. Forward 1D DFT in-place (no normalisation): `w ← FFT(w)`.
    /// 3. For each frequency bin k:
    ///    ```text
    ///    w[k] *= H_k
    ///    ```
    ///    where `H_k` is taken from the pre-computed mask matching `step_size`.
    /// 4. Inverse 1D DFT in-place with 1/N normalisation: `w ← IFFT(w)`.
    /// 5. Write complex waveform back: `p[i, j, t] = w[t]`.
    ///
    /// The pre-computed mask is selected by comparing `step_size` against
    /// `config.dz / 2` and `config.dz` with a relative tolerance of 1e-10.
    /// Callers outside the standard Strang-split order will fall through to
    /// the full-step mask, which is physically safe (conservative attenuation).
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
        // Select the pre-computed mask for this step_size.
        // Theorem: H[k] depends only on f_k and step_size; both variants are
        // pre-computed in new(), so no per-call Vec allocation is needed.
        //
        // Tolerance: dz is O(10⁻⁴)–O(10⁻³); 1e-10 * dz is ~1e-13 to 1e-14,
        // well within IEEE 754 double representability for the exact dz/2.0
        // literal that the solver passes.
        let h_mask: &Vec<f64> =
            if self.config.dz.mul_add(-0.5, step_size).abs() <= self.config.dz * 1e-10 {
                &self.h_mask_half
            } else {
                &self.h_mask_full
            };

        // Parallelise over i-slabs: each slab [i, :, :] is disjoint, so Moirai
        // can schedule slab chunks without shared mutable aliases. A slab-local
        // `waveform` scratch replaces the former shared `self.waveform`.
        let nt = self.config.nt;
        let ny = self.config.ny;
        let slab_len = ny * nt;
        let pressure_values = pressure
            .as_slice_memory_order_mut()
            .expect("invariant: KZK absorption pressure is standard-layout");
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            pressure_values,
            slab_len,
            |_i, slab| {
                let mut waveform = Array1::<Complex64>::zeros(nt);
                for row in slab.chunks_exact_mut(nt) {
                    for (w, &p) in waveform.iter_mut().zip(row.iter()) {
                        *w = p;
                    }

                    fft_1d_complex_inplace(&mut waveform);

                    for (w, &h) in waveform.iter_mut().zip(h_mask.iter()) {
                        *w *= h;
                    }

                    ifft_1d_complex_inplace(&mut waveform);

                    for (p, &w) in row.iter_mut().zip(waveform.iter()) {
                        *p = w;
                    }
                }
            },
        );
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

    /// Return the 1/e amplitude penetration depth at frequency `frequency_hz` (Hz).
    ///
    /// ```text
    /// d(f) = 1 / α(f)   (m)
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
