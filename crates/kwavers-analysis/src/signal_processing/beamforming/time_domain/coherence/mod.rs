//! Coherence-factor adaptive weighting for time-domain delay-and-sum (DAS).
//!
//! Coherence factors are per-pixel quality weights that suppress incoherent
//! (off-axis clutter, reverberation, electronic-noise) energy in DAS images by
//! measuring how *coherent* the delay-aligned aperture data is at each output
//! sample. The weight multiplies the DAS output:
//! `y_cf[j] = CF[j] · y_das[j]`, with `CF[j] ∈ [0, 1]`.
//!
//! All estimators consume the **unapodized** delay-aligned aperture matrix from
//! [`super::das::align_channels`] (shape `(n_elements, n_samples)`): coherence
//! measures the wavefront, not the receive taper, so a perfectly coherent
//! wavefront yields `CF = 1` regardless of the DAS apodization (Mallart & Fink
//! 1994).
//!
//! # Variants
//!
//! - [`CoherenceFactor::Amplitude`] — Mallart & Fink (1994) amplitude coherence
//!   factor, the ratio of coherent to total aperture energy.
//! - [`CoherenceFactor::Sign`] — Camacho, Fritsch & Cruza (2009) sign coherence
//!   factor, robust to amplitude outliers.
//! - [`CoherenceFactor::Phase`] — Camacho et al. (2009) phase coherence factor,
//!   the aperture instantaneous-phase dispersion (uses the analytic signal).
//! - [`CoherenceFactor::Generalized`] — Li & Li (2003) generalized coherence
//!   factor, the fraction of aperture spectral energy in a low-spatial-frequency
//!   passband; `m0 = 0` reduces exactly to the amplitude CF.
//!
//! # References
//!
//! - Mallart, R., & Fink, M. (1994). "Adaptive focusing in scattering media
//!   through sound-speed inhomogeneities: The van Cittert–Zernike approach and
//!   focusing criterion." *J. Acoust. Soc. Am.* 96(6), 3721–3732. — coherence
//!   factor as the normalized coherent energy.
//! - Hollman, K. W., Rigby, K. W., & O'Donnell, M. (1999). "Coherence factor of
//!   speckle from a multi-row probe." *IEEE Ultrason. Symp.* — CF imaging use.
//! - Camacho, J., Parrilla, M., & Fritsch, C. (2009). "Phase coherence imaging."
//!   *IEEE Trans. Ultrason. Ferroelectr. Freq. Control* 56(5), 958–974. — sign
//!   and phase coherence factors.
//! - Li, P.-C., & Li, M.-L. (2003). "Adaptive imaging using the generalized
//!   coherence factor." *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*
//!   50(2), 128–141. — GCF as low-spatial-frequency spectral energy ratio.

use super::das::{align_channels, sum_aligned};
use super::delay_reference::DelayReference;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_signal::analytic::hilbert_transform;
use leto::{
    Array1,
    Array2,
    Array3,
};
use eunomia::Complex64;
use std::f64::consts::PI;

/// Amplitude coherence factor from pre-accumulated aperture sums
/// (Mallart & Fink 1994).
///
/// `CF = |coherent_sum|² / (n · sum_of_squares)`, where `coherent_sum = Σᵢ xᵢ`
/// (or the magnitude of the complex/baseband coherent sum) and
/// `sum_of_squares = Σᵢ |xᵢ|²` is the **sum of per-element energies** (not the
/// square of the sum of magnitudes). The Cauchy–Schwarz inequality bounds the
/// ratio to `[0, 1]`, so a perfectly coherent aperture yields `1`. Returns `0`
/// for an empty or zero-energy aperture.
///
/// This is the canonical scalar entry point for accumulator-style beamformers
/// (e.g. SAFT) that build the coherent and energy sums voxel-by-voxel; the
/// matrix-column path ([`CoherenceFactor::weights`]) routes through it too.
#[must_use]
pub fn amplitude_coherence_from_sums(coherent_sum: f64, sum_of_squares: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let denom = (n as f64) * sum_of_squares;
    if denom <= 0.0 {
        return 0.0;
    }
    ((coherent_sum * coherent_sum) / denom).clamp(0.0, 1.0)
}

/// Phase coherence factor from a set of per-element aperture phases
/// (Camacho, Parrilla & Fritsch 2009).
///
/// `PCF = max(0, 1 − (γ / σ₀)·s)`, where `s = min(σ(φ), σ(ψ))` is the smaller of
/// the population standard deviations of the aperture phases `φᵢ` and of the
/// auxiliary phases `ψᵢ = φᵢ − sign(φᵢ)·π`, `σ₀ = π/√3` is the standard
/// deviation of a phase uniformly distributed on `[−π, π]` (full incoherence),
/// and `γ = sensitivity ≥ 0` (canonical `1.0`).
///
/// The auxiliary phase shifts the wrap discontinuity off `±π`, so a wavefront
/// whose phases are coherent but straddle the `±π` branch cut is scored with the
/// small `σ(ψ)` rather than the spuriously large `σ(φ)` — the defining feature of
/// the phase (vs sign) coherence factor. A perfectly coherent aperture
/// (`s = 0`) yields `1`; fully incoherent (`s = σ₀`, `γ = 1`) yields `0`.
/// Returns `0` for an empty aperture.
///
/// This is the canonical scalar entry point; the matrix-column path
/// ([`CoherenceFactor::weights`] with [`CoherenceFactor::Phase`]) derives the
/// per-element phases from the analytic signal and routes through it.
#[must_use]
pub fn phase_coherence_from_phases(phases: &[f64], sensitivity: f64) -> f64 {
    let n = phases.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    // Population standard deviation of a per-phase transform.
    let std_of = |f: &dyn Fn(f64) -> f64| -> f64 {
        let mean = phases.iter().map(|&p| f(p)).sum::<f64>() / nf;
        let var = phases
            .iter()
            .map(|&p| {
                let d = f(p) - mean;
                d * d
            })
            .sum::<f64>()
            / nf;
        var.sqrt()
    };
    let s_phi = std_of(&|p| p);
    let s_aux = std_of(&|p| p - p.signum() * PI);
    let s = s_phi.min(s_aux);
    let sigma_ref = PI / 3.0_f64.sqrt();
    (1.0 - (sensitivity / sigma_ref) * s).clamp(0.0, 1.0)
}

/// Per-element instantaneous phase (`arg` of the analytic signal) of each aligned
/// aperture row, shape `(n_elements, n_samples)` — the input to the phase
/// coherence factor.
fn instantaneous_phase_matrix(aligned: &Array2<f64>) -> Array2<f64> {
    let (n_elements, n_samples) = aligned.dim();
    let mut phase = Array2::<f64>::zeros((n_elements, n_samples));
    for (i, row) in aligned.outer_iter().enumerate() {
        let analytic = hilbert_transform(
            &leto::Array1::from_vec([n_samples], row.iter().copied().collect())
                .expect("coherence row length must match its Leto shape"),
        );
        for (j, z) in analytic.iter().enumerate() {
            phase[[i, j]] = z.arg();
        }
    }
    phase
}

/// Phase coherence factor (Camacho et al. 2009) from a **complex IQ/baseband**
/// aperture `(n_elements, n_samples)`, length-`n_samples` weights in `[0, 1]`.
///
/// Identical to [`CoherenceFactor::Phase`] but takes pre-formed analytic/IQ data
/// directly — each per-element instantaneous phase is `arg(iq[i, j])` — bypassing
/// the Hilbert transform of the real-RF path. Use this when the front end already
/// provides baseband I/Q (e.g. a quadrature demodulator or narrowband snapshot
/// extraction); it routes through the same [`phase_coherence_from_phases`] core.
///
/// # Errors
/// - [`KwaversError::InvalidInput`] on an empty aperture (`n_elements == 0`) or a
///   non-finite / `< 0` sensitivity.
pub fn phase_coherence_from_iq_aperture(
    iq: &Array2<Complex64>,
    sensitivity: f64,
) -> KwaversResult<Array1<f64>> {
    CoherenceFactor::Phase { sensitivity }.validate()?;
    let (n_elements, n_samples) = iq.dim();
    if n_elements == 0 {
        return Err(KwaversError::InvalidInput(
            "phase_coherence_from_iq_aperture requires n_elements > 0".to_owned(),
        ));
    }
    let mut cf = Array1::<f64>::zeros(n_samples);
    let mut phases = vec![0.0_f64; n_elements];
    for j in 0..n_samples {
        for (i, slot) in phases.iter_mut().enumerate() {
            *slot = iq[[i, j]].arg();
        }
        cf[j] = phase_coherence_from_phases(&phases, sensitivity);
    }
    Ok(cf)
}

/// Per-output-sample coherence-factor estimator.
///
/// Each variant maps the delay-aligned aperture column `xⱼ = aligned[:, j]` to a
/// weight `CF[j] ∈ [0, 1]`: `1` for a perfectly coherent aperture, `0` for a
/// fully incoherent one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoherenceFactor {
    /// Mallart & Fink (1994) amplitude coherence factor:
    /// `CF = |Σᵢ xᵢ|² / (N · Σᵢ xᵢ²)`.
    ///
    /// The numerator is the coherent (in-phase) energy and the denominator is
    /// `N×` the incoherent (total) energy; the Cauchy–Schwarz inequality bounds
    /// the ratio to `[0, 1]`. Returns `0` for an all-zero column.
    Amplitude,
    /// Camacho et al. (2009) sign coherence factor:
    /// `SCF = (1 − √(1 − b̄²))^p`, where `b̄ = (1/N) Σᵢ sign(xᵢ)` and
    /// `p = sensitivity ≥ 1`.
    ///
    /// Depends only on the sign pattern across the aperture, so it is robust to
    /// amplitude outliers (bright reflectors, gain mismatch). `sensitivity`
    /// sharpens the taper; `1.0` is the canonical default.
    Sign {
        /// Exponent `p ≥ 1`; larger values reject incoherent energy more aggressively.
        sensitivity: f64,
    },
    /// Camacho et al. (2009) phase coherence factor:
    /// `PCF = max(0, 1 − (γ/σ₀)·min(σ(φ), σ(ψ)))`, where `φᵢ` are the per-element
    /// instantaneous phases (`arg` of the analytic signal), `ψᵢ = φᵢ − sign(φᵢ)·π`
    /// the auxiliary phases, `σ₀ = π/√3`, and `γ = sensitivity`.
    ///
    /// Unlike [`CoherenceFactor::Sign`] (which uses only the sign bit), PCF uses
    /// the full instantaneous phase; the auxiliary-phase minimum makes it immune
    /// to the `±π` branch cut. Computed from the analytic signal of each aperture
    /// row, so the column path requires `n_samples ≥ 2`. `sensitivity = 1.0` is
    /// the canonical default.
    Phase {
        /// Scaling `γ ≥ 0`; larger values reject phase dispersion more aggressively.
        sensitivity: f64,
    },
    /// Li & Li (2003) generalized coherence factor:
    /// `GCF = (Σ_{|k|≤m0} |Xₖ|²) / (Σₖ |Xₖ|²)`, where `Xₖ` is the spatial DFT of
    /// the aperture column across elements.
    ///
    /// The numerator is the aperture spectral energy in the low-spatial-frequency
    /// passband `|k| ≤ m0` (the coherent wavefront concentrates at low spatial
    /// frequency); the denominator is the total energy (`= N·Σ xᵢ²` by Parseval).
    /// `m0 = 0` counts only the DC bin and reduces **exactly** to
    /// [`CoherenceFactor::Amplitude`]; larger `m0` admits more aperture spatial
    /// frequencies as coherent, relaxing the factor toward `1`
    /// (`m0 ≥ N/2 ⇒ GCF = 1`).
    Generalized {
        /// Low-frequency cutoff `M₀`: spatial-frequency bins each side of DC
        /// counted as coherent. `0` ⇒ amplitude CF.
        m0: usize,
    },
}

impl CoherenceFactor {
    /// Validate variant parameters.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if `Sign::sensitivity` is non-finite or
    ///   `< 1`, or if `Phase::sensitivity` is non-finite or `< 0`.
    pub fn validate(&self) -> KwaversResult<()> {
        match *self {
            CoherenceFactor::Sign { sensitivity } => {
                if !sensitivity.is_finite() || sensitivity < 1.0 {
                    return Err(KwaversError::InvalidInput(format!(
                        "CoherenceFactor::Sign requires finite sensitivity >= 1.0; got {sensitivity}"
                    )));
                }
            }
            CoherenceFactor::Phase { sensitivity } => {
                if !sensitivity.is_finite() || sensitivity < 0.0 {
                    return Err(KwaversError::InvalidInput(format!(
                        "CoherenceFactor::Phase requires finite sensitivity >= 0.0; got {sensitivity}"
                    )));
                }
            }
            CoherenceFactor::Amplitude | CoherenceFactor::Generalized { .. } => {}
        }
        Ok(())
    }

    /// Coherence weight for a single delay-aligned aperture column.
    #[must_use]
    fn weight_for_column(&self, column: &[f64]) -> f64 {
        let n = column.len();
        if n == 0 {
            return 0.0;
        }
        match *self {
            CoherenceFactor::Amplitude => {
                let mut coherent = 0.0_f64; // Σ xᵢ
                let mut sum_of_squares = 0.0_f64; // Σ xᵢ²
                for &x in column {
                    coherent += x;
                    sum_of_squares += x * x;
                }
                amplitude_coherence_from_sums(coherent, sum_of_squares, n)
            }
            CoherenceFactor::Sign { sensitivity } => {
                let mut sign_sum = 0.0_f64;
                for &x in column {
                    // Explicit three-way sign: f64::signum(0.0) is +1.0, but the
                    // sign coherence factor requires sign(0) := 0 (NaN ↦ 0 too).
                    sign_sum += if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                }
                let b = sign_sum / (n as f64); // b̄ ∈ [-1, 1]
                let base = 1.0 - (1.0 - b * b).max(0.0).sqrt();
                base.clamp(0.0, 1.0).powf(sensitivity)
            }
            // The `column` here is a column of instantaneous phases (see
            // `weights`), not raw RF amplitudes.
            CoherenceFactor::Phase { sensitivity } => {
                phase_coherence_from_phases(column, sensitivity)
            }
            CoherenceFactor::Generalized { m0 } => {
                // Total aperture energy via Parseval: Σₖ|Xₖ|² = N·Σᵢxᵢ².
                let sum_sq: f64 = column.iter().map(|&x| x * x).sum();
                let denom = (n as f64) * sum_sq;
                if denom <= 0.0 {
                    return 0.0;
                }
                // Spatial DFT energy |Xₖ|² = |Σₜ xₜ e^{-j2πkt/N}|².
                let nf = n as f64;
                let dft_energy = |k: usize| -> f64 {
                    let w = -2.0 * PI * (k as f64) / nf;
                    let (mut re, mut im) = (0.0_f64, 0.0_f64);
                    for (t, &x) in column.iter().enumerate() {
                        let ang = w * (t as f64);
                        re += x * ang.cos();
                        im += x * ang.sin();
                    }
                    re * re + im * im
                };
                // Sum |Xₖ|² over the passband k ∈ {0, ±1, …, ±m0} (mod N). For real
                // input |X_{N−k}| = |Xₖ|, so the ±k pair is `2·|Xₖ|²`; the DC bin
                // (k=0) and, when N is even, the Nyquist bin (k=N/2) are single.
                let kmax = m0.min(n / 2);
                let mut numer = dft_energy(0); // DC always coherent
                for k in 1..=kmax {
                    if k == n - k {
                        numer += dft_energy(k); // Nyquist (N even): single bin
                    } else {
                        numer += 2.0 * dft_energy(k); // ±k conjugate pair
                    }
                }
                (numer / denom).clamp(0.0, 1.0)
            }
        }
    }

    /// Per-output-sample coherence weights in `[0, 1]`, length `aligned.ncols()`.
    ///
    /// `aligned` is the delay-aligned, **unapodized** aperture matrix from
    /// [`align_channels`], shape `(n_elements, n_samples)`.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] on invalid variant parameters or an
    ///   empty aperture (`n_elements == 0`).
    pub fn weights(&self, aligned: &Array2<f64>) -> KwaversResult<Array1<f64>> {
        self.validate()?;
        let (n_elements, n_samples) = aligned.dim();
        if n_elements == 0 {
            return Err(KwaversError::InvalidInput(
                "CoherenceFactor::weights requires n_elements > 0".to_owned(),
            ));
        }
        // The phase CF measures aperture *phase* coherence, so it reduces the
        // per-element instantaneous phase (arg of the analytic signal) rather
        // than the raw RF amplitude; every other variant reduces the RF directly.
        let phase_matrix = match self {
            CoherenceFactor::Phase { .. } => Some(instantaneous_phase_matrix(aligned)),
            _ => None,
        };
        let feature = phase_matrix.as_ref().unwrap_or(aligned);
        let mut cf = Array1::<f64>::zeros(n_samples);
        let mut column = vec![0.0_f64; n_elements];
        for j in 0..n_samples {
            for (i, slot) in column.iter_mut().enumerate() {
                *slot = feature[[i, j]];
            }
            cf[j] = self.weight_for_column(&column);
        }
        Ok(cf)
    }
}

/// Time-domain DAS with coherence-factor adaptive weighting.
///
/// Aligns the aperture once, computes the chosen coherence factor on the raw
/// (unapodized) aligned data, sums with `weights`, and scales each output sample
/// by its coherence weight.
///
/// # Returns
/// `(output, coherence_map)`:
/// - `output`: coherence-weighted beamformed signal, shape `(1, 1, n_samples)`.
/// - `coherence_map`: the per-sample `CF[j] ∈ [0, 1]`, length `n_samples`.
///
/// # Errors
/// - [`KwaversError::InvalidInput`] on DAS contract violations (see
///   [`align_channels`]) or invalid coherence parameters.
pub fn delay_and_sum_coherence(
    sensor_data: &Array3<f64>,
    sampling_frequency_hz: f64,
    delays_s: &[f64],
    weights: &[f64],
    reference: DelayReference,
    factor: CoherenceFactor,
) -> KwaversResult<(Array3<f64>, Array1<f64>)> {
    factor.validate()?;
    let aligned = align_channels(
        sensor_data,
        sampling_frequency_hz,
        delays_s,
        weights,
        reference,
    )?;
    let cf = factor.weights(&aligned)?;
    let mut output = sum_aligned(&aligned, weights);
    for (j, &w) in cf.iter().enumerate() {
        output[[0, 0, j]] *= w;
    }
    Ok((output, cf))
}

#[cfg(test)]
mod tests;

