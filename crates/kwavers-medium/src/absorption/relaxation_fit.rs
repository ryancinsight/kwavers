//! Fit a discrete relaxation spectrum to a **heterogeneous power-law**
//! absorption law `α(x, f) = α₀(x)·(f/f_ref)^{γ(x)}`.
//!
//! # Why a fit rather than a closed form
//!
//! The Fung (1993) weighting `ΔMₗ ∝ τₗ^{1−γ}` used by
//! [`crate::viscoelastic::GeneralizedMaxwellModel::power_law`] reproduces the
//! *slope* `γ` only asymptotically in the band interior and only for `γ` near
//! unity; near the band edges, and for the low (`γ ≈ 0.4`) and high
//! (`γ ≈ 1.6`) exponents observed in tissue, the realized `α(f)` departs from
//! the target by tens of percent. A least-squares fit of the arm strengths on a
//! shared relaxation-time grid removes that error and — critically — lets the
//! **exponent vary voxel to voxel**, because each voxel simply carries its own
//! non-negative strength vector on the *same* `τₗ` grid. A shared `τₗ` grid is
//! what makes heterogeneous `γ` tractable: the time-domain solver allocates one
//! memory field per arm regardless of how many distinct `(α₀, γ)` pairs the
//! medium contains.
//!
//! # Model
//!
//! For `L` Maxwell arms over an equilibrium modulus `M_∞`,
//!
//! ```text
//!   M*(ω) = M_∞ + Σₗ ΔMₗ · iωτₗ/(1 + iωτₗ)
//!   k(ω)  = ω·√(ρ/M*(ω))          α(ω) = |Im k|      c_p(ω) = ω/Re k
//! ```
//!
//! The fit solves, for each voxel,
//!
//! ```text
//!   min ‖ A·ΔM − 1 ‖₂   subject to   ΔM ≥ 0
//!   A[i][l] = ωᵢ² τₗ / ( (1 + (ωᵢτₗ)²) · 2 M'(ωᵢ) c_p(ωᵢ) · α_target(ωᵢ) )
//! ```
//!
//! i.e. the weak-loss linearization `α ≈ ω·M''/(2 M' c_p)` with each row scaled
//! by `1/α_target(ωᵢ)`, so the residual minimized is the **relative** error,
//! uniform across a band over which `α` spans a decade. `M'(ω)` and the
//! equilibrium modulus are then refreshed from the solution and the fit
//! repeated (a fixed point that converges in a handful of passes, since the
//! loss correction to `M'` is second order in `α/k`). Each pass recalibrates
//! `M_∞` by bisection so the **dispersive phase velocity at `f_ref` equals the
//! prescribed `c₀`** — without that step the medium propagates fast by the
//! full Kramers–Krönig dispersion increment, a systematic time-of-flight error.
//!
//! Accuracy is reported as the maximum relative error of the **exact** `α`
//! (evaluated from `M*(ω)`, not the linearization) against the target over the
//! fit band, so a caller can gate on it rather than trust the fit blindly.
//!
//! # References
//!
//! - Emmerich, H. & Korn, M. (1987). "Incorporation of attenuation into
//!   time-domain computations of seismic wave fields." *Geophysics* 52(9),
//!   1252–1264. (Least-squares relaxation-spectrum fitting.)
//! - Blanch, J.O., Robertsson, J.O.A. & Symes, W.W. (1995). "Modeling of a
//!   constant Q." *Geophysics* 60(1), 176–184.
//! - Pinton, G.F., Dahl, J., Rosenzweig, S. & Trahey, G.E. (2009). "A
//!   heterogeneous nonlinear attenuating full-wave model of ultrasound."
//!   *IEEE Trans. UFFC* 56(3), 474–488.
//! - Treeby, B.E. & Cox, B.T. (2010). "Modeling power law absorption and
//!   dispersion for acoustic propagation using the fractional Laplacian."
//!   *JASA* 127(5), 2741–2748.

use std::collections::HashMap;

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::Complex64;
use leto::{Array1, Array2, Array3};
use leto_ops::{nnls, NnlsConfig};

/// `2π`.
const TWO_PI: f64 = std::f64::consts::TAU;

/// Fixed-point passes refreshing `M'(ω)` and `M_∞` between least-squares solves.
const REFINEMENT_PASSES: usize = 6;

/// Tikhonov damping `λ` applied to the column-equilibrated design matrix.
///
/// Columns are unit-ℓ₂, so `√λ = 1e-3` damps the fit three orders below the
/// column scale: directions the data constrain are untouched (measured cost at
/// six arms over 0.5–5 MHz: 0.071 % → 0.085 % worst-case error) while the
/// near-null directions of a dense `τ` grid — where the unregularized
/// active-set solve returns strengths that miss the target by over 10 % — are
/// resolved. Below `1e-8` the instability returns; above `1e-4` the bias
/// dominates.
const RIDGE: f64 = 1.0e-6;

/// Bisection steps used to calibrate `M_∞` against the prescribed phase
/// velocity — 80 halvings drive the bracket to `f64` resolution.
const CALIBRATION_STEPS: usize = 80;

/// The power law a voxel's absorption must follow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerLawTarget {
    /// Amplitude absorption `α₀` at `f_ref` \[Np·m⁻¹].
    pub alpha_ref_np_m: f64,
    /// Frequency exponent `γ` (dimensionless); tissue spans roughly `0.4…1.6`.
    pub exponent: f64,
    /// Reference frequency `f_ref` \[Hz] at which `α₀` and `c₀` are quoted.
    pub f_ref: f64,
    /// Small-signal phase velocity `c₀` at `f_ref` \[m·s⁻¹].
    pub sound_speed: f64,
    /// Mass density `ρ` \[kg·m⁻³].
    pub density: f64,
}

impl PowerLawTarget {
    /// Target absorption `α₀·(f/f_ref)^γ` \[Np·m⁻¹] at frequency `f` \[Hz].
    #[must_use]
    pub fn alpha_at(&self, f: f64) -> f64 {
        self.alpha_ref_np_m * (f / self.f_ref).powf(self.exponent)
    }

    /// Bit-exact identity key, so a labelled (piecewise-constant) medium fits
    /// once per distinct tissue rather than once per voxel.
    fn key(&self) -> [u64; 5] {
        [
            self.alpha_ref_np_m.to_bits(),
            self.exponent.to_bits(),
            self.f_ref.to_bits(),
            self.sound_speed.to_bits(),
            self.density.to_bits(),
        ]
    }
}

/// Discretization of the fit: relaxation-time placement and frequency sampling.
#[derive(Debug, Clone, Copy)]
pub struct FitBand {
    /// Lower edge of the band the power law must hold over \[Hz].
    pub f_min: f64,
    /// Upper edge of the band the power law must hold over \[Hz].
    pub f_max: f64,
    /// Number of relaxation arms `L`.
    pub n_arms: usize,
    /// Frequency samples (rows of the least-squares system); must be `≥ n_arms`.
    pub n_samples: usize,
    /// Decades by which the `τₗ` grid extends beyond the band on each side.
    ///
    /// Relaxation arms peak at `ω = 1/τ`, so arms placed strictly inside the
    /// band have no lever on the edges; a half-decade of padding removes the
    /// characteristic edge roll-off without wasting arms.
    pub tau_padding_decades: f64,
}

impl FitBand {
    /// Band `[f_min, f_max]` with `n_arms` arms, `8·n_arms` frequency samples
    /// (floor 64), and a half-decade of `τ` padding.
    ///
    /// # Errors
    /// `0 < f_min < f_max` and `n_arms ≥ 1` are required.
    pub fn new(f_min: f64, f_max: f64, n_arms: usize) -> KwaversResult<Self> {
        let band = Self {
            f_min,
            f_max,
            n_arms,
            n_samples: (8 * n_arms).max(64),
            tau_padding_decades: 0.5,
        };
        band.validate()?;
        Ok(band)
    }

    /// # Errors
    /// Rejects non-finite or non-monotonic band edges, zero arms, fewer samples
    /// than arms, and negative padding.
    pub fn validate(&self) -> KwaversResult<()> {
        let ok = self.f_min.is_finite()
            && self.f_max.is_finite()
            && self.f_min > 0.0
            && self.f_max > self.f_min
            && self.n_arms >= 1
            && self.n_samples >= self.n_arms
            && self.tau_padding_decades >= 0.0;
        if ok {
            Ok(())
        } else {
            Err(KwaversError::InvalidInput(
                "fit band requires 0 < f_min < f_max, n_arms ≥ 1, n_samples ≥ n_arms, padding ≥ 0"
                    .to_owned(),
            ))
        }
    }

    /// Log-spaced relaxation times `τₗ` \[s], ascending, spanning the padded band.
    #[must_use]
    pub fn relaxation_times(&self) -> Vec<f64> {
        let pad = 10.0_f64.powf(self.tau_padding_decades);
        let tau_min = 1.0 / (TWO_PI * self.f_max * pad);
        let tau_max = pad / (TWO_PI * self.f_min);
        log_spaced(tau_min, tau_max, self.n_arms)
    }

    /// Log-spaced fit frequencies \[Hz], ascending, across `[f_min, f_max]`.
    #[must_use]
    pub fn frequencies(&self) -> Vec<f64> {
        log_spaced(self.f_min, self.f_max, self.n_samples)
    }
}

/// `n` logarithmically spaced points from `lo` to `hi` inclusive (`n == 1`
/// yields the geometric midpoint).
fn log_spaced(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let (ln_lo, ln_hi) = (lo.ln(), hi.ln());
    (0..n)
        .map(|i| {
            let frac = if n == 1 {
                0.5
            } else {
                i as f64 / (n - 1) as f64
            };
            (ln_lo + (ln_hi - ln_lo) * frac).exp()
        })
        .collect()
}

/// A relaxation spectrum reproducing one voxel's power law.
#[derive(Debug, Clone, PartialEq)]
pub struct RelaxationSpectrumFit {
    m_inf: f64,
    weights: Vec<f64>,
    taus: Vec<f64>,
    density: f64,
    max_relative_error: f64,
}

impl RelaxationSpectrumFit {
    /// Equilibrium (relaxed, `ω→0`) modulus `M_∞` \[Pa].
    #[must_use]
    pub fn equilibrium_modulus(&self) -> f64 {
        self.m_inf
    }

    /// Fitted arm strengths `ΔMₗ` \[Pa], aligned with [`Self::relaxation_times`].
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Relaxation times `τₗ` \[s].
    #[must_use]
    pub fn relaxation_times(&self) -> &[f64] {
        &self.taus
    }

    /// Largest `|α_fit − α_target| / α_target` over the fit band, using the
    /// exact `α` of the complex modulus (not the linearization the fit solves).
    #[must_use]
    pub fn max_relative_error(&self) -> f64 {
        self.max_relative_error
    }

    /// Complex modulus `M*(ω)` \[Pa].
    #[must_use]
    pub fn complex_modulus(&self, omega: f64) -> Complex64 {
        complex_modulus(self.m_inf, &self.weights, &self.taus, omega)
    }

    /// Exact amplitude absorption `α(ω) = |Im k|` \[Np·m⁻¹].
    #[must_use]
    pub fn attenuation(&self, omega: f64) -> f64 {
        wavenumber(self.density, self.complex_modulus(omega), omega)
            .im
            .abs()
    }

    /// Dispersive phase velocity `c_p(ω) = ω/Re k` \[m·s⁻¹].
    #[must_use]
    pub fn phase_velocity(&self, omega: f64) -> f64 {
        let re = wavenumber(self.density, self.complex_modulus(omega), omega).re;
        if re > 0.0 {
            omega / re
        } else {
            f64::INFINITY
        }
    }
}

/// `M*(ω) = M_∞ + Σₗ ΔMₗ·iωτₗ/(1+iωτₗ)`.
fn complex_modulus(m_inf: f64, weights: &[f64], taus: &[f64], omega: f64) -> Complex64 {
    let mut m = Complex64::new(m_inf, 0.0);
    for (&dm, &tau) in weights.iter().zip(taus) {
        let iwt = Complex64::new(0.0, omega * tau);
        m += dm * iwt / (1.0 + iwt);
    }
    m
}

/// `k = ω√(ρ/M*)`.
fn wavenumber(density: f64, m: Complex64, omega: f64) -> Complex64 {
    (Complex64::new(density, 0.0) / m).sqrt() * omega
}

/// Storage modulus `M'(ω) = M_∞ + Σₗ ΔMₗ(ωτₗ)²/(1+(ωτₗ)²)` \[Pa].
fn storage_modulus(m_inf: f64, weights: &[f64], taus: &[f64], omega: f64) -> f64 {
    let mut m = m_inf;
    for (&dm, &tau) in weights.iter().zip(taus) {
        let wt2 = (omega * tau).powi(2);
        m += dm * wt2 / (1.0 + wt2);
    }
    m
}

/// Choose `M_∞` so the dispersive phase velocity at `ω_ref` equals `c₀`.
///
/// `c_p(ω_ref)` increases monotonically with `M_∞` at fixed arm strengths, and
/// the relaxation arms only ever *add* stiffness, so the answer is bracketed by
/// `(0, ρc₀²]`; bisection is unconditionally safe here where a Newton step on
/// the complex-square-root residual is not.
fn calibrate_equilibrium_modulus(
    density: f64,
    weights: &[f64],
    taus: &[f64],
    omega_ref: f64,
    c_ref: f64,
) -> f64 {
    let mut lo = 0.0;
    let mut hi = density * c_ref * c_ref;
    for _ in 0..CALIBRATION_STEPS {
        let mid = 0.5 * (lo + hi);
        let m = complex_modulus(mid, weights, taus, omega_ref);
        let re_k = wavenumber(density, m, omega_ref).re;
        // c_p(mid) = ω/re_k; too fast ⇒ M_∞ is too large.
        if re_k > 0.0 && omega_ref / re_k > c_ref {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Fit a non-negative relaxation spectrum to one voxel's power law.
///
/// # Errors
/// Rejects a non-finite or non-positive `c₀`/`ρ`/`f_ref`, a negative or
/// non-finite `α₀`, a non-finite `γ`, an invalid [`FitBand`], or an inner
/// least-squares failure.
pub fn fit_power_law(
    target: &PowerLawTarget,
    band: &FitBand,
) -> KwaversResult<RelaxationSpectrumFit> {
    band.validate()?;
    let valid = target.sound_speed.is_finite()
        && target.sound_speed > 0.0
        && target.density.is_finite()
        && target.density > 0.0
        && target.f_ref.is_finite()
        && target.f_ref > 0.0
        && target.alpha_ref_np_m.is_finite()
        && target.alpha_ref_np_m >= 0.0
        && target.exponent.is_finite();
    if !valid {
        return Err(KwaversError::InvalidInput(
            "power-law target requires ρ, c₀, f_ref > 0, α₀ ≥ 0, and finite γ".to_owned(),
        ));
    }

    let taus = band.relaxation_times();
    let density = target.density;
    let omega_ref = TWO_PI * target.f_ref;

    // A lossless voxel is exactly M_∞ = ρc² with no arms; the fit below would
    // divide by a zero target.
    if target.alpha_ref_np_m == 0.0 {
        return Ok(RelaxationSpectrumFit {
            m_inf: density * target.sound_speed * target.sound_speed,
            weights: vec![0.0; taus.len()],
            taus,
            density,
            max_relative_error: 0.0,
        });
    }

    let freqs = band.frequencies();
    let omegas: Vec<f64> = freqs.iter().map(|f| TWO_PI * f).collect();
    let targets: Vec<f64> = freqs.iter().map(|&f| target.alpha_at(f)).collect();
    if targets.iter().any(|a| !a.is_finite() || *a <= 0.0) {
        return Err(KwaversError::InvalidInput(
            "power-law target is non-positive or overflows across the fit band".to_owned(),
        ));
    }

    let (rows, cols) = (omegas.len(), taus.len());
    // Rows are already relative-error-normalized, so the right-hand side is 1;
    // the trailing `cols` rows are the ridge block, whose target is 0.
    let total_rows = rows + cols;
    let mut rhs_data = vec![1.0_f64; total_rows];
    rhs_data[rows..].fill(0.0);
    let rhs = Array1::from_vec(total_rows, rhs_data).map_err(|e| {
        KwaversError::InvalidInput(format!("relaxation right-hand side rejected: {e}"))
    })?;

    let mut m_inf = density * target.sound_speed * target.sound_speed;
    let mut weights = vec![0.0_f64; cols];
    // Storage modulus per sample; seeded lossless and refreshed each pass.
    let mut m_storage = vec![m_inf; rows];

    for _ in 0..REFINEMENT_PASSES {
        let mut a = vec![0.0_f64; total_rows * cols];
        for (i, (&omega, &alpha_t)) in omegas.iter().zip(&targets).enumerate() {
            let m_prime = m_storage[i];
            let c_p = (m_prime / density).sqrt();
            let scale = omega * omega / (2.0 * m_prime * c_p * alpha_t);
            for (l, &tau) in taus.iter().enumerate() {
                a[i * cols + l] = scale * tau / (1.0 + (omega * tau).powi(2));
            }
        }
        // Column equilibration. Raw entries are `O(1e-8)` (an arm strength is
        // `O(1e8) Pa`), and Lawson–Hanson's stopping test is an *absolute*
        // bound on `max Aᵀ(b − Ax)`; unscaled, it fires before the active set
        // is complete and the fit stalls at a few percent, erratically in the
        // arm count. Normalising each column to unit ℓ₂ puts the multipliers on
        // the same scale as the tolerance. `x` is recovered by dividing out.
        let mut col_norm = vec![0.0_f64; cols];
        for (l, norm) in col_norm.iter_mut().enumerate() {
            let sum_sq: f64 = (0..rows).map(|i| a[i * cols + l].powi(2)).sum();
            *norm = sum_sq.sqrt();
        }
        for i in 0..rows {
            for (l, &norm) in col_norm.iter().enumerate() {
                if norm > 0.0 {
                    a[i * cols + l] /= norm;
                }
            }
        }

        // Ridge block. Densely packed relaxation times are strongly collinear
        // (neighbouring Debye peaks overlap), which leaves the active-set QR
        // near-singular and the returned strengths junk. Damping the *scaled*
        // coefficients by `√λ` with `λ = RIDGE` resolves the degeneracy while
        // shifting a well-conditioned solution far under the reported fit error.
        let sqrt_ridge = RIDGE.sqrt();
        for l in 0..cols {
            a[(rows + l) * cols + l] = sqrt_ridge;
        }

        let a = Array2::from_vec([total_rows, cols], a).map_err(|e| {
            KwaversError::InvalidInput(format!("relaxation design matrix rejected: {e}"))
        })?;
        let config = NnlsConfig {
            max_iterations: 20 * cols + 100,
            tolerance: 1e-12,
        };
        let solution = nnls(&a.view(), &rhs.view(), config).map_err(|e| {
            KwaversError::InvalidInput(format!("relaxation-spectrum NNLS failed: {e}"))
        })?;
        if !solution.converged {
            return Err(KwaversError::InvalidInput(format!(
                "relaxation-spectrum NNLS did not converge in {} iterations \
                 (γ = {}, {cols} arms)",
                solution.iterations, target.exponent
            )));
        }
        for (l, w) in weights.iter_mut().enumerate() {
            *w = if col_norm[l] > 0.0 {
                solution.solution[l] / col_norm[l]
            } else {
                0.0
            };
        }

        // Refresh the two quantities the linearization held fixed.
        m_inf =
            calibrate_equilibrium_modulus(density, &weights, &taus, omega_ref, target.sound_speed);
        for (i, &omega) in omegas.iter().enumerate() {
            m_storage[i] = storage_modulus(m_inf, &weights, &taus, omega);
        }
    }

    let fit = RelaxationSpectrumFit {
        m_inf,
        weights,
        taus,
        density,
        max_relative_error: 0.0,
    };
    let max_relative_error = omegas
        .iter()
        .zip(&targets)
        .map(|(&omega, &alpha_t)| ((fit.attenuation(omega) - alpha_t) / alpha_t).abs())
        .fold(0.0_f64, f64::max);

    Ok(RelaxationSpectrumFit {
        max_relative_error,
        ..fit
    })
}

/// Per-voxel relaxation spectra on one shared `τₗ` grid.
#[derive(Debug, Clone)]
pub struct RelaxationFieldFit {
    m_inf: Array3<f64>,
    weights: Vec<Array3<f64>>,
    taus: Vec<f64>,
    max_relative_error: f64,
}

impl RelaxationFieldFit {
    /// Equilibrium modulus field `M_∞(x)` \[Pa].
    #[must_use]
    pub fn equilibrium_modulus(&self) -> &Array3<f64> {
        &self.m_inf
    }

    /// One strength field `ΔMₗ(x)` \[Pa] per arm, aligned with
    /// [`Self::relaxation_times`].
    #[must_use]
    pub fn weights(&self) -> &[Array3<f64>] {
        &self.weights
    }

    /// Shared relaxation times `τₗ` \[s].
    #[must_use]
    pub fn relaxation_times(&self) -> &[f64] {
        &self.taus
    }

    /// Worst per-voxel [`RelaxationSpectrumFit::max_relative_error`] over the grid.
    #[must_use]
    pub fn max_relative_error(&self) -> f64 {
        self.max_relative_error
    }

    /// Arms as `(ΔMₗ(x), τₗ(x))` field pairs, the form the time-domain
    /// viscoacoustic solver's heterogeneous constructor consumes.
    #[must_use]
    pub fn arm_fields(&self) -> Vec<(Array3<f64>, Array3<f64>)> {
        let shape = self.m_inf.shape();
        self.weights
            .iter()
            .zip(&self.taus)
            .map(|(dm, &tau)| (dm.clone(), Array3::from_elem(shape, tau)))
            .collect()
    }
}

/// Fit a shared-`τ` relaxation spectrum to a **heterogeneous** power law where
/// `α₀`, `γ`, `c₀`, and `ρ` all vary per voxel.
///
/// Voxels whose five defining parameters are bit-identical share one fit, so a
/// tissue-labelled medium costs one least-squares solve per distinct tissue;
/// a smoothly varying medium costs one per voxel.
///
/// # Errors
/// Rejects fields of differing shapes, an invalid [`FitBand`], or any voxel
/// [`fit_power_law`] rejects.
pub fn fit_power_law_fields(
    alpha_ref_np_m: &Array3<f64>,
    exponent: &Array3<f64>,
    sound_speed: &Array3<f64>,
    density: &Array3<f64>,
    f_ref: f64,
    band: &FitBand,
) -> KwaversResult<RelaxationFieldFit> {
    band.validate()?;
    let shape = alpha_ref_np_m.shape();
    if exponent.shape() != shape || sound_speed.shape() != shape || density.shape() != shape {
        return Err(KwaversError::InvalidInput(
            "α₀, γ, c, and ρ fields must share one grid shape".to_owned(),
        ));
    }

    let taus = band.relaxation_times();
    let n_arms = taus.len();
    let mut m_inf = Array3::<f64>::zeros(shape);
    let mut weights = vec![Array3::<f64>::zeros(shape); n_arms];
    let mut cache: HashMap<[u64; 5], RelaxationSpectrumFit> = HashMap::new();
    let mut max_relative_error = 0.0_f64;

    let n = shape[0] * shape[1] * shape[2];
    for flat in 0..n {
        let idx = [
            flat / (shape[1] * shape[2]),
            (flat / shape[2]) % shape[1],
            flat % shape[2],
        ];
        let target = PowerLawTarget {
            alpha_ref_np_m: alpha_ref_np_m[idx],
            exponent: exponent[idx],
            f_ref,
            sound_speed: sound_speed[idx],
            density: density[idx],
        };
        let fit = match cache.entry(target.key()) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(fit_power_law(&target, band)?),
        };
        m_inf[idx] = fit.equilibrium_modulus();
        max_relative_error = max_relative_error.max(fit.max_relative_error());
        for (l, field) in weights.iter_mut().enumerate() {
            field[idx] = fit.weights()[l];
        }
    }

    Ok(RelaxationFieldFit {
        m_inf,
        weights,
        taus,
        max_relative_error,
    })
}

#[cfg(test)]
mod tests;
