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
//! # Placing the relaxation times
//!
//! Fixing `τₗ` on a log-spaced grid makes the fit linear, but it wastes arms: two
//! arms cannot span a decade that way and miss the target by ~30 %. Since the
//! strengths follow from the times by one least-squares solve, the times can be
//! searched over directly with the strengths eliminated at every trial — the
//! variable-projection structure. [`RelaxationTimePlacement::Optimized`] does
//! that with a deterministic Nelder–Mead simplex on `ln τ`, and it changes what
//! arm counts are usable:
//!
//! | arms | log-spaced | optimized |
//! |---|---|---|
//! | 2 | 29.9 % | **2.0 %** |
//! | 3 | 2.6 % | **0.16 %** |
//! | 4 | 0.77 % | **0.07 %** |
//! | 6 | 0.09 % | **0.004 %** |
//!
//! (worst case over `α₀ = 0.5` dB·cm⁻¹·MHz⁻ᵞ, `γ = 0.4…1.6`, 0.5–5 MHz.)
//!
//! This matters because the time-domain solver carries **one memory field per
//! arm per voxel**: arm count, not fit quality, is what decides whether a 3-D
//! heterogeneous run fits in memory. Three optimized arms beat six log-spaced
//! ones. Fullwave 2.5 reaches the same conclusion from the other direction — it
//! ships a precomputed database fitted at two relaxation mechanisms.
//!
//! For a **heterogeneous** medium the times are one grid for the whole domain
//! (the solver cannot hold per-voxel times), so [`fit_power_law_fields`] runs a
//! single minimax search over every distinct voxel rather than optimizing each
//! independently.
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
use moirai_parallel::{enumerate_mut_with, AdaptiveWithThreshold};

/// `2π`.
const TWO_PI: f64 = std::f64::consts::TAU;

/// Fixed-point passes refreshing `M'(ω)` and `M_∞` between least-squares solves.
const REFINEMENT_PASSES: usize = 6;

/// Passes used inside the relaxation-time search. The fixed point is stable
/// to six digits by the third pass, so the extra three only refine a number
/// the search is about to discard; halving them halves the search cost.
const SEARCH_PASSES: usize = 3;

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

/// Largest ensemble the shared relaxation-time search runs against.
///
/// Sixty-four keeps the worst-case fit error within a fiftieth of a percentage
/// point of the full-ensemble result while making the search independent of grid
/// size; see [`representative_ensemble`].
const TAU_SEARCH_ENSEMBLE_CAP: usize = 64;

/// Distinct-voxel count at or above which the per-voxel solves are distributed.
///
/// Derived rather than defaulted; the reasoning is at the call site in
/// `fit_power_law_fields`.
const PARALLEL_FIT_THRESHOLD: usize = 16;

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

/// How the relaxation times are placed.
///
/// The strengths are always fitted; this chooses whether the *times* are too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaxationTimePlacement {
    /// Fixed log-spaced grid across the padded band.
    ///
    /// One non-negative least-squares solve, no search. Accurate once the grid
    /// is dense enough to cover the band (about six arms per decade), and the
    /// cheapest option when arm count is not the binding constraint.
    LogSpaced,
    /// Times optimized jointly with the strengths.
    ///
    /// The log-spaced grid seeds a Nelder-Mead search on `ln tau` whose every
    /// trial recovers the strengths exactly by non-negative least squares
    /// (variable projection). This buys accuracy *per arm*: the same target that
    /// needs six log-spaced arms is met by two optimized ones, and each arm the
    /// solver drops is one fewer memory field per voxel. Costs one search per
    /// distinct `(alpha_0, gamma, f_ref, c_0, rho)` tuple at construction, which
    /// a tissue-labelled medium pays once per tissue.
    Optimized,
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
    /// Whether the relaxation times are optimized or left on the log grid.
    pub placement: RelaxationTimePlacement,
}

impl FitBand {
    /// Band `[f_min, f_max]` with `n_arms` arms, `8·n_arms` frequency samples
    /// (floor 64), a half-decade of `τ` padding, and
    /// [`RelaxationTimePlacement::Optimized`] times.
    ///
    /// Optimized placement is the default because it is never worse than the
    /// log-spaced grid (which seeds it) and is dramatically better at the low
    /// arm counts a time-domain solver actually wants to pay for. Set
    /// [`FitBand::placement`] to opt out.
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
            placement: RelaxationTimePlacement::Optimized,
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

/// The **linear** half of the fit: with the relaxation times held fixed, the arm
/// strengths enter `alpha` linearly, so they follow from one non-negative
/// least-squares solve. Separating the problem this way is what makes optimizing
/// the times tractable - the outer search only ever sees `L` numbers, and the
/// `L` strengths are recovered exactly at each trial rather than searched over
/// (the variable-projection structure; Golub & Pereyra 1973).
struct LinearSubproblem<'a> {
    density: f64,
    sound_speed: f64,
    omega_ref: f64,
    exponent: f64,
    omegas: &'a [f64],
    targets: &'a [f64],
}

impl LinearSubproblem<'_> {
    /// Solve for `(M_inf, dM)` at the given relaxation times, running
    /// `passes` fixed-point refreshes of `M'` and `M_inf`.
    fn solve(&self, taus: &[f64], passes: usize) -> KwaversResult<(f64, Vec<f64>)> {
        let (rows, cols) = (self.omegas.len(), taus.len());
        // Rows are relative-error-normalized, so the right-hand side is 1; the
        // trailing `cols` rows are the ridge block, whose target is 0.
        let total_rows = rows + cols;
        let mut rhs_data = vec![1.0_f64; total_rows];
        rhs_data[rows..].fill(0.0);
        let rhs = Array1::from_vec(total_rows, rhs_data).map_err(|e| {
            KwaversError::InvalidInput(format!("relaxation right-hand side rejected: {e}"))
        })?;

        let mut m_inf = self.density * self.sound_speed * self.sound_speed;
        let mut weights = vec![0.0_f64; cols];
        // Storage modulus per sample; seeded lossless and refreshed each pass.
        let mut m_storage = vec![m_inf; rows];

        for _ in 0..passes {
            let mut a = vec![0.0_f64; total_rows * cols];
            for (i, (&omega, &alpha_t)) in self.omegas.iter().zip(self.targets).enumerate() {
                let m_prime = m_storage[i];
                let c_p = (m_prime / self.density).sqrt();
                let scale = omega * omega / (2.0 * m_prime * c_p * alpha_t);
                for (l, &tau) in taus.iter().enumerate() {
                    a[i * cols + l] = scale * tau / (1.0 + (omega * tau).powi(2));
                }
            }

            // Column equilibration. Raw entries are `O(1e-8)` (an arm strength
            // is `O(1e8) Pa`), and Lawson-Hanson's stopping test is an
            // *absolute* bound on `max A^T(b - Ax)`; unscaled, it fires before
            // the active set is complete and the fit stalls at a few percent,
            // erratically in the arm count. Normalising each column to unit L2
            // puts the multipliers on the same scale as the tolerance. `x` is
            // recovered by dividing out.
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

            // Ridge block. Densely packed relaxation times are strongly
            // collinear (neighbouring Debye peaks overlap), which leaves the
            // active-set QR near-singular and the returned strengths junk.
            // Damping the *scaled* coefficients by `sqrt(lambda)` with
            // `lambda = RIDGE` resolves the degeneracy while shifting a
            // well-conditioned solution far under the reported fit error.
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
                     (gamma = {}, {cols} arms)",
                    solution.iterations, self.exponent
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
            m_inf = calibrate_equilibrium_modulus(
                self.density,
                &weights,
                taus,
                self.omega_ref,
                self.sound_speed,
            );
            for (i, &omega) in self.omegas.iter().enumerate() {
                m_storage[i] = storage_modulus(m_inf, &weights, taus, omega);
            }
        }

        Ok((m_inf, weights))
    }

    /// Worst relative error in the **exact** `alpha` over the band at these
    /// relaxation times - the objective the outer search minimizes.
    ///
    /// A rejected trial (out of range, or an inner solve that fails) scores
    /// infinity, which the simplex reflects away from rather than crashing on.
    fn objective(&self, taus: &[f64]) -> f64 {
        let Ok((m_inf, weights)) = self.solve(taus, SEARCH_PASSES) else {
            return f64::INFINITY;
        };
        self.omegas
            .iter()
            .zip(self.targets)
            .map(|(&omega, &alpha_t)| {
                let m = complex_modulus(m_inf, &weights, taus, omega);
                ((wavenumber(self.density, m, omega).im.abs() - alpha_t) / alpha_t).abs()
            })
            .fold(0.0_f64, f64::max)
    }
}

/// Nelder-Mead simplex steps per dimension before the search gives up.
const SIMPLEX_ITERATIONS_PER_ARM: usize = 120;

/// Initial simplex edge in `ln tau`. Half a natural log is ~0.2 decades - a step
/// large enough to explore between neighbouring arms without leaving the band.
const SIMPLEX_EDGE: f64 = 0.5;

/// Decades beyond the padded band that a relaxation time may wander before the
/// trial is rejected. Arms far outside the band contribute an almost frequency-
/// flat loss, which the optimizer will happily exploit to shave the objective at
/// the cost of a spectrum that is nonsense outside the fit window.
const TAU_EXCURSION_DECADES: f64 = 2.0;

/// Optimize **one shared** relaxation-time grid by Nelder-Mead on `ln tau`,
/// with the strengths eliminated at every trial by [`LinearSubproblem::solve`].
///
/// The objective is the worst relative error over *every* problem in
/// `problems` - a minimax over the ensemble. That is what keeps a heterogeneous
/// medium on a single grid: the solver carries one memory field per arm for the
/// whole domain, so the times cannot be tuned per voxel. A single-voxel fit is
/// just the one-element case.
///
/// Deterministic: the initial simplex is the log-spaced grid plus fixed unit
/// steps, so the same inputs always produce the same times. Returns the starting
/// grid unchanged if the search cannot improve on it, so this can only help -
/// the log-spaced placement is the initial vertex.
fn optimize_relaxation_times(
    problems: &[LinearSubproblem<'_>],
    band: &FitBand,
    initial: &[f64],
) -> Vec<f64> {
    let n = initial.len();
    if n == 0 {
        return initial.to_vec();
    }
    let excursion = TAU_EXCURSION_DECADES * std::f64::consts::LN_10;
    let pad = band.tau_padding_decades * std::f64::consts::LN_10;
    let lo = (1.0 / (TWO_PI * band.f_max)).ln() - pad - excursion;
    let hi = (1.0 / (TWO_PI * band.f_min)).ln() + pad + excursion;

    // Sorting makes the parameterization canonical: the basis is symmetric under
    // permuting arms, so two vertices differing only by order are the same
    // spectrum and would otherwise stall the simplex.
    let evaluate = |point: &[f64]| -> (Vec<f64>, f64) {
        if point.iter().any(|v| !v.is_finite() || *v < lo || *v > hi) {
            return (Vec::new(), f64::INFINITY);
        }
        let mut taus: Vec<f64> = point.iter().map(|v| v.exp()).collect();
        taus.sort_by(f64::total_cmp);
        let score = problems
            .iter()
            .map(|p| p.objective(&taus))
            .fold(0.0_f64, f64::max);
        (taus, score)
    };

    let start: Vec<f64> = initial.iter().map(|t| t.ln()).collect();
    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    simplex.push((start.clone(), evaluate(&start).1));
    for axis in 0..n {
        let mut vertex = start.clone();
        vertex[axis] += SIMPLEX_EDGE;
        let score = evaluate(&vertex).1;
        simplex.push((vertex, score));
    }

    let centroid_of = |simplex: &[(Vec<f64>, f64)]| -> Vec<f64> {
        let mut c = vec![0.0_f64; n];
        for (vertex, _) in &simplex[..n] {
            for (ci, vi) in c.iter_mut().zip(vertex) {
                *ci += vi / n as f64;
            }
        }
        c
    };
    let combine = |a: &[f64], b: &[f64], t: f64| -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + t * (x - y)).collect()
    };

    for _ in 0..(SIMPLEX_ITERATIONS_PER_ARM * n) {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let best = simplex[0].1;
        let worst = simplex[n].1;
        // Converged once the vertices agree to a part in 1e4 of the best score.
        // The reported errors are 1e-2..1e-5, so four digits of agreement on the
        // objective is already past the point where more search changes any
        // decision - and every further digit costs simplex evaluations.
        if !worst.is_finite() && !best.is_finite() {
            break;
        }
        if worst.is_finite() && (worst - best).abs() <= 1e-4 * best.abs().max(1e-12) {
            break;
        }

        let centroid = centroid_of(&simplex);
        let reflected = combine(&centroid, &simplex[n].0, 1.0);
        let reflected_score = evaluate(&reflected).1;

        if reflected_score < best {
            let expanded = combine(&centroid, &simplex[n].0, 2.0);
            let expanded_score = evaluate(&expanded).1;
            simplex[n] = if expanded_score < reflected_score {
                (expanded, expanded_score)
            } else {
                (reflected, reflected_score)
            };
        } else if reflected_score < simplex[n - 1].1 {
            simplex[n] = (reflected, reflected_score);
        } else {
            let contracted = combine(&centroid, &simplex[n].0, -0.5);
            let contracted_score = evaluate(&contracted).1;
            if contracted_score < worst {
                simplex[n] = (contracted, contracted_score);
            } else {
                // Shrink toward the best vertex.
                let anchor = simplex[0].0.clone();
                for entry in simplex.iter_mut().skip(1) {
                    let shrunk: Vec<f64> = anchor
                        .iter()
                        .zip(&entry.0)
                        .map(|(a, v)| a + 0.5 * (v - a))
                        .collect();
                    let score = evaluate(&shrunk).1;
                    *entry = (shrunk, score);
                }
            }
        }
    }

    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (best_taus, best_score) = evaluate(&simplex[0].0);
    // Never return a placement worse than the grid we started from.
    let start_score = evaluate(&start).1;
    if best_score.is_finite() && best_score <= start_score && !best_taus.is_empty() {
        best_taus
    } else {
        initial.to_vec()
    }
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

    let density = target.density;

    // A lossless voxel is exactly M_∞ = ρc² with no arms; the fit below would
    // divide by a zero target.
    if target.alpha_ref_np_m == 0.0 {
        let taus = band.relaxation_times();
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
    let problem = LinearSubproblem {
        density,
        sound_speed: target.sound_speed,
        omega_ref: TWO_PI * target.f_ref,
        exponent: target.exponent,
        omegas: &omegas,
        targets: &targets,
    };

    let initial = band.relaxation_times();
    let taus = match band.placement {
        RelaxationTimePlacement::LogSpaced => initial,
        RelaxationTimePlacement::Optimized => {
            optimize_relaxation_times(std::slice::from_ref(&problem), band, &initial)
        }
    };
    fit_at_taus(target, &omegas, &freqs, &taus)
}

/// Solve one target's strengths on an already-chosen relaxation-time grid and
/// score the result against the exact `alpha`.
///
/// The single point at which a `RelaxationSpectrumFit` is constructed, so the
/// per-voxel and whole-field entry points cannot drift in how they solve,
/// calibrate, or score. `omegas` and `freqs` are passed in because the caller
/// has already built them - for a field fit, once for the whole grid.
fn fit_at_taus(
    target: &PowerLawTarget,
    omegas: &[f64],
    freqs: &[f64],
    taus: &[f64],
) -> KwaversResult<RelaxationSpectrumFit> {
    let density = target.density;

    // A lossless voxel is exactly M_inf = rho c^2 with no arms; the solve below
    // would divide by a zero target.
    if target.alpha_ref_np_m == 0.0 {
        return Ok(RelaxationSpectrumFit {
            m_inf: density * target.sound_speed * target.sound_speed,
            weights: vec![0.0; taus.len()],
            taus: taus.to_vec(),
            density,
            max_relative_error: 0.0,
        });
    }

    let targets: Vec<f64> = freqs.iter().map(|&f| target.alpha_at(f)).collect();
    if targets.iter().any(|a| !a.is_finite() || *a <= 0.0) {
        return Err(KwaversError::InvalidInput(
            "power-law target is non-positive or overflows across the fit band".to_owned(),
        ));
    }
    let problem = LinearSubproblem {
        density,
        sound_speed: target.sound_speed,
        omega_ref: TWO_PI * target.f_ref,
        exponent: target.exponent,
        omegas,
        targets: &targets,
    };
    let (m_inf, weights) = problem.solve(taus, REFINEMENT_PASSES)?;

    let fit = RelaxationSpectrumFit {
        m_inf,
        weights,
        taus: taus.to_vec(),
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

/// Fit a shared-`tau` relaxation spectrum to a **heterogeneous** power law where
/// `alpha_0`, `gamma`, `c_0`, and `rho` all vary per voxel.
///
/// The relaxation times are **one grid for the whole domain** - the time-domain
/// solver carries one memory field per arm across every voxel, so per-voxel
/// times are not representable. Under
/// [`RelaxationTimePlacement::Optimized`] that grid is chosen by a single
/// minimax search over every distinct voxel present, so the reported
/// [`RelaxationFieldFit::max_relative_error`] is the worst any voxel suffers on
/// the grid they all share. Only the strengths vary voxel to voxel.
///
/// Voxels whose five defining parameters are bit-identical share one fit, so a
/// tissue-labelled medium costs one strength solve per distinct tissue and one
/// search for the whole medium; a smoothly varying medium costs one strength
/// solve per voxel.
///
/// # Errors
/// Rejects fields of differing shapes, an invalid [`FitBand`], or any voxel
/// whose target [`fit_power_law`] would reject.
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
            "alpha_0, gamma, c, and rho fields must share one grid shape".to_owned(),
        ));
    }

    let index_of = |flat: usize| {
        [
            flat / (shape[1] * shape[2]),
            (flat / shape[2]) % shape[1],
            flat % shape[2],
        ]
    };
    let voxel_count = shape[0] * shape[1] * shape[2];
    let target_at = |flat: usize| {
        let idx = index_of(flat);
        PowerLawTarget {
            alpha_ref_np_m: alpha_ref_np_m[idx],
            exponent: exponent[idx],
            f_ref,
            sound_speed: sound_speed[idx],
            density: density[idx],
        }
    };

    // Distinct targets, in first-appearance order so the result never depends on
    // hash iteration order.
    let mut seen: HashMap<[u64; 5], usize> = HashMap::new();
    let mut distinct: Vec<PowerLawTarget> = Vec::new();
    for flat in 0..voxel_count {
        let target = target_at(flat);
        if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(target.key()) {
            e.insert(distinct.len());
            distinct.push(target);
        }
    }

    // Choose the shared relaxation times against every *lossy* distinct target
    // at once. Lossless voxels impose no constraint (they take no arms), and
    // including their degenerate objective would only pollute the search.
    let freqs = band.frequencies();
    let omegas: Vec<f64> = freqs.iter().map(|f| TWO_PI * f).collect();
    let mut lossy_targets: Vec<Vec<f64>> = Vec::new();
    let mut lossy_index: Vec<usize> = Vec::new();
    for (i, t) in distinct.iter().enumerate() {
        if t.alpha_ref_np_m <= 0.0 {
            continue;
        }
        let alphas: Vec<f64> = freqs.iter().map(|&f| t.alpha_at(f)).collect();
        if alphas.iter().any(|a| !a.is_finite() || *a <= 0.0) {
            return Err(KwaversError::InvalidInput(
                "power-law target is non-positive or overflows across the fit band".to_owned(),
            ));
        }
        lossy_targets.push(alphas);
        lossy_index.push(i);
    }

    let taus = match band.placement {
        RelaxationTimePlacement::LogSpaced => band.relaxation_times(),
        RelaxationTimePlacement::Optimized if lossy_targets.is_empty() => band.relaxation_times(),
        RelaxationTimePlacement::Optimized => {
            let chosen = representative_ensemble(&distinct, &lossy_index);
            let problems: Vec<LinearSubproblem<'_>> = chosen
                .iter()
                .map(|&slot| LinearSubproblem {
                    density: distinct[lossy_index[slot]].density,
                    sound_speed: distinct[lossy_index[slot]].sound_speed,
                    omega_ref: TWO_PI * f_ref,
                    exponent: distinct[lossy_index[slot]].exponent,
                    omegas: &omegas,
                    targets: &lossy_targets[slot],
                })
                .collect();
            let initial = band.relaxation_times();
            optimize_relaxation_times(&problems, band, &initial)
        }
    };

    // Solve each distinct voxel's strengths on the shared grid, in parallel.
    //
    // The solves are independent and `fit_at_taus` is deterministic, so writing
    // each result to its own index reproduces the serial answer *bit for bit* -
    // there is no reduction here whose order could change. That is what
    // `parallel_field_fit_matches_the_serial_solve` pins.
    //
    // The threshold is not the crate default. `Adaptive` parallelizes at 1024
    // elements, sized for cheap per-element work; one element here is a
    // Lawson-Hanson solve over the whole fit band, **measured at 273 us**
    // against a task-dispatch cost around 1 us. Sixteen is well clear of
    // break-even while keeping small labelled media - which dedup to a handful
    // of distinct tuples - on the serial path.
    //
    // Scope worth knowing at this call site: this parallelizes the *solves*,
    // which is 5.9x measured. On the `Optimized` placement it is not where the
    // time goes - the joint tau search above dominates by three orders of
    // magnitude (KW-MED-091).
    let mut slots: Vec<Option<KwaversResult<RelaxationSpectrumFit>>> =
        (0..distinct.len()).map(|_| None).collect();
    enumerate_mut_with::<AdaptiveWithThreshold<PARALLEL_FIT_THRESHOLD>, _, _>(
        &mut slots,
        |i, slot| {
            *slot = Some(fit_at_taus(&distinct[i], &omegas, &freqs, &taus));
        },
    );
    let mut solved: Vec<RelaxationSpectrumFit> = Vec::with_capacity(distinct.len());
    for slot in slots {
        solved.push(slot.expect("invariant: every index is visited exactly once")?);
    }

    let n_arms = taus.len();
    let mut m_inf = Array3::<f64>::zeros(shape);
    let mut weights = vec![Array3::<f64>::zeros(shape); n_arms];
    let mut max_relative_error = 0.0_f64;
    for flat in 0..voxel_count {
        let idx = index_of(flat);
        let slot = seen[&target_at(flat).key()];
        let fit = &solved[slot];
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

/// Choose at most [`TAU_SEARCH_ENSEMBLE_CAP`] lossy targets spanning the
/// `(gamma, alpha_0)` range, returning positions into `lossy_index`.
///
/// # Why the search does not need every voxel
///
/// `optimize_relaxation_times` minimizes the **worst** relative error over the
/// ensemble, and that maximum is governed almost entirely by the exponent.
/// Measured across a 160x range of `alpha_0` at fixed `gamma`, the objective
/// moves only in its fourth significant figure (2.383e-2 to 2.416e-2); across
/// `gamma` it moves by a third, and non-monotonically, peaking near 1.2. So the
/// ensemble needs *coverage of the exponent range*, not one problem per voxel.
///
/// Selection is by rank rather than by stride over the distinct list, because
/// that list is in first-appearance order - a raster scan, which spans the range
/// only for a field that happens to vary monotonically along it. Sorting by
/// `(gamma, alpha_0)` and taking evenly spaced ranks including both endpoints
/// covers the range whatever the voxel ordering, and is deterministic.
///
/// # What this costs
///
/// It is an approximation, and the honest measure of it is the worst-case fit
/// error over the *whole* field under the resulting times - not how far the
/// times themselves moved, which is self-referential. On a 12^3 smoothly
/// varying field: the full 1728-problem search took 30.81 s and left a worst
/// error of 0.2505 %; a 64-problem ensemble took 1.15 s and left 0.2703 %. The
/// search cost also stops scaling with the grid, which is the point - the same
/// field at 128^3 extrapolated to roughly 12 hours before this (KW-MED-091).
fn representative_ensemble(distinct: &[PowerLawTarget], lossy_index: &[usize]) -> Vec<usize> {
    let count = lossy_index.len();
    if count <= TAU_SEARCH_ENSEMBLE_CAP {
        return (0..count).collect();
    }

    let mut ranked: Vec<usize> = (0..count).collect();
    ranked.sort_by(|&a, &b| {
        let (ta, tb) = (&distinct[lossy_index[a]], &distinct[lossy_index[b]]);
        ta.exponent
            .total_cmp(&tb.exponent)
            .then(ta.alpha_ref_np_m.total_cmp(&tb.alpha_ref_np_m))
    });

    // Evenly spaced ranks, endpoints included, so the extremes of the exponent
    // range are always in the ensemble.
    let last = TAU_SEARCH_ENSEMBLE_CAP - 1;
    let mut chosen: Vec<usize> = (0..TAU_SEARCH_ENSEMBLE_CAP)
        .map(|s| ranked[(s * (count - 1)) / last])
        .collect();
    chosen.sort_unstable();
    chosen.dedup();
    chosen
}

#[cfg(test)]
mod tests;
