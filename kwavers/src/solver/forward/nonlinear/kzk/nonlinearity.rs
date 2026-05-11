//! Nonlinear sub-step operator for the KZK equation.
//!
//! # KZK Nonlinear Term
//!
//! The nonlinear sub-problem within the Strang-split KZK solver is:
//!
//! ```text
//! ∂p/∂z = (β / (ρ₀c₀³)) · p · ∂p/∂τ
//! ```
//!
//! where β = 1 + B/(2A) is the nonlinearity coefficient.  This is the
//! one-dimensional Burgers equation without diffusion, in retarded time.
//!
//! # Complex-field convention
//!
//! The pressure array is stored as `Array3<Complex64>` to support the
//! complex-field parabolic diffraction operator (which maintains both
//! in-phase and quadrature components for correct phase accumulation over
//! many axial steps).  The nonlinear coupling `p · ∂p/∂τ` is a physical
//! quantity involving only the real (in-phase) pressure; the quadrature
//! (imaginary) component does not contribute to nonlinear self-interaction.
//!
//! Corrections `δp` are therefore computed from `Re[p]` and applied to
//! `Re[p]` only; `Im[p]` is left unchanged by this sub-step.
//!
//! # Pre-allocated scratch buffers
//!
//! `delta` (nx × ny × nt, f64) and `w_scratch` (nt, Complex64) are allocated
//! once in `NonlinearOperator::new` and reused across every call to `apply`.
//! This eliminates a 128 × 128 × nt × 8 = 131 MB heap allocation per step
//! (`delta`) and 128 × 128 × nt × 16 = 268 MB of aggregate per-spatial-point
//! allocations (`w_scratch`).
//!
//! # References
//!
//! - Aanonsen SI et al. (1984). "Distortion and harmonic generation in the
//!   nearfield of a finite amplitude sound beam."
//!   J. Acoust. Soc. Am. 75(3), 749–768. DOI: 10.1121/1.390585
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press.
//!   §4.2.1, eq. (4.2.3).

use super::KZKConfig;
use crate::math::fft::{fft_1d_complex_inplace, ifft_1d_complex_inplace, Complex64};
use ndarray::{Array1, Array3};

/// Nonlinear operator for the KZK equation.
///
/// Encapsulates the quadratic pressure self-interaction term that drives
/// harmonic generation and shock formation.
///
/// All per-call scratch memory (`delta`, `w_scratch`) is pre-allocated at
/// construction and reused on every `apply` call to eliminate hot-path
/// heap allocation.
#[derive(Debug)]
pub struct NonlinearOperator {
    /// Nonlinearity coefficient β = 1 + B/(2A)  (dimensionless)
    beta: f64,
    /// Solver configuration (grid sizes, timestep, medium properties)
    config: KZKConfig,
    /// Pre-allocated update buffer δp[i,j,t], shape (nx, ny, nt).
    ///
    /// Sized at construction to match the pressure array dimensions;
    /// `fill(0.0)` is called at the start of each `apply` invocation.
    /// Holding it here eliminates a 131 MB allocation per z-step.
    delta: Array3<f64>,
    /// Pre-allocated 1-D spectral work buffer, length nt.
    ///
    /// Reused across all (i,j) iterations in the inner spatial loop,
    /// replacing per-iteration `collect`-based allocations totalling
    /// nx × ny × nt × 16 bytes ≈ 268 MB per step.
    w_scratch: Array1<Complex64>,
}

impl NonlinearOperator {
    /// Construct the nonlinear operator from solver configuration.
    ///
    /// Computes β = 1 + B/(2A) from the medium ratio `b_over_a` stored in
    /// `config` and pre-allocates the two scratch buffers.
    ///
    /// # Theorem (β definition)
    ///
    /// From the Taylor expansion of the equation of state
    ///   p = ρ₀c₀²(ρ'/ρ₀) + (B/A)/2 · ρ₀c₀²(ρ'/ρ₀)² + …
    /// the nonlinear term in the KZK equation carries the combined coefficient
    ///   β = 1 + B/(2A)
    ///
    /// Reference: Hamilton & Blackstock (1998) §2.3.2, eq. (2.3.10).
    #[must_use]
    pub fn new(config: &KZKConfig) -> Self {
        let delta = Array3::<f64>::zeros((config.nx, config.ny, config.nt));
        let w_scratch = Array1::<Complex64>::zeros(config.nt);
        Self {
            beta: 1.0 + config.b_over_a / 2.0, // β = 1 + B/(2A)
            delta,
            w_scratch,
            config: config.clone(),
        }
    }

    /// Apply the nonlinear sub-step over one axial increment `step_size` (m).
    ///
    /// ## Algorithm
    ///
    /// Solves the isolated nonlinear sub-problem:
    ///
    /// ```text
    /// ∂p/∂z = (β / (ρ₀c₀³)) · p · ∂p/∂τ
    /// ```
    ///
    /// using explicit Euler in z with a central-difference stencil in retarded
    /// time τ:
    ///
    /// ```text
    /// δp[i,j,t] = (β·Δz / (ρ₀c₀³)) · p[i,j,t] · (∂p/∂τ)[i,j,t]
    /// ```
    ///
    /// where ∂p/∂τ is computed by **spectral differentiation** (FFT-based),
    /// which is exact for bandlimited periodic signals.  A central-difference
    /// approximation (∂p/∂τ ≈ (p[t+1]−p[t−1])/(2Δτ)) underestimates the
    /// derivative by the factor sin(ωΔτ)/(ωΔτ); at 8 samples/period this
    /// causes ~10% systematic error in harmonic generation.
    ///
    /// ## Spectral differentiation algorithm
    ///
    /// For each spatial point (i,j):
    /// 1. Fill `w_scratch[t] = Re[pressure[i,j,t]]` as complex input.
    /// 2. Forward FFT: `W[k] = Σ_t w[t] e^{−2πikt/N}`.
    /// 3. Multiply: `W[k] ← W[k] · iω[k]`, where
    ///    `ω[k] = 2πk/(NΔτ)` for `k ≤ N/2`, and `2π(k−N)/(NΔτ)` for `k > N/2`.
    ///    The Nyquist bin (`k=N/2` for even N) is set to zero to preserve
    ///    real-valued output.
    /// 4. Inverse FFT (with 1/N normalisation): `∂p/∂τ[t] = Re[IFFT(W)][t]`.
    ///
    /// All increments `δp` are accumulated into the pre-allocated buffer
    /// `self.delta` **before** being added back to `pressure`.  This ensures
    /// that the entire right-hand side is evaluated at the same z-level (the
    /// input state), satisfying the operator-isolation requirement of Strang
    /// splitting.
    ///
    /// ## Theorem (operator isolation)
    ///
    /// **Statement.** In Strang splitting, each sub-operator must evaluate its
    /// RHS at a consistent input state.  In-place central-difference updates
    /// violate this: at step `t=k`, the stencil references `p[k−1]` which has
    /// already been overwritten at `t=k−1`, introducing an O(Δz·Δτ) consistency
    /// error that accumulates over the propagation.
    ///
    /// **Fix.** Let `δp[i,j,t] = f(p[i,j,t], p[i,j,t±1])` for all (i,j,t)
    /// simultaneously, storing into a separate array `delta`.  Apply
    /// `pressure += delta` as a single vectorised pass after all RHS values
    /// are computed.  The RHS is then evaluated at the consistent input state
    /// for every grid point.
    ///
    /// **Proof of correctness.** Explicit Euler applied to
    ///   dp/dz = g(p) with g evaluated at pⁿ
    /// gives pⁿ⁺¹ = pⁿ + Δz·g(pⁿ).  Buffering enforces this invariant.
    ///
    /// ## References
    ///
    /// - Strang G (1968). SIAM J. Numer. Anal. 5(3), 506–517. DOI:10.1137/0705041
    /// - Aanonsen SI et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768, §3 eq. (10).
    /// - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics §4.2.1 eq. (4.2.3).
    pub fn apply(
        &mut self,
        pressure: &mut Array3<Complex64>,
        _pressure_prev: &Array3<Complex64>,
        step_size: f64,
    ) {
        let dt = self.config.dt;
        // Nonlinearity coefficient in SI: β / (ρ₀c₀³)  [Pa⁻¹·m⁻¹·s]
        let coeff = self.beta * step_size / (self.config.rho0 * self.config.c0.powi(3));

        // Spectral differentiation for ∂p/∂τ — exact for bandlimited periodic
        // signals; avoids the sin(ωΔτ)/(ωΔτ) attenuation of central differences.
        let nt = self.config.nt;
        let nx = self.config.nx;
        let ny = self.config.ny;
        // Angular-frequency multiplier per FFT bin:  ω[k] = 2πk/(N·Δτ)
        let two_pi_over_n_dt = 2.0 * std::f64::consts::PI / (nt as f64 * dt);

        // Zero the pre-allocated delta buffer (replaces a 131 MB allocation per step).
        self.delta.fill(0.0);

        for i in 0..nx {
            for j in 0..ny {
                // Step 1: fill pre-allocated scratch with Re[p] as complex input.
                // Replaces `Array1::collect()` allocation of nt × 16 bytes per (i,j).
                for t in 0..nt {
                    self.w_scratch[t] = Complex64::new(pressure[[i, j, t]].re, 0.0);
                }

                // Step 2: forward FFT (no normalisation)
                fft_1d_complex_inplace(&mut self.w_scratch);

                // Step 3: multiply each bin by iω[k]
                // Positive frequencies: k = 0..nt/2; negative: k = nt/2+1..nt-1
                // Nyquist bin (k = nt/2 for even nt) → zero for real output
                for k in 0..nt {
                    let freq_k = if k <= nt / 2 {
                        k as f64
                    } else {
                        k as f64 - nt as f64
                    };
                    let omega_k = freq_k * two_pi_over_n_dt;
                    // Multiply by iω: (re + i·im) × iω = −im·ω + i·re·ω
                    let re = self.w_scratch[k].re;
                    let im = self.w_scratch[k].im;
                    self.w_scratch[k] = Complex64::new(-im * omega_k, re * omega_k);
                }
                // Zero Nyquist to guarantee real-valued derivative
                if nt.is_multiple_of(2) {
                    self.w_scratch[nt / 2] = Complex64::new(0.0, 0.0);
                }

                // Step 4: inverse FFT (includes 1/N normalisation)
                ifft_1d_complex_inplace(&mut self.w_scratch);

                // Step 5: form δp[t] = coeff · p[t] · (∂p/∂τ)[t]
                for t in 0..nt {
                    let p_t = pressure[[i, j, t]].re;
                    let dp_dt = self.w_scratch[t].re; // exact spectral derivative
                    self.delta[[i, j, t]] = coeff * p_t * dp_dt;
                }
            }
        }

        // Apply corrections to the real (physical) component only.
        // Im[p] carries the quadrature component for diffraction phase tracking
        // and does not participate in nonlinear self-coupling.
        for i in 0..nx {
            for j in 0..ny {
                for t in 0..nt {
                    pressure[[i, j, t]].re += self.delta[[i, j, t]];
                }
            }
        }
    }

    /// Shock formation distance for a plane wave (m).
    ///
    /// ## Formula
    ///
    /// ```text
    /// z_shock = ρ₀c₀³ / (β · ω · p₀)
    /// ```
    ///
    /// where ω = 2πf is the angular frequency and p₀ is the source pressure
    /// amplitude.  This is the distance at which the initially sinusoidal
    /// waveform first develops an infinite gradient (Fubini solution,
    /// Γ = z/z_shock = 1).
    ///
    /// ## Reference
    ///
    /// Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics §4.3, eq. (4.3.5).
    #[must_use]
    pub fn shock_distance(&self, frequency: f64, amplitude: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        self.config.rho0 * self.config.c0.powi(3) / (self.beta * omega * amplitude)
    }

    /// Gol'dberg number (dimensionless nonlinearity strength) at axial distance z.
    ///
    /// ## Formula
    ///
    /// ```text
    /// Γ = z / z_shock = β · ω · p₀ · z / (ρ₀c₀³)
    /// ```
    ///
    /// Γ < 1: weakly nonlinear (Fubini regime, harmonic amplitudes ~ Γⁿ).
    /// Γ ≈ 1: shock formation.
    /// Γ > 1: sawtooth wave regime (Blackstock solution).
    ///
    /// ## Reference
    ///
    /// Aanonsen SI et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768,
    ///   eq. (1) and surrounding discussion.
    #[must_use]
    pub fn goldberg_number(&self, z: f64, frequency: f64, amplitude: f64) -> f64 {
        z / self.shock_distance(frequency, amplitude)
    }
}
