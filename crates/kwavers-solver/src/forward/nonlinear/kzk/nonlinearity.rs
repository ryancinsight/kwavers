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
//! # Pre-allocated scratch buffer
//!
//! `delta` (nx × ny × nt, f64) is allocated once in `KzkNonlinearOperator::new`
//! and reused across every call to `apply`, eliminating a 128 × 128 × nt × 8 =
//! 131 MB heap allocation per z-step.  The per-(i,j) spectral work buffer `w`
//! is created per Rayon thread (one `nt × 16`-byte allocation per thread,
//! not per spatial point), so no shared `w_scratch` field is needed.
//!
//! # References
//!
//! - Aanonsen SI et al. (1984). "Distortion and harmonic generation in the
//!   nearfield of a finite amplitude sound beam."
//!   J. Acoust. Soc. Am. 75(3), 749–768. DOI: 10.1121/1.390585
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press.
//!   §4.2.1, eq. (4.2.3).

use super::KZKConfig;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::{fft_1d_complex_inplace, ifft_1d_complex_inplace, Complex64};
use ndarray::{Array1, Array3};

/// Nonlinear operator for the KZK equation.
///
/// Encapsulates the quadratic pressure self-interaction term that drives
/// harmonic generation and shock formation.
///
/// `delta` is pre-allocated at construction and reused on every `apply` call
/// to eliminate 131 MB of heap allocation per z-step.  Per-thread spectral
/// work buffers `w` are allocated inside the Rayon closure (one `nt × 16`-byte
/// buffer per thread), giving race-free parallel execution without a shared
/// mutable field.
#[derive(Debug)]
pub struct KzkNonlinearOperator {
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
}

impl KzkNonlinearOperator {
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
        Self {
            beta: 1.0 + config.b_over_a / 2.0, // β = 1 + B/(2A)
            delta,
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
        let ny = self.config.ny;
        // Angular-frequency multiplier per FFT bin:  ω[k] = 2πk/(N·Δτ)
        let two_pi_over_n_dt = TWO_PI / (nt as f64 * dt);

        // Zero the pre-allocated delta buffer (replaces a 131 MB allocation per step).
        self.delta.fill(0.0);

        // Parallelise the (i,j) spatial loop using disjoint i-row slices.
        //
        // Theorem (operator isolation, parallel):
        //   Each i-row of `pressure` and `delta` is disjoint. Parallel writes to
        //   `delta[i, j, t]` for different `i` values are race-free. The full
        //   delta is computed before any pressure update begins (see below), so
        //   the Strang-split invariant (RHS evaluated at consistent input state)
        //   is preserved even with parallel execution.
        //
        // Per-thread scratch `w` replaces the shared `self.w_scratch`;
        // one Array1 allocation of `nt × 16 bytes` per Rayon thread, not per (i,j).
        ndarray::Zip::from(self.delta.axis_iter_mut(ndarray::Axis(0)))
            .and(pressure.axis_iter(ndarray::Axis(0)))
            .par_for_each(|mut delta_row, p_row| {
                let mut w = Array1::<Complex64>::zeros(nt);
                for j in 0..ny {
                    // Step 1: fill per-thread scratch with Re[p] as complex input.
                    for t in 0..nt {
                        w[t] = Complex64::new(p_row[[j, t]].re, 0.0);
                    }

                    // Step 2: forward FFT (no normalisation).
                    fft_1d_complex_inplace(&mut w);

                    // Step 3: multiply each bin by iω[k]
                    for k in 0..nt {
                        let freq_k = if k <= nt / 2 {
                            k as f64
                        } else {
                            k as f64 - nt as f64
                        };
                        let omega_k = freq_k * two_pi_over_n_dt;
                        let re = w[k].re;
                        let im = w[k].im;
                        w[k] = Complex64::new(-im * omega_k, re * omega_k);
                    }
                    // Zero Nyquist to guarantee real-valued derivative.
                    if nt.is_multiple_of(2) {
                        w[nt / 2] = Complex64::new(0.0, 0.0);
                    }

                    // Step 4: inverse FFT (includes 1/N normalisation).
                    ifft_1d_complex_inplace(&mut w);

                    // Step 5: δp[j,t] = coeff · p[j,t] · (∂p/∂τ)[j,t]
                    for t in 0..nt {
                        let p_t = p_row[[j, t]].re;
                        let dp_dt = w[t].re; // exact spectral derivative
                        delta_row[[j, t]] = coeff * p_t * dp_dt;
                    }
                }
            });

        // Apply corrections to the real (physical) component only.
        // Im[p] carries the quadrature component for diffraction phase tracking
        // and does not participate in nonlinear self-coupling.
        // Parallelise the application pass over i-rows as well.
        ndarray::Zip::from(pressure.axis_iter_mut(ndarray::Axis(0)))
            .and(self.delta.axis_iter(ndarray::Axis(0)))
            .par_for_each(|mut p_row, delta_row| {
                for j in 0..ny {
                    for t in 0..nt {
                        p_row[[j, t]].re += delta_row[[j, t]];
                    }
                }
            });
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
        let omega = TWO_PI * frequency;
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

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::{
        DENSITY_TISSUE, DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM,
    };
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use kwavers_core::constants::tissue_acoustics::B_OVER_A_WATER;
    use crate::forward::nonlinear::kzk::KZKConfig;

    /// Helper: minimal `KZKConfig` for unit tests of `KzkNonlinearOperator`.
    /// Grid dimensions are kept small (1×1×1, nt=4) because `KzkNonlinearOperator::new`
    /// allocates a `(nx, ny, nt)` scratch buffer; the nonlinear sub-step math is
    /// independent of the grid size for the pure formula tests below.
    fn minimal_config(c0: f64, rho0: f64, b_over_a: f64, frequency: f64) -> KZKConfig {
        KZKConfig {
            nx: 1,
            ny: 1,
            nz: 1,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            nt: 4,
            c0,
            rho0,
            b_over_a,
            alpha0: 0.0,
            alpha_power: 1.0,
            include_diffraction: false,
            include_absorption: false,
            include_nonlinearity: true,
            frequency,
        }
    }

    /// **Theorem (shock distance, Fubini solution):**
    ///
    /// For a monochromatic plane wave of amplitude p₀ and angular frequency ω
    /// propagating through a lossless medium with nonlinearity coefficient β,
    /// the shock formation distance is:
    ///
    /// ```text
    /// z_shock = ρ₀c₀³ / (β · ω · p₀)
    /// ```
    ///
    /// Reference: Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics §4.3, eq. (4.3.5).
    ///
    /// **Verification:** with ρ₀ = 1000 kg/m³, c₀ = 1500 m/s, B/A = 5 (β = 3.5),
    /// f = 1 MHz (ω = 2π×10⁶ rad/s), p₀ = 10⁵ Pa:
    ///
    /// ```text
    /// z_shock = 1000 × 1500³ / (3.5 × 2π×10⁶ × 10⁵)
    ///         = 3.375×10¹² / (2.199×10¹²)
    ///         ≈ 1.535 m
    /// ```
    #[test]
    fn shock_distance_matches_analytical_formula() {
        let c0 = SOUND_SPEED_WATER_SIM;
        let rho0 = DENSITY_WATER_NOMINAL;
        let b_over_a = 5.0_f64; // water at 25°C (Beyer 1960)
        let frequency = MHZ_TO_HZ; // 1 MHz
        let p0 = 1.0e5_f64; // 100 kPa source amplitude

        let config = minimal_config(c0, rho0, b_over_a, frequency);
        let op = KzkNonlinearOperator::new(&config);

        let z_shock_computed = op.shock_distance(frequency, p0);

        // Analytical: z_shock = ρ₀c₀³ / (β·ω·p₀)
        let beta = 1.0 + b_over_a / 2.0; // = 3.5
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let z_shock_analytic = rho0 * c0.powi(3) / (beta * omega * p0);

        let rel_err = (z_shock_computed - z_shock_analytic).abs() / z_shock_analytic;
        assert!(
            rel_err < 1e-12,
            "shock_distance: computed={z_shock_computed:.6e} analytic={z_shock_analytic:.6e} \
             rel_err={rel_err:.2e} (must be < 1e-12)"
        );
        // Dimensional sanity: should be ~1.535 m for these parameters
        assert!(
            (0.5..5.0).contains(&z_shock_computed),
            "z_shock = {z_shock_computed:.4} m should be in [0.5, 5.0] m for water at 1 MHz"
        );
    }

    /// **Theorem (Goldberg number scaling):**
    ///
    /// The Goldberg (Gol'dberg) number Γ = z / z_shock satisfies:
    ///
    /// - Γ(z = z_shock) = 1.0  exactly (shock formation onset).
    /// - Γ(z = 0) = 0.0  exactly (no nonlinear distortion at source).
    /// - Γ is linear in z: Γ(2·z_shock) = 2.0.
    /// - Γ is monotone: Γ(z₁) < Γ(z₂) for 0 < z₁ < z₂.
    ///
    /// These are direct consequences of the linear formula Γ = β·ω·p₀·z/(ρ₀c₀³).
    #[test]
    fn goldberg_number_at_shock_distance_equals_one() {
        let c0 = SOUND_SPEED_TISSUE;
        let rho0 = DENSITY_TISSUE;
        let b_over_a = 6.5_f64; // soft tissue mid-range (Duck 1990)
        let frequency = 500.0e3_f64; // 500 kHz
        let p0 = 5.0e4_f64; // 50 kPa

        let config = minimal_config(c0, rho0, b_over_a, frequency);
        let op = KzkNonlinearOperator::new(&config);

        let z_shock = op.shock_distance(frequency, p0);

        // Γ(z = 0) = 0
        let gamma_at_zero = op.goldberg_number(0.0, frequency, p0);
        assert_eq!(gamma_at_zero, 0.0, "Γ(z=0) must be exactly 0");

        // Γ(z = z_shock) = 1 to floating-point precision
        let gamma_at_shock = op.goldberg_number(z_shock, frequency, p0);
        assert!(
            (gamma_at_shock - 1.0).abs() < 1e-12,
            "Γ(z=z_shock) must equal 1.0; got {gamma_at_shock:.15}"
        );

        // Γ(z = 2·z_shock) = 2 (linearity in z)
        let gamma_at_double = op.goldberg_number(2.0 * z_shock, frequency, p0);
        assert!(
            (gamma_at_double - 2.0).abs() < 1e-12,
            "Γ(2·z_shock) must equal 2.0; got {gamma_at_double:.15}"
        );

        // Γ > 1 past shock formation (post-shock regime)
        let gamma_past_shock = op.goldberg_number(1.1 * z_shock, frequency, p0);
        assert!(
            gamma_past_shock > 1.0,
            "Γ must exceed 1 at z = 1.1·z_shock; got {gamma_past_shock:.6}"
        );
    }

    /// **Theorem (Goldberg number increases with amplitude):**
    ///
    /// For fixed z, f, and medium: Γ ∝ p₀ (linear in source amplitude).
    /// Doubling p₀ halves z_shock and thus doubles Γ at any fixed z.
    #[test]
    fn goldberg_number_scales_linearly_with_amplitude() {
        let c0 = SOUND_SPEED_WATER_SIM;
        let rho0 = DENSITY_WATER_NOMINAL;
        let b_over_a = B_OVER_A_WATER;
        let frequency = MHZ_TO_HZ;

        let config = minimal_config(c0, rho0, b_over_a, frequency);
        let op = KzkNonlinearOperator::new(&config);

        let z = 0.5; // fixed observation distance (m)
        let p0_low = 1.0e4_f64;
        let p0_high = 2.0 * p0_low;

        let gamma_low = op.goldberg_number(z, frequency, p0_low);
        let gamma_high = op.goldberg_number(z, frequency, p0_high);

        let ratio = gamma_high / gamma_low;
        assert!(
            (ratio - 2.0).abs() < 1e-12,
            "Γ must scale linearly with p₀: ratio={ratio:.15} (expected 2.0)"
        );
    }
}
