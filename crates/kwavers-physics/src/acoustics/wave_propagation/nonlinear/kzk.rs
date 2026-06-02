//! Khokhlov-Zabolotskaya-Kuznetsov (KZK) Equation — Physics Trait
//!
//! # KZK Equation
//!
//! The KZK equation (Zabolotskaya & Khokhlov 1969; Kuznetsov 1971) describes
//! the propagation of finite-amplitude acoustic beams in the paraxial
//! approximation:
//!
//! ```text
//! ∂²p/∂z∂τ = (c₀/2) ∇⊥²p
//!           + (δ/(2c₀³)) ∂³p/∂τ³
//!           + (β/(2ρ₀c₀³)) ∂²(p²)/∂τ²
//! ```
//!
//! where:
//! - z: axial (propagation) coordinate (m)
//! - τ = t − z/c₀: retarded time (s)
//! - ∇⊥² = ∂²/∂x² + ∂²/∂y²: transverse Laplacian [m⁻²]
//! - δ: diffusivity of sound [m²/s]
//! - β = 1 + B/(2A): nonlinearity coefficient (dimensionless)
//! - ρ₀: ambient density [kg/m³]
//! - c₀: small-signal sound speed (m/s)
//!
//! # Operator Splitting
//!
//! The three terms — diffraction (D), absorption (A), and nonlinearity (N) —
//! are solved as independent sub-problems and combined by Strang splitting:
//!
//! ```text
//! U(Δz) ≈ D(Δz/2) · A(Δz/2) · N(Δz) · A(Δz/2) · D(Δz/2)
//! ```
//!
//! Strang splitting achieves second-order accuracy in Δz (Strang 1968).
//!
//! # Parabolic Approximation Validity
//!
//! The KZK equation is valid for beams with half-angle divergence θ < ~17°.
//! For wider beams, the exact Westervelt equation should be used instead.
//!
//! # References
//!
//! - Zabolotskaya EA, Khokhlov RV (1969). "Quasi-plane waves in the nonlinear
//!   acoustics of confined beams." Sov. Phys. Acoust. 15(1), 35–40.
//! - Kuznetsov VP (1971). "Equations of nonlinear acoustics."
//!   Sov. Phys. Acoust. 16(4), 467–470.
//! - Aanonsen SI, Barkve T, Tjøtta JN, Tjøtta S (1984). "Distortion and
//!   harmonic generation in the nearfield of a finite amplitude sound beam."
//!   J. Acoust. Soc. Am. 75(3), 749–768. DOI: 10.1121/1.390585
//! - Lee Y-S, Hamilton MF (1995). "Time-domain modeling of pulsed
//!   finite-amplitude sound beams." J. Acoust. Soc. Am. 97(2), 906–917.
//!   DOI: 10.1121/1.412000
//! - Strang G (1968). "On the construction and comparison of difference
//!   schemes." SIAM J. Numer. Anal. 5(3), 506–517. DOI: 10.1137/0705041
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press.

/// Trait for KZK beam propagation solvers.
///
/// Implementations advance the acoustic pressure field axially in the
/// retarded-time frame, one z-plane at a time, using Strang operator
/// splitting for the diffraction, absorption, and nonlinearity sub-steps.
///
/// The trait is intentionally minimal to allow different backends
/// (spectral, finite-difference, GPU) to satisfy it uniformly.
///
/// # Contract
///
/// - After n calls to `step_forward(dz)`, the internal z-coordinate has
///   advanced by n·dz from its initial position.  Steps are monotonically
///   cumulative: the n-th call operates on the state left by the (n−1)-th.
/// - `current_field()` returns the **RMS pressure** in Pa, averaged over
///   retarded time τ at the current z-plane.  Shape: `[nx, ny]`.
/// - `peak_pressure()` returns the peak positive pressure in Pa.  The
///   default implementation returns zeros; backends with direct 3D array
///   access should override it.
///
/// # Example
///
/// ```ignore
/// use kwavers::solver::forward::nonlinear::kzk::{KZKConfig, KZKSolver};
/// use kwavers::physics::acoustics::wave_propagation::nonlinear::kzk::KZKSolverTrait;
/// use ndarray::Array2;
///
/// let config = KZKConfig::default();
/// let dz = config.dz;
/// let mut solver = KZKSolver::new(config).unwrap();
///
/// // Set 5 mm Gaussian source at 1 MHz
/// let source = Array2::from_elem((128, 128), 1.0e5_f64);
/// solver.set_source(source, 1.0e6);
///
/// // March 100 axial steps
/// for _ in 0..100 {
///     solver.step_forward(dz);
/// }
///
/// // Extract 2D RMS pressure at z = 100·dz
/// let field_2d: Array2<f64> = solver.current_field();
/// ```
pub trait KZKSolverTrait {
    /// Advance the acoustic pressure field by one axial step of length `dz` (m).
    ///
    /// Applies the full Strang-split sequence:
    ///   D(dz/2) · A(dz/2) · N(dz) · A(dz/2) · D(dz/2)
    ///
    /// # Arguments
    ///
    /// * `dz` — axial step size in metres.  Must be positive.
    fn step_forward(&mut self, dz: f64);

    /// Return the RMS pressure field (Pa) at the current axial z-plane.
    ///
    /// ## Definition
    ///
    /// ```text
    /// p_rms(i, j) = √( (1/nt) Σ_{t=0}^{nt−1} p[i, j, t]² )     (Pa)
    /// ```
    ///
    /// This is the L² norm of the retarded-time waveform at each (i,j), scaled
    /// by 1/√nt.  It is proportional to the time-averaged acoustic intensity:
    ///
    /// ```text
    /// I(i, j) = p_rms(i, j)² / (ρ₀c₀)    [W/m²]
    /// ```
    ///
    /// (using the plane-wave relation).
    ///
    /// ## Returns
    ///
    /// `Array2<f64>` of shape `(nx, ny)` with units of Pa.
    fn current_field(&self) -> ndarray::Array2<f64>;

    /// Return the peak positive pressure field (Pa) at the current z-plane.
    ///
    /// ## Definition
    ///
    /// ```text
    /// p_peak(i, j) = max_{t} p[i, j, t]
    /// ```
    ///
    /// Relevant for HIFU dosimetry: thermal and mechanical bioeffects correlate
    /// with peak positive pressure (Szabo 2004, §11).
    ///
    /// ## Default implementation
    ///
    /// Returns zeros.  Backends with direct 3D array access should override
    /// this with an efficient implementation.
    fn peak_pressure(&self) -> ndarray::Array2<f64> {
        // Default: return zeros as a sentinel value.
        // Implementations with 3D pressure access should override this.
        let rms = self.current_field();
        ndarray::Array2::zeros(rms.raw_dim())
    }
}
