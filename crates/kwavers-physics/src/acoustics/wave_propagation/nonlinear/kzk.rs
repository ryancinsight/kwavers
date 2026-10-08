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
//! # Wide-angle extension
//!
//! The classical KZK model advances diffraction with the paraxial propagator
//!
//! ```text
//! H_paraxial(k_T) = exp(-i k_T² Δz / (2k₀))
//! ```
//!
//! which follows from the small-angle expansion
//! `√(k₀²-k_T²) - k₀ ≈ -k_T²/(2k₀)`. For strongly focused or steered beams,
//! the same retarded-time splitting can instead use the exact Helmholtz
//! transfer function
//!
//! ```text
//! H_wide(k_T) = exp(i (kz - k₀) Δz)
//! kz = √(k₀² - k_T²)                  for k_T ≤ k₀
//! kz = i √(k_T² - k₀²), |H| = exp(-|Im kz| Δz)  for k_T > k₀
//! ```
//!
//! so the paraxial KZK solver becomes the narrow-angle limit of the
//! wide-angle formulation rather than a separate physical model.
//!
//! Between these two extremes, rational Padé approximations replace the exact
//! square-root dispersion relation with stable low-order quotients in
//! `s = (k_T / k₀)²`:
//!
//! ```text
//! φ₁₁ = k₀Δz · (−s/2) / (1 − s/4)
//! φ₂₂ = k₀Δz · (−s/2 + s²/4) / (1 − 3s/4 + s²/16)
//! ```
//!
//! The `[1,1]` form extends useful accuracy to roughly 35° beam half-angle and
//! the `[2,2]` form to roughly 55°, while preserving the same retarded-time
//! splitting structure as the Cartesian KZK solver.
//!
//! # Cylindrical KZK
//!
//! For azimuthally symmetric sources the KZK equation reduces to an
//! axisymmetric problem in `(r, τ, z)` with radial Laplacian
//!
//! ```text
//! ∇⊥²p = (1/r) ∂/∂r (r ∂p/∂r)
//! ```
//!
//! A cylindrical solver can therefore replace the 2-D transverse FFT-based
//! diffraction update with a 1-D radial Crank-Nicolson march while leaving the
//! absorption and nonlinearity operators local in radius. This is exact for
//! axisymmetric sources and much cheaper than a full Cartesian transverse grid.
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
//! For wider beams, the paraxial diffraction sub-step loses accuracy; the
//! wide-angle Helmholtz propagator extends the same retarded-time marching
//! framework to large propagating angles while retaining KZK-style operator
//! splitting.
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
/// Implementations may use either the classical paraxial diffraction operator
/// or the wide-angle exact Helmholtz propagator in the retarded-time frame.
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
/// use kwavers_physics::acoustics::wave_propagation::nonlinear::kzk::KZKSolverTrait;
/// use leto::Array2;
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
    fn current_field(&self) -> leto::Array2<f64>;

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
    fn peak_pressure(&self) -> leto::Array2<f64> {
        // Default: return zeros as a sentinel value.
        // Implementations with 3D pressure access should override this.
        let rms = self.current_field();
        leto::Array2::zeros(rms.shape())
    }
}

/// Marker trait for KZK solvers whose diffraction sub-step remains accurate
/// outside the paraxial cone.
///
/// This trait adds no new API beyond [`KZKSolverTrait`]; it documents that the
/// implementation supports the wide-angle retarded-time Helmholtz propagator
///
/// ```text
/// H_wide(k_T) = exp(i (kz - k₀) Δz)
/// ```
///
/// and is therefore suitable for strongly focused apertures, low F-number
/// transducers, and large steering angles where the standard parabolic
/// approximation becomes inaccurate.
pub trait WideAngleKZKSolverTrait: KZKSolverTrait {}

/// Marker trait for cylindrical (axisymmetric) KZK solvers.
///
/// This documents that the implementation advances an axisymmetric field
/// `p(r, τ, z)` and applies diffraction through the cylindrical radial
/// Laplacian
///
/// ```text
/// (1/r) ∂/∂r (r ∂p/∂r)
/// ```
///
/// rather than a full 2-D Cartesian transverse operator.
pub trait CylindricalKZKSolverTrait {}
