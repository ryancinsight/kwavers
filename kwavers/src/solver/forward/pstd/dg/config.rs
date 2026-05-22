//! DG Solver Configuration
//!
//! Defines all configuration types for the Discontinuous Galerkin time-domain solver.
//!
//! ## Time integration
//!
//! The default integrator is **SSP-RK3** (Strong Stability Preserving Runge–Kutta order 3,
//! Shu & Osher 1988).  `ForwardEuler` is provided for legacy regression tests only.
//!
//! ## Shock capturing
//!
//! When `ShockCaptureConfig::enabled` is `true`, the DG core applies a
//! conservative troubled-cell projection after the configured RK stages. The
//! standalone WENO limiter module remains available for field-level limiting.
//!
//! ## References
//!
//! - Shu & Osher (1988). "Efficient implementation of essentially non-oscillatory shock capturing
//!   schemes." *J. Comput. Phys.* 77(2):439–471.
//! - Jiang & Shu (1996). "Efficient implementation of weighted ENO schemes."
//!   *J. Comput. Phys.* 126(1):202–228.
//! - Cockburn & Shu (2001). "Runge-Kutta discontinuous Galerkin methods for convection-dominated
//!   problems." *J. Sci. Comput.* 16(3):173–261.

/// Time integration scheme for the DG solver.
///
/// **Theorem (SSP-RK3 TVD property, Shu & Osher 1988 Thm. 2.1):**
/// The SSP-RK3 scheme
/// ```text
///   u⁽¹⁾ = u^n + dt·L(u^n)
///   u⁽²⁾ = (3/4) u^n + (1/4)[u⁽¹⁾ + dt·L(u⁽¹⁾)]
///   u^{n+1} = (1/3) u^n + (2/3)[u⁽²⁾ + dt·L(u⁽²⁾)]
/// ```
/// satisfies the TVD property provided the forward Euler sub-step is TVD:
/// `‖u⁽¹⁾‖_TV ≤ ‖u^n‖_TV`.
///
/// **CFL condition for DG(p):** `CFL ≤ 1/(2p+1)` (Cockburn & Shu 2001 §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DgTimeIntegrator {
    /// Strong Stability Preserving Runge–Kutta, 3rd-order (Shu & Osher 1988).
    ///
    /// Recommended for all production simulations with p ≥ 1.
    #[default]
    SspRk3,
    /// Forward Euler — first-order only.
    ///
    /// Conditionally stable for p = 0; **unconditionally unstable for p ≥ 1** under DG
    /// (Cockburn & Shu 2001 §4).  Provided only for baseline regression comparisons.
    ForwardEuler,
}

/// Degree selector for the WENO shock-capturing limiter.
///
/// - `Weno3` uses two stencils (Jiang & Shu 1996 §2.1), 3rd-order accurate in smooth regions.
/// - `Weno7` uses four stencils, 7th-order accurate in smooth regions.
///   Optimal linear weights: d = [0.05, 0.45, 0.45, 0.05].
///
/// ## Reference
/// Jiang & Shu (1996). *J. Comput. Phys.* 126(1):202–228.
/// Liu, Osher & Chan (1994). *J. Comput. Phys.* 115(1):200–212.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WenoDegree {
    /// WENO3 — two stencils, 3rd-order in smooth regions.
    #[default]
    Weno3,
    /// WENO7 — four stencils, 7th-order in smooth regions.
    Weno7,
}

/// Shock-capture configuration applied after SSP-RK stages.
///
/// When `enabled = true`, the DG time stepper detects troubled elements from
/// quadrature-weighted element-mean jumps and intra-element variation. Flagged
/// elements are replaced by a DG-mass-preserving TVD linear reconstruction
/// before the next sub-stage when `apply_per_stage = true`, and after the final
/// stage in all enabled cases. This damps Gibbs oscillations without changing
/// the element integral represented by the diagonal mass matrix.
///
/// ## Mathematical basis
///
/// The smoothness indicator for WENO3 (Jiang & Shu 1996 eq. 2.17):
/// ```text
///   β₀ = (u_{i+1} − u_i)²
///   β₁ = (u_i − u_{i-1})²
///   ω_k = d_k / (ε + β_k)²,   ε = 1e-6
/// ```
/// For WENO7 four candidates with d = [0.05, 0.45, 0.45, 0.05] (Liu et al. 1994).
#[derive(Debug, Clone, Copy)]
pub struct ShockCaptureConfig {
    /// Enable shock capturing (default: `true` for p ≥ 2, `false` for p ≤ 1).
    pub enabled: bool,
    /// Which WENO degree to apply.
    pub limiter: WenoDegree,
    /// Shock indicator threshold — elements with indicator > threshold are limited.
    /// Typical values: 0.05–0.20 (Persson & Peraire 2006).
    pub threshold: f64,
    /// Apply the limiter after **each** SSP-RK sub-stage (recommended: `true`).
    ///
    /// Setting `false` applies the limiter only at the end of the full step,
    /// which may permit oscillations to build within a step.
    pub apply_per_stage: bool,
}

impl Default for ShockCaptureConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            limiter: WenoDegree::Weno3,
            threshold: 0.1,
            apply_per_stage: true,
        }
    }
}

/// Boundary condition used by DG face fluxes on one Cartesian axis.
///
/// `Periodic` preserves the historical wraparound topology and is the default
/// for conservation proofs. `AbsorbingCharacteristic` replaces exterior
/// boundary states by the one-way acoustic characteristic state with zero
/// incoming wave and is the first open-boundary policy needed before adding
/// DG-native CPML memory variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DgBoundaryCondition {
    /// Couple each exterior face to the opposite-side element.
    #[default]
    Periodic,
    /// Preserve outgoing acoustic characteristics and set incoming
    /// characteristics to zero at physical exterior faces.
    AbsorbingCharacteristic,
}

/// Complete configuration for the Discontinuous Galerkin solver.
#[derive(Debug, Clone, Copy)]
pub struct DGConfig {
    /// Polynomial order p; yields (p+1) nodes per element and O(h^{p+1}) spatial accuracy.
    pub polynomial_order: usize,
    /// Basis type for the DG discretisation (Legendre or Chebyshev).
    pub basis_type: super::basis::BasisType,
    /// Numerical flux type (Lax–Friedrichs upwind recommended for acoustics).
    pub flux_type: super::flux::FluxType,
    /// Time integration scheme (default: `SspRk3`).
    pub time_integrator: DgTimeIntegrator,
    /// Enable slope limiter (legacy field, superseded by `shock_capture`).
    pub use_limiter: bool,
    /// Legacy limiter type selector.
    pub limiter_type: super::flux::LimiterType,
    /// Legacy shock threshold (superseded by `shock_capture.threshold`).
    pub shock_threshold: f64,
    /// Shock-capture configuration (WENO limiter + detector).
    pub shock_capture: ShockCaptureConfig,
    /// Background acoustic wave speed [m/s] used for CFL and Lax–Friedrichs flux.
    ///
    /// Defaults to 1500.0 m/s (water/soft tissue). Override for other media.
    pub sound_speed: f64,
    /// Per-axis boundary conditions for DG face fluxes in `[x, y, z]` order.
    ///
    /// Lower-dimensional simulations ignore inactive axes. Embedded 2-D slab
    /// comparisons can therefore use absorbing in-plane boundaries while
    /// preserving a periodic out-of-plane invariant direction.
    pub boundary_conditions: [DgBoundaryCondition; 3],
    /// Optional CPML configuration for the tensor acoustic DG solver.
    ///
    /// When set, every per-axis spatial derivative used by
    /// `compute_acoustic_tensor_rhs_with_cpml_into` is replaced by the
    /// Roden-Gedney stretched derivative `(1/κ_a) D_a + ψ_{q, a}`, with the
    /// auxiliary memory `ψ` advanced jointly with the field state under the
    /// SSP-RK3 stepper. The outermost element face still uses the per-axis
    /// `boundary_conditions[a]` policy (typically `AbsorbingCharacteristic`)
    /// so any residual outgoing wave that survives the absorbing layer is
    /// dissipated at the physical boundary rather than reflected. `None`
    /// disables CPML and reproduces the periodic / characteristic-only path
    /// exactly.
    pub cpml: Option<super::cpml::DgCpmlConfig>,
}

impl Default for DGConfig {
    fn default() -> Self {
        Self {
            polynomial_order: 2,
            basis_type: super::basis::BasisType::Legendre,
            flux_type: super::flux::FluxType::LaxFriedrichs,
            time_integrator: DgTimeIntegrator::SspRk3,
            use_limiter: false,
            limiter_type: super::flux::LimiterType::Minmod,
            shock_threshold: 0.1,
            shock_capture: ShockCaptureConfig::default(),
            sound_speed: 1500.0,
            boundary_conditions: [DgBoundaryCondition::Periodic; 3],
            cpml: None,
        }
    }
}
