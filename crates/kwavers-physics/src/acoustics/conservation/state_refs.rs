//! Borrow-grouped parameter types for conservation validation kernels.
//!
//! Groups the frequently co-passed acoustic field arrays and scalar check
//! parameters into `Copy`-or-borrow aggregates so that the conservation
//! functions remain below Clippy's argument-count threshold while preserving
//! field-name explicitness at every call site.

use leto::Array3;

/// Instantaneous acoustic field state (all shared borrows).
///
/// Bundles `pressure`, the three velocity components, `density`,
/// `sound_speed`, and `absorption` — the arrays co-passed to energy, entropy,
/// mass, and momentum conservation checks.
#[derive(Debug)]
pub struct AcousticStateRefs<'a> {
    /// Acoustic pressure field `p` (Pa).
    pub pressure: &'a Array3<f64>,
    /// Velocity component along x (m/s).
    pub velocity_x: &'a Array3<f64>,
    /// Velocity component along y (m/s).
    pub velocity_y: &'a Array3<f64>,
    /// Velocity component along z (m/s).
    pub velocity_z: &'a Array3<f64>,
    /// Ambient density `ρ₀` (kg/m³).
    pub density: &'a Array3<f64>,
    /// Sound speed `c₀` (m/s).
    pub sound_speed: &'a Array3<f64>,
    /// Power-law absorption coefficient `α` (Np/m).  Used by entropy and heat.
    pub absorption: &'a Array3<f64>,
}

/// Three-component velocity field (shared borrows).
///
/// Used for the previous-timestep velocity triple in momentum checks and
/// wherever only velocity components are needed.
#[derive(Debug)]
pub struct VelocityFieldRefs<'a> {
    /// x-component (m/s).
    pub x: &'a Array3<f64>,
    /// y-component (m/s).
    pub y: &'a Array3<f64>,
    /// z-component (m/s).
    pub z: &'a Array3<f64>,
}

/// Optional previous-timestep fields for conservation checks that require
/// finite-difference time derivatives.
#[derive(Debug)]
pub struct PreviousFields<'a> {
    /// Previous pressure (unused by current checks, reserved for future use).
    pub pressure: Option<&'a Array3<f64>>,
    /// Previous velocity triple — required for momentum residual.
    pub velocity: Option<VelocityFieldRefs<'a>>,
    /// Previous density — required for mass continuity residual.
    pub density: Option<&'a Array3<f64>>,
}

/// Scalar parameters controlling a [`validate_conservation`](super::validate_conservation) call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConservationParams {
    /// Total acoustic energy at `t = 0` used as the reference for the relative
    /// energy error (J).
    pub initial_energy: f64,
    /// Simulation time step `Δt` (s).
    pub dt: f64,
    /// Ambient absolute temperature `T₀` for entropy production (K).
    pub temperature: f64,
    /// Acceptance threshold below which numerical residuals are considered
    /// conservation-satisfying (dimensionless relative error).
    pub tolerance: f64,
}
