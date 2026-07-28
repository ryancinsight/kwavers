//! # Differential Operators
//!
//! SSOT shim after ADR 0018 and ADR 0033: the 3-D first-derivative kernels
//! (central 2nd/4th/6th and Yee staggered forward/backward) live in
//! `leto_ops::FiniteDifference3D` and `leto_ops::FiniteDifference3DScheme`.
//! This module re-exports them so that downstream `kwavers` crates can keep
//! importing through `kwavers_math::numerics::operators`.
//!
//! ## Architecture
//!
//! - `FiniteDifference3D` / `FiniteDifference3DScheme` — provider-SSOT,
//!   re-exported from `leto_ops`.
//!
//! ## References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on
//!   arbitrarily spaced grids." *Mathematics of Computation*, 51(184),
//!   699-706.
//! - Shubin & Bell (1987). Modified-equation fourth-order methods for
//!   acoustic wave propagation.
//! - Yee, K. (1966). Numerical solution of initial boundary value
//!   problems involving Maxwell's equations in isotropic media. *IEEE
//!   Trans. Antennas Propag.*, 14(3), 302-307.
//!
//! ## SSR / SSOT contract
//!
//! The provider is `FiniteDifference3D<T: RealField + FloatElement +
//! Copy>` in `leto-ops`. All differential stencils are owned by leto-ops.

// ── SSOT re-exports ─────────────────────────────────────────────────────────

/// Provider-SSOT for 3-D first-derivative kernels: central 2nd/4th/6th +
/// Yee staggered forward/backward. Owned by `leto_ops`; re-exported here so
/// that `kwavers_math::numerics::operators::FiniteDifference3D` continues
/// to resolve for downstream solver/bench/test paths.
pub use leto_ops::{FiniteDifference3D, FiniteDifference3DScheme};
