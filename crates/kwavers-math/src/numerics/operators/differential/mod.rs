//! # Differential Operators
//!
//! SSOT shim after ADR 0018: the 3-D central-difference kernels
//! (`CentralSecondOrder`, `CentralFourthOrder`, `CentralSixthOrder`)
//! live in `let_ops::FiniteDifference3D` and
//! `let_ops::FiniteDifference3DScheme`. The Yee staggered operator
//! (`StaggeredGridOperator`) remains kwavers-side pending the
//! staggered SSOT sweep tracked in ADR 0018's "Follow-up (staggered
//! half SSOT)" section.
//!
//! ## Architecture
//!
//! - `FiniteDifference3D` / `FiniteDifference3DScheme` — provider-SSOT,
//!   re-exported from `leto_ops`.
//! - `StaggeredGridOperator` — Yee face forward/backward
//!   difference kernel. Kept here pending the staggered half SSOT
//!   sweep; the central half has been retired.
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
//! Copy>`. The central half is owned by leto-ops. The staggered
//! half remains pending the follow-up SSOT sweep.

use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3};

// Implementation modules — only the staggered half + its compat helpers remain.
mod staggered_grid;
mod traversal;

// ── SSOT re-exports (central half migrated to leto-ops in ADR 0018) ───────────

/// Provider-SSOT for 3-D first-derivative kernels: central 2nd/4th/6th +
/// Yee staggered forward/backward. Owned by `let_ops`; re-exported here so
/// that `kwavers_math::numerics::operators::FiniteDifference3D` continues
/// to resolve for downstream solver/bench/test paths.
pub use leto_ops::{FiniteDifference3D, FiniteDifference3DScheme};

// ── Staggered half (kwavers-side pending the SSOT sweep) ─────────────────────

pub use staggered_grid::StaggeredGridOperator;
