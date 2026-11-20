//! Value-semantic tests for `GenericFdtdSolver` (CFL stability, leapfrog energy).
//!
//! ## Theorems verified
//!
//! ### CFL stability condition (Courant, Friedrichs & Lewy 1928; Yee 1966)
//!
//! For the 3D staggered-grid Yee leapfrog scheme with uniform spacing Δx and
//! p-th-order central differences, the von Neumann stability criterion requires:
//!
//! ```text
//! c₀ · Δt · √K ≤ 1        (K = number of spatial dimensions = 3)
//! ```
//!
//! Equivalently:
//!
//! ```text
//! Δt_max = Δx / (c₀ · √K)     [2nd-order stencil, isotropic]
//! ```
//!
//! The `max_stable_dt` accessor implements:
//!
//! ```text
//! dt_max = cfl_factor × cfl_limit(order) × Δx_min / c_max
//! ```
//!
//! where `cfl_limit(order)` = 1/√3 (2nd), 1/√15 (4th), 1/√27 (6th) for
//! the 3D von Neumann stability limits of each stencil order
//! (Gustafsson et al. 1995).
//!
//! ### Leapfrog discrete energy conservation (Taflove & Hagness 2005, §3.4)
//!
//! In a lossless (α=0) homogeneous medium with no external source, the discrete
//! total acoustic energy:
//!
//! ```text
//! E_n = Σ_i [ p_i² / (2ρ₀c₀²) ] · ΔV
//! ```
//!
//! is conserved over the leapfrog update to within floating-point round-off
//! when the CFL condition is satisfied. Specifically, the field must remain
//! bounded (not exponentially growing) for subcritical Δt.
//!
//! ## References
//!
//! - Courant R, Friedrichs K, Lewy H (1928). Math. Ann. 100, 32–74.
//! - Yee KS (1966). IEEE Trans. Antennas Propag. 14(3), 302–307.
//! - Gustafsson B et al. (1995). Time Compact Difference Schemes. §4.
//! - Taflove A, Hagness SC (2005). Computational Electrodynamics, 3rd ed. §3.4.

use crate::forward::fdtd::config::FdtdConfig;
use crate::forward::fdtd::solver::FdtdSolver;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::GridSource;

mod absorption;
mod energy;
mod phase_split;
mod stability;
mod stepping;
mod time_integration;
mod validation;

/// Helper: create a minimal FdtdSolver for unit tests.
///
/// `dt` comes from the solver's own `max_stable_dt`, which reads the Courant
/// limit off the staggered stencil's coefficients. Deriving it here from
/// `1/√3` instead — as this helper did — is the collocated second-order limit,
/// and the staggered limit falls with the order: `1/(√3·Σ|cₙ|)` is 0.4949 at
/// order 4 against 0.5774 at order 2. A fourth-order run at `cfl_factor` 0.95
/// therefore took a step 11% past its limit, and `fdtd_step_phase_split` timed
/// a field that had grown to 1e250 by the end of its loops while the
/// `is_finite` guard stayed quiet.
fn make_solver(
    n: usize,
    dx: f64,
    c0: f64,
    rho0: f64,
    cfl_factor: f64,
    spatial_order: usize,
) -> FdtdSolver {
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    let config = FdtdConfig {
        spatial_order,
        staggered_grid: true,
        cfl_factor,
        enable_nonlinear: false,
        // Replaced below by the limit this solver reports for its own stencil;
        // construction needs some positive step to validate.
        dt: cfl_factor / (3.0_f64).sqrt() * dx / c0,
        nt: 10,
        ..Default::default()
    };
    let provisional =
        FdtdSolver::new(config.clone(), &grid, &medium, GridSource::new_empty()).unwrap();
    let config = FdtdConfig {
        dt: provisional.max_stable_dt(c0),
        ..config
    };
    FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap()
}
