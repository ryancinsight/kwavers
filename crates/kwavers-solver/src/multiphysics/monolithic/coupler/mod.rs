//! Monolithic coupler state and public facade.
//!
//! The coupler owns Newton-Krylov configuration, reusable solver workspaces,
//! registered physics components, and convergence state.  Implementation
//! details are partitioned by lifecycle responsibility:
//!
//! - `construction`: constructors and initial workspace state;
//! - `validation`: pre-solve simulation contract checks;
//! - `solve`: coupled Newton-Krylov step execution;
//! - `plugins`: mutable physics component/coefficient configuration;
//! - `accessors`: read-only public state access.
//!
//! All child modules add inherent methods to [`MonolithicCoupler`].  No wrapper
//! type, compatibility alias, or dynamic dispatch layer is introduced.

use kwavers_domain::plugin::Plugin;
use crate::integration::nonlinear::{GMRESConfig, GMRESSolver};
use ndarray::Array3;
use std::collections::HashMap;

use super::config::{NewtonKrylovConfig, PhysicsCoefficients};

mod accessors;
mod construction;
mod plugins;
mod solve;
mod validation;

/// Monolithic multiphysics coupler.
///
/// Solves coupled multiphysics systems simultaneously without subcycling or
/// iteration lag. The nonlinear system is advanced by Jacobian-Free
/// Newton-Krylov with reusable full-state workspaces for the dominant
/// allocation sites.
#[derive(Debug)]
pub struct MonolithicCoupler {
    /// Newton-Krylov configuration.
    pub(super) newton_config: NewtonKrylovConfig,

    /// GMRES linear solver configuration.
    pub(super) gmres_config: GMRESConfig,

    /// Newton residual history for the most recent coupled step.
    pub(super) convergence_history: Vec<f64>,

    /// Physics components registered for future plugin-backed coupling.
    pub(super) physics_components: HashMap<String, Box<dyn Plugin>>,

    /// Physical coefficients for the coupled PDE residual system.
    pub(super) physics_coefficients: PhysicsCoefficients,

    /// Pre-allocated correction vector δu for Newton iterations.
    ///
    /// Lazily initialized on the first coupled step once grid dimensions are
    /// known. Avoids one full-state `Array3::zeros` allocation per Newton
    /// iteration.
    pub(super) du_scratch: Option<Array3<f64>>,

    /// Pre-allocated previous-state snapshot for Newton residual evaluation.
    ///
    /// `F(u) = u - u_prev - dt * R(u)` requires a stable copy of the incoming
    /// flattened state for the duration of a coupled step. This workspace keeps
    /// that snapshot allocation solver-owned across calls.
    pub(super) u_prev_scratch: Option<Array3<f64>>,

    /// Pre-allocated Newton linear-system right-hand side `-F(u)`.
    pub(super) rhs_scratch: Option<Array3<f64>>,

    /// Pre-allocated trial state for Newton line search candidates.
    pub(super) line_search_state_scratch: Option<Array3<f64>>,

    /// Pre-allocated perturbed state for Jacobian-vector products.
    pub(super) jvp_state_scratch: Option<Array3<f64>>,

    /// Reusable GMRES solver instance.
    pub(super) gmres_solver: Option<GMRESSolver>,

    /// Grid cell spacings `(dx, dy, dz)` in metres from the active solve grid.
    pub(super) grid_spacing: (f64, f64, f64),
}

#[cfg(test)]
mod tests;
