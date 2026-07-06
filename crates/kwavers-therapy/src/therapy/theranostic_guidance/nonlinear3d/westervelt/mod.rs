//! 3-D Westervelt propagation and discrete-adjoint FWI.
//!
//! The forward recurrence is the same finite-difference Westervelt update
//! documented in `solver::forward::nonlinear::westervelt`, specialized here
//! to CT-derived arrays so the inverse path can expose every pressure history
//! to a reverse-mode discrete adjoint.
//!
//! # Theorem
//!
//! For recurrence
//! `p[n+1] = S * (2p[n] - p[n-1] + N[n] / D[n] + s[n])`, where
//! `N[n] = c^2 dt^2 Lp[n] + 2 beta (p[n] - p[n-1])^2 / (rho c^2)` and
//! `D[n] = 1 - 2 beta p[n] / (rho c^2)`, the reverse sweep in `gradient`
//! computes the exact derivative of the discrete trace least-squares objective
//! with respect to the nodal sound speed and nonlinearity coefficient.
//!
//! # Proof sketch
//!
//! Each loop applies the transpose of the Jacobian of the scalar recurrence
//! to the adjoint variables and sums the local `∂p[n+1]/∂c` term;
//! reverse-mode accumulation over an acyclic time-unrolled graph is the chain
//! rule.
//!
//! # Performance contract
//!
//! - The forward pressure history is stored as exact sparse checkpoints.
//!   Each checkpoint contains `p[n-2]`, `p[n-1]`, and `p[n]`; the reverse
//!   sweep replays one bounded interval at a time with the same recurrence.
//!   This preserves dense-history gradients while reducing retained forward
//!   state from `O(steps * cells)` to `O((steps / interval + interval) *
//!   cells)`.
//! - The adjoint variables use four rolling `Vec<f64>` states for
//!   `lambda[n+1]`, `lambda[n]`, `lambda[n-1]`, and `lambda[n-2]`. This is
//!   algebraically equivalent to storing `(steps + 1)` adjoint states because
//!   the Westervelt recurrence has temporal stencil width three; the reverse
//!   sweep never reads an adjoint state after it shifts past this window.
//! - The forward cell update is Moirai-parallel over x-slabs: each worker
//!   writes only to its owned `next` slab without coloring, atomics, or locks.
//! - The four rotating buffers (older, previous, current, next) are
//!   `mem::swap`-rotated each step. No `vec![0.0; cells]` allocation occurs
//!   inside the time loop.
//! - The FWI loop owns reusable candidate and residual workspaces. Backtracking
//!   line search overwrites candidate `c`/`beta` buffers in place, and each
//!   source-encoded shot overwrites one residual trace buffer before adjoint
//!   evaluation; no candidate or residual vector allocation occurs inside the
//!   iteration loop.

mod calibration;
mod fwi;
pub(super) mod types;

// Re-export sibling-module items so sub-modules (fwi, calibration, tests)
// can access them via `super::` without needing pub(super) escalation across
// two levels (grandchild cannot directly reach pub(super) items from siblings
// of its grandparent; the parent-level re-export bridges the gap).
pub(super) use super::adjoint::{gradient, GradientInput, ParameterGradient};
pub(super) use super::encoding::{EncodedTrace, SourceEncoding};
pub(super) use super::forward::{
    forward_with_schedule, source_plan_metrics, time_schedule, ForwardInput, TimeSchedule,
};
pub(super) use super::metrics::metrics_from_score;
pub(super) use super::optimization::{
    apply_line_search, objective_for_model, LineSearchInput, LineSearchWorkspace, ObjectiveInput,
};
pub(super) use super::regularization::{
    add_h1_gradient, h1_penalty, multiparameter_score, smooth_gradient,
};
pub(super) use super::stencil::index;
pub(super) use super::types::{flat_index, Nonlinear3dAperture, Nonlinear3dConfig};

pub(super) use fwi::run_fwi;

#[cfg(test)]
mod tests;
