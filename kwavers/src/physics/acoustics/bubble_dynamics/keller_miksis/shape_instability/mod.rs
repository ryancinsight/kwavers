//! Bubble Shape Instability - Plesset-Prosperetti surface mode analysis.
//!
//! A spherically oscillating bubble is subject to surface instabilities when
//! the bubble-wall acceleration is large. The surface perturbation is expanded
//! as `r(theta, phi, t) = R(t) + sum_n a_n(t) Y_n^0(theta)`, where `R(t)`
//! follows the Keller-Miksis equation and each `a_n(t)` follows the
//! Plesset-Prosperetti linear mode equation:
//!
//! ```text
//! a_ddot_n + D_n(t) a_dot_n - G_n(t) a_n = 0
//!
//! D_n = 3 R_dot/R + 4 nu (n + 2)(2n + 1)/R^2
//! G_n = (n - 1)[R_ddot/R - (n + 2)(R_dot/R)^2]
//!       - n(n - 1)(n + 2) sigma/(rho_L R^3)
//! ```
//!
//! Breakup is flagged when any tracked mode amplitude exceeds 30% of the
//! current radius. Near a rigid boundary, jet formation uses the Blake-Taib-
//! Doherty stand-off scaling.
//!
//! References: Plesset (1954), Prosperetti (1977), Brennen (1995), Blake et al.
//! (1986), Lauterborn (1974).

mod constants;
mod dynamics;
mod jet;
mod state;

pub use constants::{BREAKUP_FRACTION, JET_STANDOFF_CRITICAL, N_MODES};
pub use dynamics::advance_shape_modes;
pub use jet::jet_speed;
pub use state::ShapeModeState;

#[cfg(test)]
mod tests;
