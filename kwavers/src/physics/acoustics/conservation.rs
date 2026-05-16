//! Conservation law validation and entropy production for acoustic simulations.
//!
//! # Conservation Laws in Linear Acoustics
//!
//! The linearised equations of compressible fluid mechanics conserve energy,
//! mass, and momentum. Verifying these laws is a necessary diagnostic for
//! acoustic solver correctness.
//!
//! ## Acoustic energy theorem
//!
//! For pressure `p`, velocity `v`, density `rho0`, and sound speed `c0`,
//!
//! ```text
//! e = 1/2 rho0 |v|^2 + p^2 / (2 rho0 c0^2).
//! ```
//!
//! Proof: multiplying the linearised momentum equation by `v` and the
//! compressibility equation by `p/(rho0 c0^2)`, then summing, yields
//! `partial_t e + div(p v)=0` in a lossless medium.
//!
//! ## Entropy production theorem
//!
//! For absorption `alpha >= 0`, the irreversible entropy production rate is
//!
//! ```text
//! dS_irr/dt = integral_V 2 alpha c0 e / T0 dV >= 0.
//! ```
//!
//! Proof: progressive-wave absorbed power density is `2 alpha c0 e`; dividing
//! irreversible heat production by positive absolute temperature gives the
//! local entropy production. Nonnegative absorption and energy density imply
//! nonnegative total entropy production.
//!
//! References: Morse & Ingard (1968), Hamilton & Blackstock (1998),
//! Blackstock (2000), Landau & Lifshitz (1987), Pennes (1948), Sapareto &
//! Dewey (1984).

mod energy;
mod entropy;
mod heat;
mod intensity;
mod mass;
mod metrics;
mod momentum;
mod state_refs;
mod validation;

#[cfg(test)]
mod tests;

pub use energy::validate_energy_conservation;
pub use entropy::entropy_production_rate;
pub use heat::acoustic_heat_source;
pub use intensity::{acoustic_intensity, acoustic_power_through_z_plane};
pub use mass::validate_mass_conservation;
pub use metrics::ConservationMetrics;
pub use momentum::validate_momentum_conservation;
pub use state_refs::{AcousticStateRefs, ConservationParams, PreviousFields, VelocityFieldRefs};
pub use validation::validate_conservation;
