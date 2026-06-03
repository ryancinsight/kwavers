//! Gas-diffusion dissolution of bubbles (Epstein–Plesset and extensions).
//!
//! Models the loss (or gain) of gas by a bubble through diffusion across its
//! interface — the process that sets how long residual cavitation bubbles
//! persist between pulses (and hence the inter-pulse shielding time τ_d).
//!
//! * [`EpsteinPlessetDissolution`] — the complete Epstein–Plesset (1950) free
//!   bubble model (surface-tension Laplace drive + transient diffusion term).
//! * [`ShellPermeationDissolution`] — encapsulated microbubbles with a finite
//!   shell gas permeability (Sarkar 2009): the same diffusion in series with the
//!   shell resistance, which stabilises contrast agents.
//! * Both implement the sealed [`DissolutionModel`] trait, so new variants
//!   (multi-gas osmotic stabilisation, rectified-diffusion growth, …) plug in
//!   without changing callers.
//! * [`integrate_dissolution`] / [`dissolution_time_numeric`] integrate `R(t)`
//!   and report the dissolution time.

mod epstein_plesset;
mod integrator;
mod shelled;
mod traits;

#[cfg(test)]
mod tests;

pub use epstein_plesset::EpsteinPlessetDissolution;
pub use integrator::{dissolution_time_numeric, integrate_dissolution, DissolutionTrajectory};
pub use shelled::ShellPermeationDissolution;
pub use traits::{DissolutionModel, GasDiffusionParams};
