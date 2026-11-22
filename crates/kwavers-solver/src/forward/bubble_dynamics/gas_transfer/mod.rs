//! Conservative transient diffusion for a prescribed or Rayleigh–Plesset spherical interface.
//!
//! The state is the liquid concentration field in the material coordinate,
//! retained across each pulse and its off interval. The far boundary uses the
//! bulk concentration. Set its distance and the radial and time resolution
//! from convergence checks for the exposure of interest. This solver returns
//! gas exchange for coupling to a bubble mass balance.

mod error;
mod grid;
mod rayleigh_plesset;
mod step;

pub use error::GasTransferError;
pub use rayleigh_plesset::{DiffusiveBubble, DiffusiveBubbleError, DiffusiveBubbleState};
pub use step::{GasTransfer, TransientGasDiffusion};
