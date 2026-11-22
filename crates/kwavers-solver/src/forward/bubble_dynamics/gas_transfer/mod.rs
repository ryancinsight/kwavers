//! Conservative transient diffusion for a prescribed or dynamically coupled spherical interface.
//!
//! The state is the liquid concentration field in the material coordinate,
//! retained across each pulse and its off interval. The far boundary uses the
//! bulk concentration. Set its distance and the radial and time resolution
//! from convergence checks for the exposure of interest. This solver returns
//! gas exchange for coupling to a bubble mass balance.

mod coupled;
mod error;
mod grid;
mod heat;
mod models;
mod step;

pub use coupled::{DiffusiveBubble, DiffusiveBubbleError, DiffusiveBubbleState};
pub use error::GasTransferError;
pub use models::{BubbleForcing, DiffusiveBubbleModels, GasHeatTransfer, RadialMechanics};
pub use step::{GasTransfer, TransientGasDiffusion};
