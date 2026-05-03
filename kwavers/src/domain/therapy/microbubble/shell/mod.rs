//! Marmottant Shell Model for Coated Microbubbles
//!
//! Implements the Marmottant et al. (2005) model for large amplitude oscillations
//! of lipid-coated microbubbles. The shell transitions between Buckled, Elastic,
//! and Ruptured states.
//!
//! ## References
//!
//! - Marmottant et al. (2005): JASA 118(6):3499–3505

mod properties;
mod state;
#[cfg(test)]
mod tests;

pub use properties::MarmottantShellProperties;
pub use state::ShellState;
