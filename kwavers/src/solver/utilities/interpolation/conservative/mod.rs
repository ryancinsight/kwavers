//! Conservative Interpolation for Energy-Preserving Multi-Grid Coupling
//!
//! Sparse volume-overlap transfer operators that satisfy:
//! ```text
//! ∫_T u_target dV = ∫_S u_source dV   (conservation)
//! Σᵢ T_{ij} = 1                        (partition of unity)
//! ```
//!
//! | Submodule      | Contents                                              |
//! |----------------|-------------------------------------------------------|
//! | `mode`         | `ConservationMode` — mass / energy / momentum / all  |
//! | `interpolator` | `UtilConservativeInterpolator` — build, transfer, verify  |
//!
//! # References
//! - Grandy, J. (1999). "Conservative Remapping." *J. Comput. Phys.*, 148(2), 433–466.

mod interpolator;
mod mode;
#[cfg(test)]
mod tests;

pub use interpolator::UtilConservativeInterpolator;
pub use mode::ConservationMode;
