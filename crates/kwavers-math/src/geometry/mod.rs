//! Geometry Helper Functions
//!
//! Geometry creation functions for transducer masks,
//! region-of-interest definitions, and spatial configurations.
//!
//! | Toolbox Function | Kwavers Equivalent | Description |
//! |-----------------|-------------------|-------------|
//! | `makeDisc` | [`make_disc`] | 2D circular mask |
//! | `makeBall` | [`make_ball`] | 3D spherical mask |
//! | `makeSphere` | [`make_sphere`] | Alias for [`make_ball`] |
//! | `makeLine` | [`make_line`] | Linear mask connecting two points |
//!
//! # References
//!
//! - Treeby, B. E., & Cox, B. T. (2010). "MATLAB toolbox for the simulation
//!   and reconstruction of photoacoustic wave fields." J. Biomed. Opt. 15(2), 021314.

pub mod delays;

mod circle;
mod line;
mod primitives;
#[cfg(test)]
mod tests;
mod utils;

pub use circle::make_circle;
pub use line::make_line;
pub use primitives::{make_ball, make_disc, make_sphere};
pub use utils::{distance3, normalize3, orthogonal_basis_from_normal3};
