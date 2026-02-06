//! Encapsulated Bubble Dynamics with Shell Mechanics
//!
//! This module implements models for ultrasound contrast agent (UCA) microbubbles
//! with viscoelastic shells. These bubbles consist of a gas core surrounded by
//! a thin shell (lipid, protein, or polymer) that significantly affects the
//! bubble dynamics.
//!
//! # Models Implemented
//!
//! 1. **Church Model (1995)** - Linear viscoelastic shell
//! 2. **Marmottant Model (2005)** - Buckling and rupture behavior for lipid shells
//!
//! # References
//!
//! - Church, C. C. (1995). "The effects of an elastic solid surface layer on the
//!   radial pulsations of gas bubbles." J. Acoust. Soc. Am. 97(3), 1510-1521.
//! - Marmottant, P., et al. (2005). "A model for large amplitude oscillations of
//!   coated bubbles accounting for buckling and rupture." J. Acoust. Soc. Am.
//!   118(6), 3499-3505.
//! - Stride, E. & Coussios, C. C. (2010). "Nucleation, mapping and control of
//!   cavitation for drug delivery." Nat. Rev. Drug Discov. 9, 527-536.

pub mod model;
pub mod shell;

pub use model::{ChurchModel, MarmottantModel};
pub use shell::ShellProperties;
