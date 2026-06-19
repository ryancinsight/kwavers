//! Encapsulated Bubble Dynamics with Shell Mechanics
//!
//! This module implements models for ultrasound contrast agent (UCA) microbubbles
//! with viscoelastic shells. These bubbles consist of a gas core surrounded by
//! a thin shell (lipid, protein, or polymer) that significantly affects the
//! bubble dynamics.
//!
//! # Models Implemented
//!
//! All share the [`EncapsulatedShellModel`] modified-Rayleigh-Plesset driver.
//!
//! 1. **Church Model (1995)** - thin-shell shear-elastic viscoelastic shell
//! 2. **Hoff Model (2000)** - thin-shell, linear-displacement elastic restoring
//! 3. **Marmottant Model (2005)** - buckling and rupture for lipid shells
//! 4. **Sarkar Model (2005)** - interfacial elasticity + surface viscosity
//!
//! # References
//!
//! - Church, C. C. (1995). "The effects of an elastic solid surface layer on the
//!   radial pulsations of gas bubbles." J. Acoust. Soc. Am. 97(3), 1510-1521.
//! - Hoff, L., Sontum, P. C., & Hovem, J. M. (2000). "Oscillations of polymeric
//!   microbubbles: Effect of the encapsulating shell." J. Acoust. Soc. Am.
//!   107(4), 2272-2280.
//! - Marmottant, P., et al. (2005). "A model for large amplitude oscillations of
//!   coated bubbles accounting for buckling and rupture." J. Acoust. Soc. Am.
//!   118(6), 3499-3505.
//! - Sarkar, K., Shi, W. T., Chatterjee, D., & Forsberg, F. (2005). "Characterization
//!   of ultrasound contrast microbubbles using in vitro experiments and viscous and
//!   viscoelastic interface models for encapsulation." J. Acoust. Soc. Am. 118(1), 539-550.
//! - Doinikov, A. A., & Bouakaz, A. (2011). "Review of shell models for contrast
//!   agent microbubbles." IEEE Trans. UFFC 58(5), 981-993.
//! - Stride, E. & Coussios, C. C. (2010). "Nucleation, mapping and control of
//!   cavitation for drug delivery." Nat. Rev. Drug Discov. 9, 527-536.

pub mod model;
pub mod shell;

pub use model::{ChurchModel, EncapsulatedShellModel, HoffModel, MarmottantModel, SarkarModel};
pub use shell::ShellProperties;
