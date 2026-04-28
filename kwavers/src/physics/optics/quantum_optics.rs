//! Quantum optics models for radiative transition rates and plasma corrections.
//!
//! # Scope
//!
//! This module provides quantum optical emission models needed for complete
//! photon production analysis in extreme light-matter environments such as
//! single-bubble sonoluminescence (SBSL):
//!
//! 1. Einstein A/B coefficients for two-level atomic systems.
//! 2. Quantum correction assessment against classical bremsstrahlung.
//! 3. Relativistic plasma parameter and hydrogenic Lamb-shift scaling.
//! 4. Frequency- and temperature-dependent free-free Gaunt factor.
//!
//! # Classical sufficiency theorem for SBSL bremsstrahlung
//!
//! For SBSL temperatures `T = 10_000-30_000 K`,
//! `k_B T / (m_e c^2) = 1.7e-6..5.1e-6`. First-order relativistic
//! bremsstrahlung corrections are `O(k_B T / (m_e c^2))`, so classical
//! free-free emission is accurate to better than 0.001% in this regime.
//!
//! The hydrogen 2s Lamb shift is `4.374e-6 eV`, while `k_B T = 0.86 eV` at
//! 10,000 K. Thus `Delta E_Lamb / k_B T = 5.1e-6`, below thermal Doppler
//! broadening and unobservable in thermal SBSL continua.
//!
//! # References
//!
//! 1. Einstein, A. (1917). Zur Quantentheorie der Strahlung. *Phys. Z.* 18, 121-128.
//! 2. Lamb, W. E., & Retherford, R. C. (1947). *Phys. Rev.* 72(3), 241-243.
//! 3. Berestetskii, V. B., Lifshitz, E. M., & Pitaevskii, L. P. (1982).
//!    *Quantum Electrodynamics*. 2nd ed. Pergamon.
//! 4. Sobelman, I. I. (1992). *Atomic Spectra and Radiative Transitions*. 2nd ed.
//! 5. Sutherland, R. A. (1998). *J. Quant. Spectrosc. Radiat. Transfer* 60(6), 1010-1030.
//! 6. Rybicki, G. B., & Lightman, A. P. (1979). *Radiative Processes in Astrophysics*.
//! 7. Brenner, M. P., Hilgenfeldt, S., & Lohse, D. (2002). *Rev. Mod. Phys.* 74(2), 425-484.

mod constants;
mod corrections;
mod einstein;
mod gaunt;
mod special;

#[cfg(test)]
mod tests;

pub use corrections::{lamb_shift_ev, relativistic_parameter, QuantumCorrectionAssessment};
pub use einstein::EinsteinCoefficients;
pub use gaunt::gaunt_factor_ff;
