//! Numerical methods validation tests
//!
//! References:
//! - Treeby & Cox (2010) - "MATLAB toolbox"
//! - Gear & Wells (1984) - "Multirate linear multistep methods"
//! - Berger & Oliger (1984) - "Adaptive mesh refinement"
//! - Persson & Peraire (2006) - "Sub-cell shock capturing"

#[cfg(test)]
mod amr;
#[cfg(test)]
pub(crate) mod helpers;
#[cfg(test)]
mod mms;
#[cfg(test)]
mod pstd;
#[cfg(test)]
mod shock;
#[cfg(test)]
mod time_integration;
