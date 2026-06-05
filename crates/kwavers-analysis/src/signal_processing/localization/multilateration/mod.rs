//! Multilateration Source Localization with Advanced TDOA Methods
//!
//! Implements multilateration algorithms for acoustic source localization
//! using time-difference-of-arrival (TDOA) measurements from sensor arrays.
//! Handles overdetermined systems with more sensors than required, improving
//! accuracy through statistical optimization.
//!
//! # References
//!
//! - Foy, W. H. (1976). *IEEE Trans. Aerosp. Electron. Syst.* AES-12(2), 187-194.
//! - Smith, J. O., & Abel, J. S. (1987). *IEEE Trans. ASSP* 35(12), 1661-1669.
//! - Chan, Y. T., & Ho, K. C. (1994). *IEEE Trans. Signal Process.* 42(8), 1905-1915.

mod solver;
#[cfg(test)]
mod tests;
mod types;

pub use solver::Multilateration;
pub use types::MultilaterationConfig;
