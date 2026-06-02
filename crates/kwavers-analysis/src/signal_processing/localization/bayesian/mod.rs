//! Bayesian Filtering for Source Localization — Extended Kalman Filter
//!
//! ## References
//! * Kalman RE (1960). "A new approach to linear filtering and prediction problems."
//!   *Trans. ASME J. Basic Eng.* 82:35–45.
//! * Bar-Shalom Y, Li XR, Kirubarajan T (2001). *Estimation with Applications to
//!   Tracking and Navigation*. Wiley. §5.2, §6.2.
//! * Bierman GJ (1977). *Factorization Methods for Discrete Sequential Estimation*.
//!   Academic Press. Ch. 6.
//! * Singer RA (1970). "Estimating optimal tracking filter performance for manned
//!   maneuvering targets." *IEEE Trans. Aerosp. Electron. Syst.* 6(4):473–483.

mod config;
mod filter;
mod linalg;
#[cfg(test)]
mod tests;

pub use config::{KalmanFilterConfig, KalmanFilterType};
pub use filter::BayesianFilter;
