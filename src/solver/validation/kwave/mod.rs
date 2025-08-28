//! k-Wave validation and comparison module
//!
//! This module provides comprehensive validation against the k-Wave toolbox,
//! a widely-used acoustic simulation package for MATLAB and C++.
//!
//! ## References
//!
//! 1. **Treeby, B. E., & Cox, B. T. (2010)**. "k-Wave: MATLAB toolbox for the
//!    simulation and reconstruction of photoacoustic wave fields." *Journal of
//!    Biomedical Optics*, 15(2), 021314. DOI: 10.1117/1.3360308
//!
//! 2. **Treeby, B. E., Jaros, J., Rendell, A. P., & Cox, B. T. (2012)**.
//!    "Modeling nonlinear ultrasound propagation in heterogeneous media with
//!    power law absorption using a k-space pseudospectral method." *The Journal
//!    of the Acoustical Society of America*, 131(6), 4324-4336. DOI: 10.1121/1.4712021

pub mod analytical;
pub mod comparison;
pub mod report;
pub mod test_cases;
pub mod validator;

pub use analytical::AnalyticalSolutions;
pub use comparison::ComparisonMetrics;
pub use report::{TestResult, ValidationReport};
pub use test_cases::{KWaveTestCase, ReferenceSource};
pub use validator::KWaveValidator;
