//! k-Wave Comparison and Validation Module
//!
//! This module provides infrastructure for comparing kwavers acoustic solvers
//! against k-Wave (MATLAB toolbox), the industry standard for ultrasound simulation.
//!
//! # Overview
//!
//! k-Wave is a widely-used MATLAB toolbox for acoustic and elastic wave propagation
//! developed by Bradley Treeby and Ben Cox at University College London. It uses
//! k-space pseudospectral time domain (k-space PSTD) methods with exact spatial
//! derivatives computed in Fourier space.
//!
//! This module enables:
//! 1. **Analytical Validation**: Compare against exact solutions (plane waves, Gaussian beams, etc.)
//! 2. **Numerical Comparison**: Compare kwavers vs k-Wave on identical test cases
//! 3. **Gap Analysis**: Identify missing features and correctness issues
//! 4. **Performance Benchmarking**: Runtime and accuracy comparisons
//!
//! # Mathematical Foundation
//!
//! ## Linear Acoustic Wave Equation
//!
//! Both k-Wave and kwavers solve the first-order acoustic equations:
//!
//! ```text
//! ∂p/∂t + ρ₀c₀²∇·u = 0           (1) Pressure evolution
//! ∂u/∂t + (1/ρ₀)∇p = 0            (2) Velocity evolution
//! ```
//!
//! Where:
//! - `p(x,t)` = acoustic pressure [Pa]
//! - `u(x,t)` = particle velocity [m/s]
//! - `ρ₀` = ambient density [kg/m³]
//! - `c₀` = sound speed [m/s]
//!
//! ## k-Space Operator (k-Wave's Innovation)
//!
//! k-Wave computes spatial derivatives exactly in Fourier space:
//!
//! ```text
//! ∇f → F^{-1}[ik·F[f]]             (3) Exact spatial derivative
//! ```
//!
//! This eliminates numerical dispersion, unlike FDTD methods which have
//! discretization-dependent dispersion errors.
//!
//! ## Validation Strategy
//!
//! 1. **Analytical Solutions** (Ground Truth):
//!    - Plane waves: p(x,t) = A sin(kx - ωt)
//!    - Gaussian beams: Paraxial beam propagation
//!    - Spherical waves: Point source radiation
//!
//! 2. **Comparison Metrics**:
//!    - L² error: ε₂ = √(∫(p_num - p_ana)²dV / ∫p_ana²dV)
//!    - L∞ error: ε∞ = max|p_num - p_ana| / max|p_ana|
//!    - Phase error: Δφ = acos(correlation)
//!
//! 3. **Acceptance Criteria** (k-Wave Standard):
//!    - L² error < 0.01 (1%)
//!    - L∞ error < 0.05 (5%)
//!    - Phase error < 0.1 rad (5.7°)
//!
//! # Usage Example
//!
//! ```rust,no_run
//! use kwavers::solver::validation::kwave_comparison::analytical::{PlaneWave, ErrorMetrics};
//! use kwavers::domain::grid::Grid;
//!
//! // Create analytical solution
//! let wave = PlaneWave::new(1e5, 1e6, 1500.0, [1.0, 0.0, 0.0], 0.0).unwrap();
//!
//! // Evaluate on grid
//! let grid = Grid::new(128, 128, 128, 0.5e-3, 0.5e-3, 0.5e-3).unwrap();
//! let analytical = wave.pressure_field(&grid, 0.0);
//!
//! // Run numerical solver (kwavers PSTD)
//! // let numerical = run_pstd_solver(...);
//!
//! // Compare results
//! // let metrics = ErrorMetrics::compute(numerical.view(), analytical.view());
//! // println!("{}", metrics.report());
//! ```
//!
//! # References
//!
//! 1. Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation
//!    and reconstruction of photoacoustic wave fields." *J. Biomed. Opt.*, 15(2), 021314.
//! 2. Treeby, B. E., Jaros, J., Rendell, A. P., & Cox, B. T. (2012). "Modeling nonlinear
//!    ultrasound propagation in heterogeneous media with power law absorption using a
//!    k-space pseudospectral method." *JASA*, 131(6), 4324-4336.
//! 3. Pierce, A. D. (1989). *Acoustics: An Introduction to Its Physical Principles
//!    and Applications*. Acoustical Society of America.
//!
//! # Author
//!
//! Ryan Clanton (@ryancinsight)
//! Sprint 217 Session 8 - k-Wave Comparison Framework

pub mod analytical;

// Re-export key types for convenience
pub use analytical::{ErrorMetrics, GaussianBeam, PlaneWave, SphericalWave};
