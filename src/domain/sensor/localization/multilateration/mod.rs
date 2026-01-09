//! Multilateration module
//!
//! Provides range-based localization solvers:
//! - `core`: Gauss–Newton LS/WLS/ML multilateration with Cholesky solve and Levenberg–Marquardt damping for numerical stability
//! - `trilateration`: Exact closed-form for three-sphere intersection with degeneracy handling, LS fallback, and fourth-sensor disambiguation
//!
//! The module re-exports commonly used types for convenience.
//!
//! Examples
//! --------
//! Basic adaptive multilateration:
//! ```
//! use kwavers::sensor::localization::array::{ArrayGeometry, Sensor, SensorArray};
//! use kwavers::sensor::localization::multilateration::{MultilaterationMethod, MultilaterationSolver};
//! use kwavers::sensor::localization::Position;
//! let sensors = vec![
//!     Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
//!     Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
//!     Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
//!     Sensor::new(3, Position::new(0.0, 0.0, 1.0)),
//! ];
//! let array = SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary);
//! let source = Position::new(0.3, 0.25, 0.2);
//! let ranges: Vec<f64> = (0..array.num_sensors())
//!     .map(|i| array.get_sensor_position(i).distance_to(&source))
//!     .collect();
//! let solver = MultilaterationSolver::new(MultilaterationMethod::LeastSquares);
//! let pos = solver.solve_adaptive(&ranges, None, &array).unwrap();
//! assert!(pos.distance_to(&source) < 0.05);
//! ```
//!
//! Trilateration with fourth-sensor disambiguation:
//! ```
//! use kwavers::sensor::localization::array::{ArrayGeometry, Sensor, SensorArray};
//! use kwavers::sensor::localization::multilateration::trilateration::TrilaterationSolver;
//! use kwavers::sensor::localization::Position;
//! let sensors = vec![
//!     Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
//!     Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
//!     Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
//!     Sensor::new(3, Position::new(0.5, 0.5, 2.0)),
//! ];
//! let array = SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary);
//! let source = Position::new(0.3, 0.25, 0.2);
//! let r0 = array.get_sensor_position(0).distance_to(&source);
//! let r1 = array.get_sensor_position(1).distance_to(&source);
//! let r2 = array.get_sensor_position(2).distance_to(&source);
//! let r3 = array.get_sensor_position(3).distance_to(&source);
//! let tri = TrilaterationSolver::new(&array);
//! let res = tri.solve_three_with_fourth([r0, r1, r2], [0, 1, 2], (3, r3)).unwrap();
//! let est = Position::from_array(res.position);
//! assert!(est.distance_to(&source) < 1e-6);
//! ```

pub mod core;
pub mod trilateration;

pub use core::{MultilaterationMethod, MultilaterationSolver};
// Numerical Stability
// -------------------
// - The Gauss–Newton path solves normal equations via Cholesky (SPD) factorization.
// - Diagonal LM damping is applied adaptively to ensure descent when ill-conditioned.
// - Steps are accepted only if the weighted residual cost decreases, preventing divergence.
//
// References
// ----------
// - K. Levenberg (1944), A method for the solution of certain nonlinear problems in least squares.
// - D. W. Marquardt (1963), An algorithm for least-squares estimation of nonlinear parameters.
// - Nocedal & Wright (2006), Numerical Optimization, Springer.
