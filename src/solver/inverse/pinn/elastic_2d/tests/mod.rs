//! Test Suite for 2D Elastic Wave PINN Solver
//!
//! This module contains comprehensive validation tests for the PINN implementation,
//! organized by testing strategy:
//!
//! - `gradient_validation`: Property-based tests comparing autodiff vs finite differences
//! - Future: `analytical_solutions`: Validation against known analytic solutions
//! - Future: `convergence_tests`: Convergence studies for various scenarios
//!
//! # Testing Philosophy
//!
//! Following the verification hierarchy:
//! 1. Mathematical specifications (formal theorems)
//! 2. Property-based tests (this module)
//! 3. Unit/integration tests (in respective modules)
//! 4. Performance benchmarks (in benches/)
//!
//! # Gradient Validation
//!
//! The most critical validation is ensuring autodiff gradients are mathematically
//! correct. We validate by comparison with finite difference approximations:
//!
//! - First derivatives: ∂u/∂x, ∂u/∂y, ∂u/∂t
//! - Second derivatives: ∂²u/∂x², ∂²u/∂y², ∂²u/∂t²
//! - Mixed derivatives: ∂²u/∂x∂y
//!
//! Acceptance criteria:
//! - Relative error < 1e-3 for first derivatives
//! - Relative error < 1e-2 for second derivatives
//!
//! # Running Tests
//!
//! ```bash
//! # Run all PINN tests
//! cargo test --features pinn --lib
//!
//! # Run only gradient validation tests
//! cargo test --features pinn --lib gradient_validation
//!
//! # Run with output
//! cargo test --features pinn --lib gradient_validation -- --nocapture
//! ```

#[cfg(all(test, feature = "pinn"))]
pub mod gradient_validation;
