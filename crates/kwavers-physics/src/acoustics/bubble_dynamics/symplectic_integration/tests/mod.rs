//! Validation tests for symplectic bubble integrators.
//!
//! Covers:
//! - `test_minnaert_period`: Minnaert frequency error < 0.5% at dt = T₀/200
//! - `test_hamiltonian_no_drift`: H stays in [0.5 H₀, 2 H₀] over 1000 periods
//! - `test_yoshida4_order`: Convergence order 4.0 ± 30% on SHO
//! - `test_equilibrium_preserved`: |R−R₀|/R₀ < 1e-12 at exact equilibrium

mod equilibrium;
mod hamiltonian;
mod helpers;
mod period;
mod yoshida_order;
