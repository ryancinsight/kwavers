//! Monolithic Multiphysics Coupling
//!
//! Implements simultaneous solution of coupled multiphysics systems where all
//! physics are solved together in a single nonlinear system. Essential for
//! strong coupling with implicit stability and energy conservation.
//!
//! # Theory
//!
//! **Monolithic System:**
//!
//! Given coupled PDEs for acoustic pressure p, optical intensity I, temperature T:
//!
//! ```text
//! ∂²p/∂t² = c² ∇²p + S(I,T)           (Acoustic)
//! ∂I/∂t = -∇·F - α I + D∇²I           (Optical)
//! ∂T/∂t = κ∇²T + G(p,I)               (Thermal)
//! ```
//!
//! **Implicit Discretization:**
//!
//! ```text
//! [p^{n+1} - p^n] / Δt = c² L_p(p^{n+1}) + S(I^{n+1}, T^{n+1})
//! [I^{n+1} - I^n] / Δt = -L_I(I^{n+1}) + D∇²I^{n+1}
//! [T^{n+1} - T^n] / Δt = κ L_T(T^{n+1}) + G(p^{n+1}, I^{n+1})
//! ```
//!
//! **Unified Residual:**
//!
//! ```text
//! F(u^{n+1}) = u^{n+1} - u^n - Δt·R(u^{n+1})  =  0
//!
//! where u = [p, I, T]ᵀ and R = [R_p, R_I, R_T]ᵀ
//! ```
//!
//! **Jacobian-Free Solution:**
//!
//! Solve F(u) = 0 using Newton-Krylov without explicit ∂F/∂u assembly:
//!
//! ```text
//! u_{k+1} = u_k - J_k^{-1} F(u_k)
//!
//! where J_k·v ≈ [F(u_k + εv) - F(u_k)] / ε
//! ```
//!
//! # Advantages Over Partitioned Coupling
//!
//! | Aspect | Monolithic | Partitioned |
//! |--------|-----------|-----------|
//! | **Stability** | Unconditionally stable (implicit) | Conditional (CFL-like restrictions) |
//! | **Convergence** | Converges in ~5-10 Newton iterations | Requires many subiterations (50+) |
//! | **Accuracy** | Conservative (no iteration lag) | Iteration lag errors (10⁻⁴-10⁻²) |
//! | **Time Step** | Large Δt possible (less restrictive) | Small Δt required (more steps) |
//! | **Code Complexity** | More complex (unified solver) | Simpler (loop physics) |
//! | **Total Cost** | Lower (fewer iterations) | Higher (more steps + iterations) |
//!
//! # References
//!
//! - Knoll & Keyes (2004). "Jacobian-free Newton-Krylov methods: a survey."
//!   Journal of Computational Physics, 193(2), 357-397.
//!   DOI: 10.1016/j.jcp.2003.08.010
//!
//! - fullwave25: Nonlinear multiphysics HIFU simulator
//!   <https://github.com/pinton-lab/fullwave25>
//!   Implements monolithic acoustic-thermal-bubble coupling
//!
//! - BabelBrain: Brain HIFU therapy planning
//!   <https://github.com/ProteusMRIgHIFU/BabelBrain>
//!   Uses monolithic thermal-acoustic coupling for safety verification

pub mod config;
pub mod coupler;
pub mod residual;
pub mod utils;

pub use config::{CouplingConvergenceInfo, NewtonKrylovConfig, PhysicsCoefficients};
pub use coupler::MonolithicCoupler;
