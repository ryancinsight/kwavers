// Ordinary Differential Equation (ODE) Solvers
//
// This module provides numerical time-integration methods for ODE systems
// arising from physics simulations. It serves as the solver-layer
// implementation for time-stepping, decoupled from physics-layer equation definitions.
//
// ## Architecture
//
// - **Physics layer** (`physics/`): Equations of motion, state types, parameter types
// - **Solver layer** (`solver/forward/ode/`): Numerical integration algorithms
//
// This separation follows the Single Responsibility Principle (SRP) and ensures
// that numerical methods can be reused across different physics domains.
//
// ## Provided Solvers
//
// ### Adaptive Runge-Kutta
// - `AdaptiveBubbleIntegrator`: RK4(5) with Richardson extrapolation error control
//   for stiff bubble dynamics ODEs (Keller-Miksis equation).
//
// ### IMEX (Implicit-Explicit)
// - `BubbleIMEXIntegrator`: IMEX methods for stiff/non-stiff term splitting
//   in bubble dynamics with thermal mass transfer.
//
// ## Literature References
//
// 1. **Hairer & Wanner (1996)**. "Solving Ordinary Differential Equations II:
//    Stiff and Differential-Algebraic Problems", 2nd Ed., Springer.
//    - Adaptive time-stepping, error control, stability regions
//
// 2. **Ascher, Ruuth & Spiteri (1997)**. "Implicit-explicit Runge-Kutta methods
//    for time-dependent PDEs", Applied Numerical Mathematics, 25(2-3), 151-167.
//
// 3. **Kennedy & Carpenter (2003)**. "Additive Runge-Kutta schemes for
//    convection-diffusion-reaction equations", Applied Numerical Mathematics,
//    44(1-2), 139-181.
//
// 4. **Storey & Szeri (2000)**. "Water vapour, sonoluminescence and
//    sonochemistry", Proceedings of the Royal Society A, 456(1999), 1685-1709.
//    - Time scales and stiffness in bubble dynamics
//
// 5. **Lauterborn & Kurz (2010)**. "Physics of bubble oscillations",
//    Reports on Progress in Physics, 73(10), 106501.
//    - Numerical challenges in bubble dynamics integration

// Bubble ODE integrators
pub mod bubble_adaptive;
pub mod bubble_imex;
pub mod bubble_symplectic;

// Re-exports for convenience
pub use bubble_adaptive::{
    integrate_bubble_dynamics_adaptive, AdaptiveBubbleConfig, AdaptiveBubbleIntegrator,
    IntegrationStatistics,
};
pub use bubble_imex::{integrate_bubble_dynamics_imex, BubbleIMEXConfig, BubbleIMEXIntegrator};
pub use bubble_symplectic::{
    integrate_bubble_dynamics_symplectic, stormer_verlet_step, yoshida4_step,
    BubbleSymplecticIntegrator, SymplecticConfig,
};
