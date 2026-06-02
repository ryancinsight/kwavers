//! Full Kuznetsov Equation Implementation for Nonlinear Acoustics
//!
//! This module implements the complete Kuznetsov equation, which provides the most
//! comprehensive model for nonlinear acoustic wave propagation in lossy media.
//!
//! ## Theorem (Kuznetsov equation derivation)
//!
//! Starting from the compressible Navier-Stokes equations for a viscous,
//! thermally conducting fluid and retaining all second-order nonlinear terms,
//! the acoustic pressure `p = p(x, t)` satisfies:
//!
//! ```text
//! ∇²p − (1/c₀²) ∂²p/∂t²
//!     = −(β/ρ₀c₀⁴) ∂²(p²)/∂t²
//!     − (δ/c₀⁴) ∂³p/∂t³
//!     + F(x, t)
//! ```
//!
//! **Variable glossary**
//!
//! | Symbol | Meaning | SI unit |
//! |--------|---------|---------|
//! | p      | Acoustic pressure perturbation | Pa |
//! | c₀     | Small-amplitude isentropic sound speed | m s⁻¹ |
//! | ρ₀     | Equilibrium density | kg m⁻³ |
//! | β = 1 + B/(2A) | Coefficient of nonlinearity | dimensionless |
//! | B/A    | Medium nonlinearity ratio (returned by `Medium::nonlinearity`) | dimensionless |
//! | δ      | Acoustic diffusivity = (4μ/3 + μ_B)/ρ₀ + κ(1/c_v − 1/c_p)/ρ₀ | m² s⁻¹ |
//! | F      | External source forcing | Pa m⁻² |
//!
//! **Term-by-term derivation**
//!
//! 1. **Linear wave operator** `∇²p − (1/c₀²)∂²p/∂t²`: the d'Alembertian applied
//!    to p, derived from the linearized continuity and momentum equations.
//!
//! 2. **Nonlinear term** `−(β/ρ₀c₀⁴)∂²(p²)/∂t²`: arises from the Taylor expansion
//!    of the equation of state to second order in density perturbation:
//!    `p = c₀²ρ' + (B/A)c₀²(ρ')²/(2ρ₀) + O(ρ'³)`.
//!    Combined with the quadratic advection term `ρ₀u·∇u`, this produces the
//!    coefficient `β = 1 + B/(2A)`. (Kuznetsov 1971, eq. 1.)
//!
//! 3. **Diffusive term** `−(δ/c₀⁴)∂³p/∂t³`: the classical Stokes-Kirchhoff
//!    thermoviscous absorption term.  In the frequency domain this yields
//!    `α(ω) = δω²/(2c₀³)` [Np/m], consistent with measured f² power-law
//!    absorption in water at low megahertz frequencies. (Lighthill 1978, §3.4.)
//!
//! ## Theorem (hierarchy of wave equations)
//!
//! The Kuznetsov equation contains all standard nonlinear acoustic equations
//! as special cases:
//!
//! | Limit | Equation obtained |
//! |-------|-------------------|
//! | β = 0, δ = 0 | Lossless linear wave equation |
//! | β = 0 | Lossy linear wave equation (Stokes-Kirchhoff) |
//! | δ = 0 | Westervelt equation (lossless nonlinear) |
//! | Paraxial approximation | KZK equation |
//! | Full | Kuznetsov equation (this module) |
//!
//! The Westervelt equation is recovered from the Kuznetsov equation by
//! replacing `∇²p − ∂²p/(c₀²∂t²)` with `∂²p/(c₀²∂t²)` on the right-hand
//! side (equivalent for quasi-plane waves).  (Hamilton & Blackstock 1998, §4.6.)
//!
//! ## Theorem (stability: k-space spectral method)
//!
//! For the homogeneous linear part (β = 0, δ = 0), the k-space RK4 scheme is
//! unconditionally stable when the wavenumber correction κ(k) satisfies:
//!
//! ```text
//! |κ(k)·c₀·Δt·|k|| ≤ 2/√3    for all k
//! ```
//!
//! This is the k-space generalisation of the explicit leapfrog CFL condition.
//! For the Treeby & Cox (2010) correction `κ = sinc(c₀Δt|k|/2)`, the
//! left-hand side is bounded by 2/π < 2/√3, so stability holds for all
//! grid spacings and time steps satisfying the sampling theorem.
//!
//! (Treeby BE, Cox BT (2010). J. Acoust. Soc. Am. 127(5), 2741–2748.
//! DOI: 10.1121/1.3377056, Appendix.)
//!
//! ## References
//!
//! - Kuznetsov VP (1971). "Equations of nonlinear acoustics."
//!   Sov. Phys. Acoust. 16(4), 467–470.
//! - Westervelt PJ (1963). "Parametric acoustic array."
//!   J. Acoust. Soc. Am. 35(4), 535–537. DOI: 10.1121/1.1918525
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press.
//!   §4.6.
//! - Lighthill MJ (1978). Waves in Fluids. Cambridge UP. §3.4.
//! - Treeby BE, Cox BT (2010). J. Acoust. Soc. Am. 127(5), 2741–2748.
//!   DOI: 10.1121/1.3377056
//!
//! ## Module Organization:
//!
//! - `config`: Configuration structures and validation
//! - `solver`: Core Kuznetsov equation solver implementation
//! - `workspace`: Memory management and workspace structures
//! - `numerical`: Numerical methods and k-space corrections
//! - `nonlinear`: Nonlinear term computation
//! - `diffusion`: Acoustic diffusivity and absorption

// Submodules
pub mod config;
pub mod diffusion;
pub mod nonlinear;
pub mod numerical;
pub mod operator_splitting;
pub mod solver;
pub mod spectral;
pub mod workspace;

// Re-export main types
pub use config::{AcousticEquationMode, KuznetsovConfig};
pub use solver::KuznetsovWave;
pub use workspace::KuznetsovWorkspace;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod validation_test;
