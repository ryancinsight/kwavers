//! Rayleigh-Plesset Equation for Bubble Dynamics
//!
//! ## Mathematical Theorem
//!
//! **Rayleigh-Plesset Equation**: RṘ + 3/2Ṙ² = (1/ρ)(p_B - p_∞ - 2σ/R - 4μṘ/R)
//!
//! **Theorem Foundation**: Derived from conservation of mass and momentum for spherical bubbles
//! in incompressible liquids (Rayleigh 1917, Plesset 1949). The equation describes the
//! radial dynamics of gas bubbles in response to acoustic forcing.
//!
//! **Physical Assumptions**:
//! - Spherical bubble geometry (valid for Mach < 0.3)
//! - Incompressible liquid (valid for frequencies < 100 MHz)
//! - Thin boundary layer approximation
//! - Surface tension and viscosity included
//!
//! **Mathematical Derivation**:
//! Starting from Euler's equation for incompressible flow and continuity equation,
//! applying spherical symmetry and boundary conditions at bubble wall.
//!
//! **Boundary Conditions**:
//! - r = R(t): Continuity of velocity and pressure (u_liquid = u_bubble = Ṙ)
//! - r → ∞: Pressure approaches p_∞ + p_acoustic(t)
//!
//! **Literature References**:
//! - Rayleigh (1917): "On the pressure developed in a liquid during the collapse of a spherical cavity"
//! - Plesset (1949): "The dynamics of cavitation bubbles" J. Appl. Mech. 16:277-282
//! - Brennen (1995): "Cavitation and Bubble Dynamics" Oxford University Press
//! - Qin et al. (2023): "Numerical investigation on acoustic cavitation characteristics"

pub mod integration;
pub mod solver;

pub use integration::integrate_bubble_dynamics_stable;
pub use solver::RayleighPlessetSolver;

#[cfg(test)]
mod tests;
