//! RK4 Bubble Dynamics Integration for Sonoluminescence
//!
//! Implements the coupled bubble dynamics + light emission
//! simulation loop using 4th-order Runge-Kutta integration.
//!
//! # Mathematical Foundation
//!
//! ## Keller-Miksis Equation (Compressible)
//!
//! ```text
//! (1 - Ṙ/c) R R̈ + (3/2)(1 - Ṙ/3c) Ṙ² = (1 + Ṙ/c)(p_B - p_∞)/ρ + R/(ρc) dp_B/dt
//! ```
//!
//! ## RK4 Integration
//!
//! State vector Y = [R, V] where R = radius, V = wall velocity:
//! ```text
//! k₁ = F(t_n, Y_n)
//! k₂ = F(t_n + dt/2, Y_n + dt/2 · k₁)
//! k₃ = F(t_n + dt/2, Y_n + dt/2 · k₂)
//! k₄ = F(t_n + dt, Y_n + dt · k₃)
//! Y_{n+1} = Y_n + dt/6 · (k₁ + 2k₂ + 2k₃ + k₄)
//! ```
//!
//! ## Adiabatic Thermodynamics
//!
//! Temperature: `T = T₀ (R₀/R)^{3(γ-1)}`
//! Pressure: `P = P₀ (R₀/R)^{3γ}`
//!
//! # References
//!
//! - Keller & Miksis (1980) J Acoust Soc Am 68(2):628-633
//! - Brenner et al. (2002) Rev Mod Phys 74(2):425-484

pub mod integrator;
pub mod thermodynamics;

pub use integrator::IntegratedSonoluminescence;

#[cfg(test)]
mod tests;
