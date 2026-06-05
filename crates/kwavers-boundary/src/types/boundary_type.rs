//! Canonical `BoundaryType` — single source of truth for boundary semantics.

use serde::{Deserialize, Serialize};

/// Canonical boundary condition types for all physics domains.
///
/// ## Mathematical Specifications
///
/// Each variant corresponds to a specific mathematical boundary condition:
///
/// - **Dirichlet**: `u = g` (essential/first-kind)
/// - **Neumann**: `∂u/∂n = g` (natural/second-kind)
/// - **Robin**: `α·u + β·∂u/∂n = g` (mixed/third-kind)
/// - **Periodic**: `u(x_min) = u(x_max) · e^(iφ)` (phase-matched periodicity)
/// - **Absorbing**: Non-reflecting boundary (PML, ABC, Sommerfeld)
/// - **Radiation**: Far-field radiation condition (Sommerfeld, Engquist-Majda)
/// - **FreeSurface**: Stress-free boundary for elastic waves
///
/// ## References
///
/// - Kreyszig, E. (2011). "Advanced Engineering Mathematics" (10th ed.).
/// - Gustafsson, B. (2008). "High Order Difference Methods for Time Dependent PDE".
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BoundaryType {
    /// Dirichlet boundary: Fixed value `u = g`
    Dirichlet,

    /// Neumann boundary: Fixed flux `∂u/∂n = g`
    Neumann,

    /// Robin boundary: Mixed condition `α·u + β·∂u/∂n = g`
    Robin {
        /// Coefficient for field value (dimensionless or matched to flux units)
        alpha: f64,
        /// Coefficient for flux term (dimensionless or matched to value units)
        beta: f64,
    },

    /// Periodic boundary: `u(x_min) = u(x_max) · e^(iφ)`
    Periodic {
        /// Phase shift between boundaries (radians)
        phase: f64,
    },

    /// Absorbing boundary: Non-reflecting condition
    Absorbing,

    /// Radiation boundary: Far-field condition (Sommerfeld)
    Radiation,

    /// Free surface: Stress-free boundary `σ·n = 0` (elastic waves)
    FreeSurface,

    /// Impedance boundary: `Z·∂u/∂n + u = 0`
    Impedance {
        /// Acoustic impedance Z = ρc (kg/m²s)
        impedance: f64,
    },
}
