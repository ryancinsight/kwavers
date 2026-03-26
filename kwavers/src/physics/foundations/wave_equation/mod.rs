//! Wave Equation Trait Specifications
//!
//! This module defines abstract trait interfaces for wave equation physics.
//! These traits specify the mathematical structure of wave propagation PDEs
//! without committing to a particular numerical method (finite difference,
//! finite element, spectral, or neural network approximation).
//!
//! TODO_AUDIT: P1 - Generalized Wave Physics - Implement complete wave equation hierarchy with nonlinear, dispersive, and multi-physics coupling
//! DEPENDS ON: physics/foundations/wave_equations/nonlinear.rs, physics/foundations/wave_equations/dispersive.rs, physics/foundations/wave_equations/coupled.rs
//! MISSING: Nonlinear wave equations (KZK, Westervelt) with exact dispersion relations
//! MISSING: Dispersive media modeling with Kramers-Kronig relations
//! MISSING: Multi-physics coupling (thermoacoustic, acousto-optic, piezoelectric)
//! MISSING: Fractional wave equations for anomalous dispersion
//! MISSING: Time-reversal acoustics for focusing and imaging
//! MISSING: Quantum acoustic effects for extreme conditions

pub mod acoustic;
pub mod core;
pub mod domain;
pub mod elastic;
pub mod source;

#[cfg(test)]
mod tests;

pub use self::core::{AutodiffWaveEquation, WaveEquation};
pub use acoustic::AcousticWaveEquation;
pub use domain::{BoundaryCondition, Domain, SpatialDimension, TimeIntegration};
pub use elastic::{AutodiffElasticWaveEquation, ElasticWaveEquation};
pub use source::SourceTerm;
