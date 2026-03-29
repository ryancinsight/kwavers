//! Wave Equation Trait Specifications
//!
//! This module defines abstract trait interfaces for wave equation physics.
//! These traits specify the mathematical structure of wave propagation PDEs
//! without committing to a particular numerical method (finite difference,
//! finite element, spectral, or neural network approximation).
//!
//! Implemented concrete wave equations:
//!
//! * **Westervelt equation** (nonlinear FDTD):
//!   [`crate::solver::forward::nonlinear::westervelt::WesterveltSolver`]
//! * **PSTD pseudospectral solver** with power-law absorption (Treeby & Cox 2010):
//!   [`crate::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver`]
//! * **Keller-Miksis bubble dynamics** (compressible, Keller & Miksis 1980):
//!   [`crate::physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel`]
//! * **Multi-bubble secondary Bjerknes coupling** (Crum 1975):
//!   [`crate::physics::acoustics::bubble_dynamics::bubble_field::BubbleField`]
//!
//! Remaining extensions (dispersive Kramers-Kronig media, fractional-order
//! equations, time-reversal acoustics, quantum acoustic effects) are future work.

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
