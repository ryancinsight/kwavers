//! Coupling interface module for hybrid PSTD/FDTD solver
//!
//! This module provides the coupling infrastructure between different
//! computational domains, ensuring conservation laws and numerical stability.

mod conservation;
mod geometry;
mod interface;
mod interpolation;
mod quality;
mod transfer;

pub use conservation::ConservationEnforcer;
pub use geometry::{DomainInfo, InterfaceGeometry, LocalGridProperties};
pub use interface::{CouplingInterface, InterfaceCoupling};
pub use interpolation::{InterpolationManager, InterpolationScheme};
pub use quality::{InterfaceQualityMetrics, InterfaceQualitySummary, QualityMonitor};
pub use transfer::{TransferOperator, TransferOperators};

// Re-export the main configuration
pub use super::CouplingInterfaceConfig;
