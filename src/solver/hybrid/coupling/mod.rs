//! Coupling interface module for hybrid PSTD/FDTD solver
//!
//! This module provides the coupling infrastructure between different
//! computational domains, ensuring conservation laws and numerical stability.

mod geometry;
mod interpolation;
mod transfer;
mod conservation;
mod quality;
mod interface;

pub use geometry::{InterfaceGeometry, DomainInfo, LocalGridProperties};
pub use interpolation::{InterpolationScheme, InterpolationManager};
pub use transfer::{TransferOperators, TransferOperator};
pub use conservation::ConservationEnforcer;
pub use quality::{InterfaceQualityMetrics, QualityMonitor, InterfaceQualitySummary};
pub use interface::{CouplingInterface, InterfaceCoupling};

// Re-export the main configuration
pub use super::CouplingInterfaceConfig;