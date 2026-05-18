//! InterfaceCoupling data carrier

use crate::solver::forward::hybrid::coupling::{
    DomainInfo, InterfaceGeometry, InterfaceQualityMetrics, TransferOperators,
};

/// Interface coupling data for a single boundary
#[derive(Debug, Clone)]
pub struct InterfaceCoupling {
    /// Source domain information
    pub source_domain: DomainInfo,
    /// Target domain information
    pub target_domain: DomainInfo,
    /// Interface geometry
    pub interface_geometry: InterfaceGeometry,
    /// Transfer operators
    pub transfer_operators: TransferOperators,
    /// Quality metrics
    pub quality_metrics: InterfaceQualityMetrics,
}
