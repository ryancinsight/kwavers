//! InterfaceCoupling data carrier

use crate::forward::hybrid::coupling::{
    HybridDomainInfo, InterfaceGeometry, InterfaceQualityMetrics, TransferOperators,
};

/// Interface coupling data for a single boundary
#[derive(Debug, Clone)]
pub struct InterfaceCoupling {
    /// Source domain information
    pub source_domain: HybridDomainInfo,
    /// Target domain information
    pub target_domain: HybridDomainInfo,
    /// Interface geometry
    pub interface_geometry: InterfaceGeometry,
    /// Transfer operators
    pub transfer_operators: TransferOperators,
    /// Quality metrics
    pub quality_metrics: InterfaceQualityMetrics,
}
