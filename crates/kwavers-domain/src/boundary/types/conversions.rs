//! `From` conversions from domain-specific boundary types to canonical `BoundaryType`.

use super::boundary_type::BoundaryType;
use super::domain_specific::{
    DomainAcousticBoundaryType, ElasticBoundaryType, ElectromagneticBoundaryType,
};

impl From<DomainAcousticBoundaryType> for BoundaryType {
    fn from(acoustic: DomainAcousticBoundaryType) -> Self {
        match acoustic {
            DomainAcousticBoundaryType::SoundSoft => Self::Dirichlet,
            DomainAcousticBoundaryType::SoundHard => Self::Neumann,
            DomainAcousticBoundaryType::Impedance { impedance } => Self::Robin {
                alpha: 1.0,
                beta: impedance,
            },
            DomainAcousticBoundaryType::Absorbing => Self::Absorbing,
            DomainAcousticBoundaryType::Radiation => Self::Radiation,
        }
    }
}

impl From<ElectromagneticBoundaryType> for BoundaryType {
    fn from(em: ElectromagneticBoundaryType) -> Self {
        match em {
            ElectromagneticBoundaryType::PerfectElectricConductor => Self::Dirichlet,
            ElectromagneticBoundaryType::PerfectMagneticConductor => Self::Neumann,
            ElectromagneticBoundaryType::Absorbing => Self::Absorbing,
            ElectromagneticBoundaryType::Periodic { k_bloch: _ } => Self::Periodic { phase: 0.0 },
            ElectromagneticBoundaryType::Impedance { impedance } => Self::Impedance { impedance },
        }
    }
}

impl From<ElasticBoundaryType> for BoundaryType {
    fn from(elastic: ElasticBoundaryType) -> Self {
        match elastic {
            ElasticBoundaryType::Clamped => Self::Dirichlet,
            ElasticBoundaryType::Free => Self::FreeSurface,
            ElasticBoundaryType::Roller => Self::Robin {
                alpha: 1.0,
                beta: 0.0,
            },
            ElasticBoundaryType::Absorbing => Self::Absorbing,
            ElasticBoundaryType::Periodic => Self::Periodic { phase: 0.0 },
        }
    }
}
