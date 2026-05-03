//! `From` conversions from domain-specific boundary types to canonical `BoundaryType`.

use super::boundary_type::BoundaryType;
use super::domain_specific::{
    AcousticBoundaryType, ElasticBoundaryType, ElectromagneticBoundaryType,
};

impl From<AcousticBoundaryType> for BoundaryType {
    fn from(acoustic: AcousticBoundaryType) -> Self {
        match acoustic {
            AcousticBoundaryType::SoundSoft => BoundaryType::Dirichlet,
            AcousticBoundaryType::SoundHard => BoundaryType::Neumann,
            AcousticBoundaryType::Impedance { impedance } => BoundaryType::Robin {
                alpha: 1.0,
                beta: impedance,
            },
            AcousticBoundaryType::Absorbing => BoundaryType::Absorbing,
            AcousticBoundaryType::Radiation => BoundaryType::Radiation,
        }
    }
}

impl From<ElectromagneticBoundaryType> for BoundaryType {
    fn from(em: ElectromagneticBoundaryType) -> Self {
        match em {
            ElectromagneticBoundaryType::PerfectElectricConductor => BoundaryType::Dirichlet,
            ElectromagneticBoundaryType::PerfectMagneticConductor => BoundaryType::Neumann,
            ElectromagneticBoundaryType::Absorbing => BoundaryType::Absorbing,
            ElectromagneticBoundaryType::Periodic { k_bloch: _ } => {
                BoundaryType::Periodic { phase: 0.0 }
            }
            ElectromagneticBoundaryType::Impedance { impedance } => {
                BoundaryType::Impedance { impedance }
            }
        }
    }
}

impl From<ElasticBoundaryType> for BoundaryType {
    fn from(elastic: ElasticBoundaryType) -> Self {
        match elastic {
            ElasticBoundaryType::Clamped => BoundaryType::Dirichlet,
            ElasticBoundaryType::Free => BoundaryType::FreeSurface,
            ElasticBoundaryType::Roller => BoundaryType::Robin {
                alpha: 1.0,
                beta: 0.0,
            },
            ElasticBoundaryType::Absorbing => BoundaryType::Absorbing,
            ElasticBoundaryType::Periodic => BoundaryType::Periodic { phase: 0.0 },
        }
    }
}
