//! Physiochemical properties of Reactive Oxygen Species

use super::types::ROSSpecies;

impl ROSSpecies {
    /// Get the name of the species
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::HydroxylRadical => "•OH",
            Self::HydrogenPeroxide => "H₂O₂",
            Self::Superoxide => "O₂•⁻",
            Self::SingletOxygen => "¹O₂",
            Self::Ozone => "O₃",
            Self::HydroperoxylRadical => "HO₂•",
            Self::AtomicOxygen => "O",
            Self::AtomicHydrogen => "H",
            Self::Peroxynitrite => "ONOO⁻",
            Self::NitricOxide => "NO",
            Self::NitrogenDioxide => "NO₂",
        }
    }

    /// Get the diffusion coefficient in water at 25°C (m²/s)
    #[must_use]
    pub fn diffusion_coefficient(&self) -> f64 {
        match self {
            Self::HydroxylRadical => 2.3e-9,
            Self::HydrogenPeroxide => 1.4e-9,
            Self::Superoxide => 1.75e-9,
            Self::SingletOxygen => 2.0e-9,
            Self::Ozone => 1.6e-9,
            Self::HydroperoxylRadical => 2.0e-9,
            Self::AtomicOxygen => 2.5e-9,
            Self::AtomicHydrogen => 7.0e-9,
            Self::Peroxynitrite => 1.0e-9,   // Approximate
            Self::NitricOxide => 1.0e-9,     // Approximate
            Self::NitrogenDioxide => 1.2e-9, // Approximate
        }
    }

    /// Get the lifetime in pure water (seconds)
    #[must_use]
    pub fn lifetime_water(&self) -> f64 {
        match self {
            Self::HydroxylRadical => 1e-9,     // 1 ns
            Self::HydrogenPeroxide => 1e3,     // Stable
            Self::Superoxide => 1e-6,          // 1 μs
            Self::SingletOxygen => 3.5e-6,     // 3.5 μs
            Self::Ozone => 1e2,                // 100 s
            Self::HydroperoxylRadical => 1e-6, // 1 μs
            Self::AtomicOxygen => 1e-12,       // 1 ps
            Self::AtomicHydrogen => 1e-12,     // 1 ps
            Self::Peroxynitrite => 1e-6,       // 1 μs
            Self::NitricOxide => 1e-6,         // 1 μs
            Self::NitrogenDioxide => 1e-6,     // 1 μs
        }
    }

    /// Get the standard reduction potential (V vs SHE)
    #[must_use]
    pub fn reduction_potential(&self) -> f64 {
        match self {
            Self::HydroxylRadical => 2.80, // Strongest oxidant
            Self::HydrogenPeroxide => 1.78,
            Self::Superoxide => -0.33, // Can act as reductant
            Self::SingletOxygen => 0.65,
            Self::Ozone => 2.07,
            Self::HydroperoxylRadical => 1.50,
            Self::AtomicOxygen => 2.42,
            Self::AtomicHydrogen => -2.30, // Strong reductant
            Self::Peroxynitrite => 0.80,   // Reductant
            Self::NitricOxide => 0.90,     // Reductant
            Self::NitrogenDioxide => 1.05, // Oxidant
        }
    }
}
