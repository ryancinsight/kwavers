//! Stone fracture mechanics for lithotripsy simulation.
//!
//! This module implements material fracture models for kidney stone fragmentation
//! under shock wave loading.

/// Stone material properties.
#[derive(Debug, Clone)]
pub struct StoneMaterial {
    /// Density (kg/m³)
    pub density: f64,
    /// Young's modulus (Pa)
    pub youngs_modulus: f64,
    /// Poisson's ratio
    pub poisson_ratio: f64,
    /// Tensile strength (Pa)
    pub tensile_strength: f64,
}

impl StoneMaterial {
    /// Calcium oxalate monohydrate (common kidney stone)
    pub fn calcium_oxalate_monohydrate() -> Self {
        Self {
            density: 2000.0,
            youngs_modulus: 20e9,
            poisson_ratio: 0.3,
            tensile_strength: 5e6,
        }
    }
}

/// Stone fracture mechanics model.
#[derive(Debug, Clone)]
pub struct StoneFractureModel {
    material: StoneMaterial,
}

impl StoneFractureModel {
    /// Create new fracture model for given material.
    pub fn new(material: StoneMaterial) -> crate::core::error::KwaversResult<Self> {
        Ok(Self { material })
    }

    /// Get material properties.
    pub fn material(&self) -> &StoneMaterial {
        &self.material
    }
}
