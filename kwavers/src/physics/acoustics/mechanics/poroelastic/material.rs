use crate::core::error::{KwaversError, KwaversResult};

/// Poroelastic material properties
///
/// Represents a biphasic material with solid matrix and fluid phase
#[derive(Debug, Clone)]
pub struct PoroelasticMaterial {
    /// Porosity (0 < φ < 1)
    pub porosity: f64,
    /// Solid density (kg/m³)
    pub solid_density: f64,
    /// Fluid density (kg/m³)
    pub fluid_density: f64,
    /// Solid bulk modulus (Pa)
    pub solid_bulk_modulus: f64,
    /// Fluid bulk modulus (Pa)
    pub fluid_bulk_modulus: f64,
    /// Shear modulus of drained frame (Pa)
    pub shear_modulus: f64,
    /// Permeability (m²)
    pub permeability: f64,
    /// Fluid viscosity (Pa·s)
    pub fluid_viscosity: f64,
    /// Tortuosity (α ≥ 1)
    pub tortuosity: f64,
}

impl Default for PoroelasticMaterial {
    fn default() -> Self {
        // Typical trabecular bone properties
        Self {
            porosity: 0.3,              // 30% porosity
            solid_density: 2000.0,      // kg/m³
            fluid_density: 1000.0,      // Water
            solid_bulk_modulus: 10e9,   // 10 GPa
            fluid_bulk_modulus: 2.25e9, // 2.25 GPa
            shear_modulus: 3.5e9,       // 3.5 GPa
            permeability: 1e-9,         // 1 nm² (Darcy)
            fluid_viscosity: 1e-3,      // Water at 20°C
            tortuosity: 1.5,            // Typical for bone
        }
    }
}

impl PoroelasticMaterial {
    /// Create new poroelastic material with validation
    pub fn new(
        porosity: f64,
        solid_density: f64,
        fluid_density: f64,
        solid_bulk_modulus: f64,
        fluid_bulk_modulus: f64,
        shear_modulus: f64,
        permeability: f64,
        fluid_viscosity: f64,
        tortuosity: f64,
    ) -> KwaversResult<Self> {
        if !(0.0..=1.0).contains(&porosity) {
            return Err(KwaversError::InvalidInput(
                "Porosity must be between 0 and 1".to_string(),
            ));
        }
        if solid_density <= 0.0 || fluid_density <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Densities must be positive".to_string(),
            ));
        }
        if solid_bulk_modulus <= 0.0 || fluid_bulk_modulus <= 0.0 || shear_modulus <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Moduli must be positive".to_string(),
            ));
        }
        if permeability <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Permeability must be positive".to_string(),
            ));
        }
        if tortuosity < 1.0 {
            return Err(KwaversError::InvalidInput(
                "Tortuosity must be ≥ 1".to_string(),
            ));
        }

        Ok(Self {
            porosity,
            solid_density,
            fluid_density,
            solid_bulk_modulus,
            fluid_bulk_modulus,
            shear_modulus,
            permeability,
            fluid_viscosity,
            tortuosity,
        })
    }

    /// Create from tissue type
    pub fn from_tissue_type(tissue: &str) -> KwaversResult<Self> {
        match tissue {
            "trabecular_bone" => Ok(Self::default()),
            "cortical_bone" => Ok(Self {
                porosity: 0.05, // 5% porosity
                solid_density: 2000.0,
                fluid_density: 1000.0,
                solid_bulk_modulus: 20e9, // 20 GPa
                fluid_bulk_modulus: 2.25e9,
                shear_modulus: 7e9,  // 7 GPa
                permeability: 1e-12, // Very low
                fluid_viscosity: 1e-3,
                tortuosity: 1.2,
            }),
            "liver" => Ok(Self {
                porosity: 0.15, // 15% vascular space
                solid_density: 1050.0,
                fluid_density: 1000.0,
                solid_bulk_modulus: 2.5e9, // 2.5 GPa
                fluid_bulk_modulus: 2.25e9,
                shear_modulus: 5e3, // 5 kPa (soft)
                permeability: 1e-11,
                fluid_viscosity: 1e-3,
                tortuosity: 1.3,
            }),
            "lung" => Ok(Self {
                porosity: 0.8,             // 80% air-filled
                solid_density: 300.0,      // Low density
                fluid_density: 1.2,        // Air
                solid_bulk_modulus: 1e6,   // Very soft
                fluid_bulk_modulus: 1.4e5, // Air at 1 atm
                shear_modulus: 1e3,        // 1 kPa
                permeability: 1e-8,        // High permeability
                fluid_viscosity: 1.8e-5,   // Air
                tortuosity: 2.0,           // Complex structure
            }),
            _ => Err(KwaversError::InvalidInput(format!(
                "Unknown tissue type: {}",
                tissue
            ))),
        }
    }

    /// Calculate bulk density ρ = (1-φ)ρ_s + φρ_f
    pub fn bulk_density(&self) -> f64 {
        (1.0 - self.porosity) * self.solid_density + self.porosity * self.fluid_density
    }

    /// Calculate effective bulk modulus (Gassmann's equation)
    pub fn effective_bulk_modulus(&self) -> f64 {
        let k_s = self.solid_bulk_modulus;
        let k_f = self.fluid_bulk_modulus;
        let phi = self.porosity;

        let term1 = (1.0 - phi) / k_s;
        let term2 = phi / k_f;
        1.0 / (term1 + term2)
    }

    /// Calculate characteristic frequency (Biot critical frequency)
    ///
    /// ω_c = (φ η) / (κ ρ_f α)
    pub fn characteristic_frequency(&self) -> f64 {
        let phi = self.porosity;
        let eta = self.fluid_viscosity;
        let kappa = self.permeability;
        let rho_f = self.fluid_density;
        let alpha = self.tortuosity;

        (phi * eta) / (kappa * rho_f * alpha)
    }
}
