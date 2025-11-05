//! Poroelastic Material Properties
//!
//! Defines the constitutive properties for poroelastic media including
//! solid skeleton properties, fluid properties, and coupling parameters.

use crate::error::{KwaversError, KwaversResult};

/// Complete poroelastic material properties
#[derive(Debug, Clone)]
pub struct PoroelasticProperties {
    /// Solid skeleton properties
    pub solid: SolidProperties,
    /// Fluid properties
    pub fluid: FluidProperties,
    /// Coupling parameters
    pub coupling: CouplingParameters,
    /// Porosity (volume fraction of fluid)
    pub porosity: f64,
    /// Tortuosity (path length factor for fluid)
    pub tortuosity: f64,
    /// Permeability (hydraulic conductivity)
    pub permeability: f64,
}

impl PoroelasticProperties {
    /// Create poroelastic properties for liver tissue
    pub fn liver() -> Self {
        Self {
            solid: SolidProperties {
                density: 1060.0,     // kg/m³
                bulk_modulus: 2.1e9, // Pa
                shear_modulus: 1.2e6, // Pa (soft tissue)
                poisson_ratio: 0.49,
            },
            fluid: FluidProperties {
                density: 1000.0,        // kg/m³ (blood/plasma)
                bulk_modulus: 2.2e9,    // Pa
                viscosity: 0.001,       // Pa·s (blood viscosity)
            },
            coupling: CouplingParameters {
                biot_coefficient: 0.95,
                biot_modulus: 1.8e10, // Pa
            },
            porosity: 0.15,        // 15% fluid content
            tortuosity: 1.8,       // Typical for liver
            permeability: 1e-10,   // m² (hydraulic permeability)
        }
    }

    /// Create poroelastic properties for kidney tissue
    pub fn kidney() -> Self {
        Self {
            solid: SolidProperties {
                density: 1040.0,
                bulk_modulus: 1.8e9,
                shear_modulus: 8e5,
                poisson_ratio: 0.48,
            },
            fluid: FluidProperties {
                density: 1000.0,
                bulk_modulus: 2.2e9,
                viscosity: 0.0008,
            },
            coupling: CouplingParameters {
                biot_coefficient: 0.92,
                biot_modulus: 2.2e10,
            },
            porosity: 0.12,
            tortuosity: 2.1,
            permeability: 8e-11,
        }
    }

    /// Create poroelastic properties for brain tissue
    pub fn brain() -> Self {
        Self {
            solid: SolidProperties {
                density: 1045.0,
                bulk_modulus: 1.5e9,
                shear_modulus: 5e5,
                poisson_ratio: 0.47,
            },
            fluid: FluidProperties {
                density: 1007.0,        // CSF density
                bulk_modulus: 2.1e9,
                viscosity: 0.0012,      // CSF viscosity
            },
            coupling: CouplingParameters {
                biot_coefficient: 0.88,
                biot_modulus: 1.5e10,
            },
            porosity: 0.08,        // Lower porosity in brain
            tortuosity: 1.6,
            permeability: 5e-11,
        }
    }

    /// Create poroelastic properties for trabecular bone
    pub fn trabecular_bone() -> Self {
        Self {
            solid: SolidProperties {
                density: 1200.0,
                bulk_modulus: 1.2e10,
                shear_modulus: 3.5e9,
                poisson_ratio: 0.3,
            },
            fluid: FluidProperties {
                density: 1000.0,        // Marrow fluid
                bulk_modulus: 2.3e9,
                viscosity: 0.005,       // Marrow viscosity
            },
            coupling: CouplingParameters {
                biot_coefficient: 0.75,
                biot_modulus: 8e10,
            },
            porosity: 0.6,         // High porosity in trabecular bone
            tortuosity: 3.2,
            permeability: 1e-8,    // Higher permeability
        }
    }

    /// Validate material properties
    pub fn validate(&self) -> KwaversResult<()> {
        // Check physical constraints
        if self.porosity <= 0.0 || self.porosity >= 1.0 {
            return Err(KwaversError::InvalidInput(
                "Porosity must be between 0 and 1".to_string()
            ));
        }

        if self.tortuosity < 1.0 {
            return Err(KwaversError::InvalidInput(
                "Tortuosity must be >= 1".to_string()
            ));
        }

        if self.permeability <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Permeability must be positive".to_string()
            ));
        }

        // Validate component properties
        self.solid.validate()?;
        self.fluid.validate()?;
        self.coupling.validate()?;

        Ok(())
    }

    /// Compute effective density (mixture density)
    pub fn effective_density(&self) -> f64 {
        self.porosity * self.fluid.density + (1.0 - self.porosity) * self.solid.density
    }

    /// Compute effective bulk modulus
    pub fn effective_bulk_modulus(&self) -> f64 {
        // Simplified mixture rule - in practice more complex
        let k_s = self.solid.bulk_modulus;
        let k_f = self.fluid.bulk_modulus;
        let phi = self.porosity;

        1.0 / ((1.0 - phi) / k_s + phi / k_f)
    }

    /// Compute characteristic frequency (Biot frequency)
    pub fn biot_frequency(&self) -> f64 {
        let eta = self.fluid.viscosity;
        let kappa = self.permeability;
        let rho_f = self.fluid.density;
        let alpha = self.coupling.biot_coefficient;

        // ω_c = (η / (κ ρ_f α²))
        eta / (kappa * rho_f * alpha * alpha)
    }
}

/// Solid skeleton properties
#[derive(Debug, Clone)]
pub struct SolidProperties {
    /// Density (kg/m³)
    pub density: f64,
    /// Bulk modulus (Pa)
    pub bulk_modulus: f64,
    /// Shear modulus (Pa)
    pub shear_modulus: f64,
    /// Poisson's ratio
    pub poisson_ratio: f64,
}

impl SolidProperties {
    pub fn validate(&self) -> KwaversResult<()> {
        if self.density <= 0.0 {
            return Err(KwaversError::InvalidInput("Solid density must be positive".to_string()));
        }
        if self.bulk_modulus <= 0.0 {
            return Err(KwaversError::InvalidInput("Bulk modulus must be positive".to_string()));
        }
        if self.shear_modulus < 0.0 {
            return Err(KwaversError::InvalidInput("Shear modulus must be non-negative".to_string()));
        }
        if self.poisson_ratio < -1.0 || self.poisson_ratio > 0.5 {
            return Err(KwaversError::InvalidInput("Poisson's ratio must be between -1 and 0.5".to_string()));
        }
        Ok(())
    }

    /// Compute Young's modulus from bulk and shear moduli
    pub fn youngs_modulus(&self) -> f64 {
        let k = self.bulk_modulus;
        let mu = self.shear_modulus;

        9.0 * k * mu / (3.0 * k + mu)
    }

    /// Compute Lame's first parameter
    pub fn lame_lambda(&self) -> f64 {
        let k = self.bulk_modulus;
        let mu = self.shear_modulus;

        k - (2.0 / 3.0) * mu
    }
}

/// Fluid properties
#[derive(Debug, Clone)]
pub struct FluidProperties {
    /// Density (kg/m³)
    pub density: f64,
    /// Bulk modulus (Pa)
    pub bulk_modulus: f64,
    /// Viscosity (Pa·s)
    pub viscosity: f64,
}

impl FluidProperties {
    pub fn validate(&self) -> KwaversResult<()> {
        if self.density <= 0.0 {
            return Err(KwaversError::InvalidInput("Fluid density must be positive".to_string()));
        }
        if self.bulk_modulus <= 0.0 {
            return Err(KwaversError::InvalidInput("Fluid bulk modulus must be positive".to_string()));
        }
        if self.viscosity < 0.0 {
            return Err(KwaversError::InvalidInput("Fluid viscosity must be non-negative".to_string()));
        }
        Ok(())
    }

    /// Compute speed of sound in fluid
    pub fn speed_of_sound(&self) -> f64 {
        (self.bulk_modulus / self.density).sqrt()
    }
}

/// Coupling parameters between solid and fluid phases
#[derive(Debug, Clone)]
pub struct CouplingParameters {
    /// Biot coefficient (α)
    pub biot_coefficient: f64,
    /// Biot modulus (M)
    pub biot_modulus: f64,
}

impl CouplingParameters {
    pub fn validate(&self) -> KwaversResult<()> {
        if self.biot_coefficient < 0.0 || self.biot_coefficient > 1.0 {
            return Err(KwaversError::InvalidInput(
                "Biot coefficient must be between 0 and 1".to_string()
            ));
        }
        if self.biot_modulus <= 0.0 {
            return Err(KwaversError::InvalidInput("Biot modulus must be positive".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liver_properties() {
        let liver = PoroelasticProperties::liver();
        assert!(liver.validate().is_ok());
        assert!(liver.porosity > 0.0 && liver.porosity < 1.0);
        assert!(liver.biot_frequency() > 0.0);
    }

    #[test]
    fn test_brain_properties() {
        let brain = PoroelasticProperties::brain();
        assert!(brain.validate().is_ok());
        assert!(brain.porosity < 0.2); // Brain has lower porosity
    }

    #[test]
    fn test_bone_properties() {
        let bone = PoroelasticProperties::trabecular_bone();
        assert!(bone.validate().is_ok());
        assert!(bone.porosity > 0.5); // Bone has high porosity
    }

    #[test]
    fn test_effective_properties() {
        let liver = PoroelasticProperties::liver();
        let effective_density = liver.effective_density();
        let effective_bulk = liver.effective_bulk_modulus();

        assert!(effective_density > 1000.0 && effective_density < 1100.0);
        assert!(effective_bulk > 2e9);
    }

    #[test]
    fn test_solid_properties_validation() {
        let mut props = SolidProperties {
            density: 1000.0,
            bulk_modulus: 1e9,
            shear_modulus: 1e8,
            poisson_ratio: 0.3,
        };

        assert!(props.validate().is_ok());
        assert!(props.youngs_modulus() > 0.0);
        assert!(props.lame_lambda() > 0.0);

        // Test invalid properties
        props.poisson_ratio = 0.8; // Invalid
        assert!(props.validate().is_err());
    }

    #[test]
    fn test_fluid_properties() {
        let props = FluidProperties {
            density: 1000.0,
            bulk_modulus: 2e9,
            viscosity: 0.001,
        };

        assert!(props.validate().is_ok());
        assert!(props.speed_of_sound() > 1000.0); // Should be around 1414 m/s
    }
}

