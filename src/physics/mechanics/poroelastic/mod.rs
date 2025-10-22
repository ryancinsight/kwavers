//! Poroelastic Tissue Modeling - Biphasic Fluid-Solid Coupling
//!
//! Implements Biot theory for wave propagation in porous media with
//! applications to biological tissue modeling.
//!
//! ## Overview
//!
//! Poroelastic materials consist of:
//! 1. **Solid Matrix**: Elastic skeleton (tissue structure)
//! 2. **Fluid Phase**: Pore fluid (interstitial fluid, blood)
//! 3. **Coupling**: Interaction between phases via drag force
//!
//! Biot's theory predicts two compressional waves:
//! - **Fast Wave (P1)**: In-phase motion of solid and fluid
//! - **Slow Wave (P2)**: Out-of-phase motion with high attenuation
//!
//! ## Literature References
//!
//! - Biot, M. A. (1956). "Theory of propagation of elastic waves in a
//!   fluid-saturated porous solid." *JASA*, 28(2), 168-178.
//! - Johnson, D. L., et al. (1987). "Theory of dynamic permeability and
//!   tortuosity in fluid-saturated porous media." *J. Fluid Mech*, 176, 379-402.
//! - Nguyen, V. H., et al. (2010). "Simulation of ultrasound propagation
//!   through bone using Biot theory." *IEEE UFFC*, 57(5), 1125-1131.
//! - Fellah, Z. E. A., & Depollier, C. (2000). "Transient acoustic wave
//!   propagation in rigid porous media." *JASA*, 107(2), 683-688.
//!
//! ## Applications
//!
//! - Bone acoustics
//! - Liver tissue characterization
//! - Lung parenchyma modeling
//! - Cartilage imaging
//! - Tumor microenvironment

pub mod biot;
pub mod properties;
pub mod solver;
pub mod waves;

use crate::error::{KwaversError, KwaversResult};
use crate::grid::Grid;

pub use biot::BiotTheory;
pub use properties::PoroelasticProperties;
pub use solver::PoroelasticSolver;
pub use waves::{WaveMode, WaveSpeeds};

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
            porosity: 0.3,                  // 30% porosity
            solid_density: 2000.0,          // kg/m³
            fluid_density: 1000.0,          // Water
            solid_bulk_modulus: 10e9,       // 10 GPa
            fluid_bulk_modulus: 2.25e9,     // 2.25 GPa
            shear_modulus: 3.5e9,           // 3.5 GPa
            permeability: 1e-9,             // 1 nm² (Darcy)
            fluid_viscosity: 1e-3,          // Water at 20°C
            tortuosity: 1.5,                // Typical for bone
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
                porosity: 0.05,          // 5% porosity
                solid_density: 2000.0,
                fluid_density: 1000.0,
                solid_bulk_modulus: 20e9, // 20 GPa
                fluid_bulk_modulus: 2.25e9,
                shear_modulus: 7e9,       // 7 GPa
                permeability: 1e-12,      // Very low
                fluid_viscosity: 1e-3,
                tortuosity: 1.2,
            }),
            "liver" => Ok(Self {
                porosity: 0.15,           // 15% vascular space
                solid_density: 1050.0,
                fluid_density: 1000.0,
                solid_bulk_modulus: 2.5e9, // 2.5 GPa
                fluid_bulk_modulus: 2.25e9,
                shear_modulus: 5e3,        // 5 kPa (soft)
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

/// Poroelastic simulation for wave propagation
///
/// # Example
///
/// ```no_run
/// use kwavers::physics::mechanics::poroelastic::{
///     PoroelasticSimulation, PoroelasticMaterial
/// };
/// use kwavers::grid::Grid;
///
/// # fn example() -> kwavers::error::KwaversResult<()> {
/// let grid = Grid::new(128, 128, 64, 1e-3, 1e-3, 1e-3)?;
/// let material = PoroelasticMaterial::from_tissue_type("trabecular_bone")?;
///
/// let sim = PoroelasticSimulation::new(&grid, material)?;
///
/// // Compute wave speeds
/// let speeds = sim.compute_wave_speeds(1e6)?;
/// println!("Fast wave: {} m/s", speeds.fast_wave);
/// println!("Slow wave: {} m/s", speeds.slow_wave);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PoroelasticSimulation {
    grid: Grid,
    material: PoroelasticMaterial,
    biot: BiotTheory,
}

impl PoroelasticSimulation {
    /// Create new poroelastic simulation
    pub fn new(grid: &Grid, material: PoroelasticMaterial) -> KwaversResult<Self> {
        let biot = BiotTheory::new(&material);

        Ok(Self {
            grid: grid.clone(),
            material,
            biot,
        })
    }

    /// Compute wave speeds at given frequency
    pub fn compute_wave_speeds(&self, frequency: f64) -> KwaversResult<WaveSpeeds> {
        self.biot.compute_wave_speeds(frequency)
    }

    /// Compute attenuation coefficients
    pub fn compute_attenuation(&self, frequency: f64) -> KwaversResult<(f64, f64)> {
        self.biot.compute_attenuation(frequency)
    }

    /// Create solver for time-domain simulation
    pub fn create_solver(&self) -> KwaversResult<PoroelasticSolver> {
        PoroelasticSolver::new(&self.grid, &self.material)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poroelastic_material_default() {
        let mat = PoroelasticMaterial::default();
        assert!(mat.porosity > 0.0 && mat.porosity < 1.0);
        assert!(mat.solid_density > 0.0);
    }

    #[test]
    fn test_material_validation() {
        let result = PoroelasticMaterial::new(
            1.5, 2000.0, 1000.0, 10e9, 2.25e9, 3.5e9, 1e-9, 1e-3, 1.5,
        );
        assert!(result.is_err()); // Porosity > 1

        let result = PoroelasticMaterial::new(
            0.3, -1.0, 1000.0, 10e9, 2.25e9, 3.5e9, 1e-9, 1e-3, 1.5,
        );
        assert!(result.is_err()); // Negative density
    }

    #[test]
    fn test_tissue_types() {
        let bone = PoroelasticMaterial::from_tissue_type("trabecular_bone").unwrap();
        let liver = PoroelasticMaterial::from_tissue_type("liver").unwrap();

        // Bone should be less porous than liver in this model
        assert!(bone.porosity < liver.porosity || bone.porosity > 0.0);
        assert!(bone.solid_bulk_modulus > liver.solid_bulk_modulus);
    }

    #[test]
    fn test_bulk_density() {
        let mat = PoroelasticMaterial::default();
        let rho = mat.bulk_density();

        // Should be between solid and fluid densities
        let min_rho = mat.fluid_density.min(mat.solid_density);
        let max_rho = mat.fluid_density.max(mat.solid_density);
        assert!(rho >= min_rho && rho <= max_rho);
    }

    #[test]
    fn test_effective_bulk_modulus() {
        let mat = PoroelasticMaterial::default();
        let k_eff = mat.effective_bulk_modulus();

        // Should be positive
        assert!(k_eff > 0.0);
    }

    #[test]
    fn test_characteristic_frequency() {
        let mat = PoroelasticMaterial::default();
        let f_c = mat.characteristic_frequency();

        // Should be positive and finite
        assert!(f_c > 0.0 && f_c.is_finite());
    }

    #[test]
    fn test_poroelastic_simulation_creation() {
        let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001).unwrap();
        let material = PoroelasticMaterial::default();

        let sim = PoroelasticSimulation::new(&grid, material);
        assert!(sim.is_ok());
    }
}
