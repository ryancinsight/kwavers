//! Poroelastic property calculations
//!
//! Reference: Johnson et al. (1987) "Theory of dynamic permeability"

use crate::physics::acoustics::mechanics::poroelastic::PoroelasticMaterial;

/// Property calculator for poroelastic materials
#[derive(Debug)]
pub struct PoroelasticProperties {
    material: PoroelasticMaterial,
}

impl PoroelasticProperties {
    /// Create new property calculator
    pub fn new(material: &PoroelasticMaterial) -> Self {
        Self {
            material: material.clone(),
        }
    }

    /// Dynamic permeability using Johnson model
    ///
    /// κ(ω) = κ₀ / (1 + jω/ω_c)
    pub fn dynamic_permeability(&self, frequency: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let omega_c = self.material.characteristic_frequency();

        // Real part of dynamic permeability
        let kappa_0 = self.material.permeability;
        kappa_0 / (1.0 + (omega / omega_c).powi(2)).sqrt()
    }

    /// Dynamic tortuosity
    pub fn dynamic_tortuosity(&self, frequency: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let omega_c = self.material.characteristic_frequency();

        let alpha_inf = self.material.tortuosity;

        // High frequency: α(ω) → α_∞
        // Low frequency: α(ω) → 1
        1.0 + (alpha_inf - 1.0) * (omega / omega_c).powi(2) / (1.0 + (omega / omega_c).powi(2))
    }
}
