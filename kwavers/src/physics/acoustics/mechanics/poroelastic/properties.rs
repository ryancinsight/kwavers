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
    #[must_use] 
    pub fn new(material: &PoroelasticMaterial) -> Self {
        Self {
            material: material.clone(),
        }
    }

    /// Dynamic permeability using Johnson model
    ///
    /// κ(ω) = κ₀ / (1 + jω/ω_c)
    #[must_use] 
    pub fn dynamic_permeability(&self, frequency: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let omega_c = self.material.characteristic_frequency();

        // Real part of dynamic permeability
        let kappa_0 = self.material.permeability;
        kappa_0 / (omega / omega_c).mul_add(omega / omega_c, 1.0).sqrt()
    }

    /// Dynamic tortuosity
    #[must_use] 
    pub fn dynamic_tortuosity(&self, frequency: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let omega_c = self.material.characteristic_frequency();

        let alpha_inf = self.material.tortuosity;

        // High frequency: α(ω) → α_∞
        // Low frequency: α(ω) → 1
        1.0 + (alpha_inf - 1.0) * (omega / omega_c).powi(2) / (omega / omega_c).mul_add(omega / omega_c, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::acoustics::mechanics::poroelastic::PoroelasticMaterial;

    /// `dynamic_permeability` at zero frequency equals the static permeability κ₀.
    ///
    /// Analytical: κ(ω) = κ₀/sqrt(1+(ω/ω_c)²); at ω=0 → κ(0) = κ₀.
    #[test]
    fn dynamic_permeability_at_zero_frequency_equals_static_permeability() {
        let m = PoroelasticMaterial::default();
        let props = PoroelasticProperties::new(&m);
        let kappa_dc = props.dynamic_permeability(0.0);
        assert!(
            (kappa_dc - m.permeability).abs() < m.permeability * 1e-10,
            "κ(0) must equal κ₀={:.3e}, got {kappa_dc:.3e}",
            m.permeability
        );
    }

    /// `dynamic_permeability` decreases monotonically with increasing frequency.
    #[test]
    fn dynamic_permeability_decreases_with_frequency() {
        let m = PoroelasticMaterial::default();
        let props = PoroelasticProperties::new(&m);
        let k1 = props.dynamic_permeability(1e3);
        let k2 = props.dynamic_permeability(1e6);
        let k3 = props.dynamic_permeability(1e9);
        assert!(k1 > k2, "κ(1kHz) must exceed κ(1MHz)");
        assert!(k2 > k3, "κ(1MHz) must exceed κ(1GHz)");
    }

    /// `dynamic_tortuosity` at very low frequency (ω << ω_c) approaches 1.
    ///
    /// Analytical: α(ω)=1+(α_∞-1)(ω/ω_c)²/((ω/ω_c)²+1) → 1 as ω→0.
    #[test]
    fn dynamic_tortuosity_at_low_frequency_approaches_one() {
        let m = PoroelasticMaterial::default();
        let props = PoroelasticProperties::new(&m);
        // ω_c for default bone = 200 rad/s; use f = 1e-4 Hz (ω ≈ 6.28e-4 rad/s << 200)
        let alpha_low = props.dynamic_tortuosity(1e-4);
        assert!(
            (alpha_low - 1.0).abs() < 1e-6,
            "tortuosity at f=1e-4 Hz must be ≈1.0 (got {alpha_low:.6})"
        );
    }

    /// `dynamic_tortuosity` at very high frequency (ω >> ω_c) approaches α_∞.
    ///
    /// For default bone α_∞ = 1.5; use f = 1e9 Hz (ω >> ω_c = 200 rad/s).
    #[test]
    fn dynamic_tortuosity_at_high_frequency_approaches_alpha_infinity() {
        let m = PoroelasticMaterial::default();
        let alpha_inf = m.tortuosity; // 1.5
        let props = PoroelasticProperties::new(&m);
        let alpha_high = props.dynamic_tortuosity(1e9);
        assert!(
            (alpha_high - alpha_inf).abs() < 1e-4,
            "tortuosity at 1GHz must approach α_∞={alpha_inf} (got {alpha_high:.4})"
        );
    }
}
