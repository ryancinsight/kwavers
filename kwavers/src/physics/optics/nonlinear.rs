//! Nonlinear optical effects
//!
//! Implements Kerr effect and other nonlinear optical phenomena important for
//! high-intensity optical and photoacoustic applications.
//!
//! References:
//! - Boyd (2008) "Nonlinear Optics"
//! - Diels & Rudolph (2006) "Ultrashort Laser Pulse Phenomena"
//! - Agrawal (2007) "Nonlinear Fiber Optics"

use crate::core::constants::fundamental::{SOUND_SPEED_TISSUE, SPEED_OF_LIGHT};

/// Kerr effect parameters
///
/// The Kerr effect describes intensity-dependent refractive index change:
/// n(I) = n₀ + n₂·I
/// where n₂ is the nonlinear refractive index coefficient
#[derive(Debug, Clone, Copy)]
pub struct KerrEffect {
    /// Linear refractive index
    pub n0: f64,
    /// Nonlinear refractive index coefficient [m²/W]
    pub n2: f64,
    /// Kerr nonlinearity parameter (simplified)
    pub chi3: f64,
}

impl KerrEffect {
    /// Create Kerr effect for a material
    #[must_use]
    pub fn new(n0: f64, n2: f64) -> Self {
        // χ³ related to n₂: n₂ ~ (3/8) * χ³ / (c·n₀²)
        let chi3 = n2 * SPEED_OF_LIGHT * n0 * n0 * (8.0 / 3.0);
        Self { n0, n2, chi3 }
    }

    /// Self-focusing parameter
    ///
    /// B = k₀·n₂·I₀·r₀²
    /// where k₀ = 2π/λ, I₀ is peak intensity, r₀ is beam radius
    #[must_use]
    pub fn self_focusing_parameter(
        &self,
        wavelength: f64,
        intensity: f64,
        beam_radius: f64,
    ) -> f64 {
        let k0 = 2.0 * std::f64::consts::PI / wavelength;
        k0 * self.n2 * intensity * beam_radius.powi(2)
    }

    /// Is self-focusing significant?
    /// (B > 0.1 typically indicates notable self-focusing)
    #[must_use]
    pub fn is_self_focusing(&self, wavelength: f64, intensity: f64, beam_radius: f64) -> bool {
        self.self_focusing_parameter(wavelength, intensity, beam_radius) > 0.1
    }

    /// Refractive index at given intensity
    #[must_use]
    pub fn refractive_index(&self, intensity: f64) -> f64 {
        self.n2.mul_add(intensity, self.n0)
    }

    /// Nonlinear phase shift
    ///
    /// φ_nl = (2π/λ) * n₂ * I * L
    #[must_use]
    pub fn phase_shift(&self, wavelength: f64, intensity: f64, distance: f64) -> f64 {
        (2.0 * std::f64::consts::PI / wavelength) * self.n2 * intensity * distance
    }

    /// Critical power for self-focusing (approximate)
    ///
    /// P_crit ≈ λ² / (8π·n₀·n₂)
    #[must_use]
    pub fn critical_power(&self, wavelength: f64) -> f64 {
        wavelength.powi(2) / (8.0 * std::f64::consts::PI * self.n0 * self.n2)
    }
}

/// Common materials with Kerr nonlinearity
impl KerrEffect {
    /// Silica glass at 1064 nm
    /// Reference: Boyd (2008)
    #[must_use]
    pub fn silica_glass() -> Self {
        Self::new(1.457, 2.7e-20) // n₂ = 2.7×10⁻²⁰ m²/W
    }

    /// Water at 800 nm
    /// Reference: Agrawal (2007)
    #[must_use]
    pub fn water() -> Self {
        Self::new(1.333, 2.5e-21) // n₂ = 2.5×10⁻²¹ m²/W
    }

    /// CS₂ (carbon disulfide) at 1064 nm
    /// Reference: Boyd (2008) - high nonlinearity
    #[must_use]
    pub fn cs2() -> Self {
        Self::new(1.629, 6.5e-19) // n₂ = 6.5×10⁻¹⁹ m²/W (high!)
    }

    /// Fused silica fiber
    #[must_use]
    pub fn fused_silica() -> Self {
        Self::new(1.46, 2.6e-20)
    }

    /// BK7 optical glass
    #[must_use]
    pub fn bk7_glass() -> Self {
        Self::new(1.517, 2.9e-20)
    }
}

/// Photoacoustic conversion efficiency
///
/// Conversion of optical absorption to acoustic waves
#[derive(Debug, Clone, Copy)]
pub struct PhotoacousticConversion {
    /// Grüneisen parameter (thermal expansion coupling)
    pub gruneisen: f64,
    /// Speed of sound (m/s)
    pub sound_speed: f64,
    /// Thermal conductivity [W/(m·K)]
    pub thermal_conductivity: f64,
    /// Volumetric heat capacity [J/(m³·K)]
    pub volumetric_heat_capacity: f64,
}

impl PhotoacousticConversion {
    /// Create photoacoustic conversion parameters
    #[must_use]
    pub fn new(
        gruneisen: f64,
        sound_speed: f64,
        thermal_conductivity: f64,
        volumetric_heat_capacity: f64,
    ) -> Self {
        Self {
            gruneisen,
            sound_speed,
            thermal_conductivity,
            volumetric_heat_capacity,
        }
    }

    /// Photoacoustic efficiency (dimensionless)
    ///
    /// η_PA = Γ·α·λ = Γ·α·c / ν
    ///
    /// Dimensionless ratio of acoustic wavelength λ = c/ν to optical penetration
    /// depth 1/α, weighted by the Grüneisen parameter Γ. Efficient stress-confined
    /// PA generation requires λ ≫ 1/α.
    ///
    /// References: Xu & Wang (2006), Wang (2009).
    #[must_use]
    pub fn efficiency(&self, optical_absorption: f64, frequency: f64) -> f64 {
        self.gruneisen * optical_absorption * self.sound_speed / frequency
    }

    /// Thermal diffusion length (m)
    ///
    /// l_th = √(D / (π·f))
    /// where D = k / (ρ·c) is thermal diffusivity
    #[must_use]
    pub fn thermal_diffusion_length(&self, frequency: f64) -> f64 {
        let thermal_diffusivity = self.thermal_conductivity / self.volumetric_heat_capacity;
        (thermal_diffusivity / (std::f64::consts::PI * frequency)).sqrt()
    }

    /// Acoustic pressure amplitude
    ///
    /// P_ac = Γ·E_optical / V
    /// where E_optical is absorbed optical energy, V is interaction volume
    #[must_use]
    pub fn acoustic_pressure(&self, absorbed_energy: f64, volume: f64) -> f64 {
        self.gruneisen * absorbed_energy / volume
    }

    // Stress- and thermal-confinement assessment lives in the canonical
    // `physics::photoacoustics::confinement::ConfinementAssessment`, which
    // applies the standard Xu & Wang (2006) conditions
    //   τ_stress  = δ_p / c_s
    //   τ_thermal = δ_p² / (4 D)
    // with the proper pulse-duration and optical-penetration-depth inputs.
    // The previously-defined `is_stress_confined`/`is_thermal_confined`
    // helpers on this struct compared incompatible quantities and produced
    // a dimensionally inconsistent thermal-relaxation time
    // ((ρc_p)² / k² has units s²/m⁴, not seconds), so they were removed.
}

/// Common photoacoustic materials
impl PhotoacousticConversion {
    /// Water
    #[must_use]
    pub fn water() -> Self {
        Self::new(
            0.13,    // Grüneisen parameter
            1480.0,  // Speed of sound [m/s]
            0.6,     // Thermal conductivity [W/(m·K)]
            4.186e6, // Volumetric heat capacity [J/(m³·K)]
        )
    }

    /// Tissue (generic)
    #[must_use]
    pub fn tissue() -> Self {
        Self::new(
            0.25,               // Grüneisen parameter
            SOUND_SPEED_TISSUE, // Speed of sound [m/s]
            0.5,                // Thermal conductivity [W/(m·K)]
            3.5e6,              // Volumetric heat capacity [J/(m³·K)]
        )
    }

    /// Metal (gold)
    #[must_use]
    pub fn gold() -> Self {
        Self::new(
            0.74,   // Grüneisen parameter (high!)
            3240.0, // Speed of sound [m/s]
            315.0,  // Thermal conductivity [W/(m·K)]
            2.5e6,  // Volumetric heat capacity [J/(m³·K)]
        )
    }

    /// Silica
    #[must_use]
    pub fn silica() -> Self {
        Self::new(
            0.27,   // Grüneisen parameter
            5970.0, // Speed of sound [m/s]
            1.38,   // Thermal conductivity [W/(m·K)]
            1.64e6, // Volumetric heat capacity [J/(m³·K)]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kerr_effect_refractive_index() {
        let kerr = KerrEffect::water();
        let n_0 = kerr.refractive_index(0.0);
        let n_high = kerr.refractive_index(1e12); // 1 TW/m²

        assert_eq!(n_0, kerr.n0);
        assert!(n_high > n_0);
    }

    #[test]
    fn test_self_focusing_parameter() {
        let kerr = KerrEffect::cs2(); // High nonlinearity
        let b = kerr.self_focusing_parameter(1064e-9, 1e12, 10e-6);
        assert!(b > 0.0);
    }

    #[test]
    fn test_critical_power() {
        let kerr = KerrEffect::silica_glass();
        let p_crit = kerr.critical_power(1064e-9);
        assert!(p_crit > 0.0);
    }

    #[test]
    fn test_kerr_materials() {
        let silica = KerrEffect::silica_glass();
        let water = KerrEffect::water();
        let cs2 = KerrEffect::cs2();

        // CS2 should have highest nonlinearity
        assert!(cs2.n2 > silica.n2);
        assert!(silica.n2 > water.n2);
    }

    #[test]
    fn test_photoacoustic_efficiency() {
        let pa = PhotoacousticConversion::tissue();
        let efficiency = pa.efficiency(100.0, 1e6); // 100 1/m absorption, 1 MHz
        assert!(efficiency > 0.0);
    }

    #[test]
    fn test_thermal_diffusion_length() {
        let pa = PhotoacousticConversion::tissue();
        let l_th = pa.thermal_diffusion_length(1e6); // 1 MHz
        assert!(l_th > 0.0);
        assert!(l_th < 1e-2); // Should be < 10 μm
    }

    #[test]
    fn test_photoacoustic_materials() {
        let water = PhotoacousticConversion::water();
        let tissue = PhotoacousticConversion::tissue();
        let gold = PhotoacousticConversion::gold();

        // Gold has highest Grüneisen parameter
        assert!(gold.gruneisen > tissue.gruneisen);
        assert!(tissue.gruneisen > water.gruneisen);
    }

    #[test]
    fn test_acoustic_pressure() {
        let pa = PhotoacousticConversion::tissue();
        let pressure = pa.acoustic_pressure(1.0, 1e-15); // 1 J in 1 fm³
        assert!(pressure > 0.0);
        assert!(pressure.is_finite());
    }
}
