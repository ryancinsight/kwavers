//! Continuous CT Hounsfield-Unit → acoustic-property mapping for tissue-varying
//! simulation media.
//!
//! A clinical CT volume must be turned into the three spatially-varying fields a
//! wave solver consumes — mass density `ρ`, sound speed `c`, and power-law
//! absorption prefactor `α₀`. A *binary* bone/not-bone threshold collapses every
//! soft tissue to one density and erases fat/muscle/liver/marrow contrast; a full
//! patient simulation needs a **continuous** map that resolves the whole HU axis.
//!
//! This model operates in **standard HU** (water = 0, air ≈ −1000, cortical bone
//! ≈ +1000…+2000) — the convention CT scanners report and the one Schneider,
//! Aubry, Marsac, and Connor use. It is the standard-HU counterpart to
//! [`super::hounsfield::HounsfieldUnits`], which mirrors k-wave's CT-number
//! convention (water ≈ 1000) for bit-level k-wave parity; the two are *not*
//! interchangeable input scalings.
//!
//! ## Calibration is scanner-dependent
//! Webb et al. (2018) showed the HU→velocity slope in skull bone ranges from
//! ~0.37 to ~1.8 m·s⁻¹·HU⁻¹ depending on photon energy and reconstruction kernel
//! (best R² ≈ 0.53), so **no single mapping is universal**. The coefficients are
//! therefore public fields: [`HuAcousticModel::default`] is the Schneider (1996)
//! whole-range linear fit — which matches Webb's 120-kVp bone-kernel measurement
//! (0.75 m·s⁻¹·HU⁻¹) — and a caller may override any coefficient for a specific
//! scanner calibration.
//!
//! ## References
//! - Schneider U, Pedroni E, Lomax A (1996). *Phys. Med. Biol.* 41(1), 111–124.
//! - Aubry J-F et al. (2003). *J. Acoust. Soc. Am.* 113(1), 84–93.
//! - Connor CW, Hynynen K (2002). *Phys. Med. Biol.* 47(12), 2213–2231.
//! - Webb TD et al. (2018). *IEEE Trans. Ultrason. Ferroelectr. Freq. Control* 65(7), 1111–1124.

use super::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_AIR, SOUND_SPEED_WATER_SIM};
use super::tissue_acoustics::{
    ACOUSTIC_ABSORPTION_BRAIN, ACOUSTIC_ABSORPTION_SKULL_MIN, B_OVER_A_BONE, B_OVER_A_SOFT_TISSUE,
    DENSITY_AIR, SOFT_TISSUE_ABSORPTION_POWER_Y,
};

/// Continuous, calibration-parameterized HU → {ρ, c, α₀} map (standard HU).
///
/// All fields are public so a caller can substitute a scanner-specific
/// calibration; [`Default`] is the Schneider (1996) / Aubry (2003) reference fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HuAcousticModel {
    /// Density at HU = 0 (water) [kg·m⁻³].
    pub density_water: f64,
    /// Density slope dρ/dHU [kg·m⁻³·HU⁻¹].
    pub density_per_hu: f64,
    /// Lower density clamp (air) [kg·m⁻³] — guards extreme-negative HU voxels.
    pub density_floor: f64,
    /// Sound speed at HU = 0 (water) [m·s⁻¹].
    pub sound_speed_water: f64,
    /// Sound-speed slope below water (HU < 0) [m·s⁻¹·HU⁻¹].
    pub sound_speed_per_hu_soft: f64,
    /// Sound-speed slope at/above water (HU ≥ 0) [m·s⁻¹·HU⁻¹].
    pub sound_speed_per_hu_bone: f64,
    /// Lower sound-speed clamp (air) [m·s⁻¹].
    pub sound_speed_floor: f64,
    /// HU at which a voxel is fully cortical bone (bone-fraction denominator).
    pub cortical_hu: f64,
    /// Soft-tissue absorption prefactor α₀ [dB·cm⁻¹·MHz⁻ʸ].
    pub absorption_soft: f64,
    /// Cortical-bone absorption prefactor α₀ [dB·cm⁻¹·MHz⁻ʸ].
    pub absorption_bone: f64,
    /// Soft-tissue power-law absorption exponent y (α ∝ fʸ).
    pub exponent_soft: f64,
    /// Cortical-bone power-law absorption exponent y.
    pub exponent_bone: f64,
    /// Soft-tissue acoustic nonlinearity parameter B/A.
    pub bovera_soft: f64,
    /// Cortical-bone acoustic nonlinearity parameter B/A.
    pub bovera_bone: f64,
}

impl Default for HuAcousticModel {
    /// Schneider (1996) whole-range linear density and sound speed, with
    /// Aubry/Connor soft↔cortical absorption blend and air floors.
    fn default() -> Self {
        Self {
            density_water: DENSITY_WATER_NOMINAL,           // 1000
            density_per_hu: 0.96,                           // Schneider 1996
            density_floor: DENSITY_AIR,                     // 1.204
            sound_speed_water: SOUND_SPEED_WATER_SIM,       // 1500
            sound_speed_per_hu_soft: 0.50,                  // Schneider, HU < 0
            sound_speed_per_hu_bone: 0.76, // Schneider, HU ≥ 0 (≈ Webb 120 kVp 0.75)
            sound_speed_floor: SOUND_SPEED_AIR, // 343
            cortical_hu: 1000.0,           // HU of fully cortical bone
            absorption_soft: ACOUSTIC_ABSORPTION_BRAIN, // 0.5 dB/cm/MHz
            absorption_bone: ACOUSTIC_ABSORPTION_SKULL_MIN, // 8.0 dB/cm/MHz
            exponent_soft: SOFT_TISSUE_ABSORPTION_POWER_Y, // 1.1 (Duck 1990)
            exponent_bone: 1.0,            // skull α ∝ f¹ (Connor & Hynynen 2002)
            bovera_soft: B_OVER_A_SOFT_TISSUE, // 6.5 (Duck 1990)
            bovera_bone: B_OVER_A_BONE,    // 8.0
        }
    }
}

impl HuAcousticModel {
    /// Mass density [kg·m⁻³] for a standard-HU voxel.
    ///
    /// `ρ(HU) = max(ρ_water + (dρ/dHU)·HU, ρ_floor)` — a continuous, strictly
    /// increasing fit (above the floor) that resolves fat (HU ≈ −100 → ~904),
    /// water (0 → 1000), muscle (≈ 50 → ~1048), and cortical bone (1000 → ~1960).
    #[must_use]
    pub fn density(&self, hu: f64) -> f64 {
        self.density_per_hu
            .mul_add(hu, self.density_water)
            .max(self.density_floor)
    }

    /// Sound speed [m·s⁻¹] for a standard-HU voxel.
    ///
    /// Bilinear Schneider fit: the slope steepens at the water crossing
    /// (soft-tissue slope for HU < 0, bone slope for HU ≥ 0), clamped to the air
    /// floor for deep-negative (gas) voxels. Continuous at HU = 0.
    #[must_use]
    pub fn sound_speed(&self, hu: f64) -> f64 {
        let slope = if hu < 0.0 {
            self.sound_speed_per_hu_soft
        } else {
            self.sound_speed_per_hu_bone
        };
        slope
            .mul_add(hu, self.sound_speed_water)
            .max(self.sound_speed_floor)
    }

    /// Bone volume fraction `φ = clamp(HU / cortical_HU, 0, 1)` — the porosity
    /// proxy that drives the soft↔bone absorption blend (Aubry 2003).
    #[must_use]
    pub fn bone_fraction(&self, hu: f64) -> f64 {
        (hu / self.cortical_hu).clamp(0.0, 1.0)
    }

    /// Power-law absorption prefactor α₀ [dB·cm⁻¹·MHz⁻ʸ], linearly blended from
    /// soft tissue to cortical bone by [`Self::bone_fraction`].
    #[must_use]
    pub fn absorption(&self, hu: f64) -> f64 {
        let phi = self.bone_fraction(hu);
        (1.0 - phi).mul_add(self.absorption_soft, phi * self.absorption_bone)
    }

    /// Power-law absorption exponent y (α ∝ fʸ), blended soft → cortical by
    /// [`Self::bone_fraction`]. Soft tissue ≈ 1.1, skull ≈ 1.0 — both branches
    /// of the CT-derived medium need their own exponent, not a single global y.
    #[must_use]
    pub fn power_law_exponent(&self, hu: f64) -> f64 {
        let phi = self.bone_fraction(hu);
        (1.0 - phi).mul_add(self.exponent_soft, phi * self.exponent_bone)
    }

    /// Acoustic nonlinearity parameter B/A, blended soft → cortical bone by
    /// [`Self::bone_fraction`].
    #[must_use]
    pub fn nonlinearity(&self, hu: f64) -> f64 {
        let phi = self.bone_fraction(hu);
        (1.0 - phi).mul_add(self.bovera_soft, phi * self.bovera_bone)
    }
}

#[cfg(test)]
mod tests {
    use super::HuAcousticModel;
    use crate::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use crate::constants::tissue_acoustics::{DENSITY_AIR, DENSITY_FAT};

    // Water (HU = 0) recovers the nominal water density and sound speed exactly —
    // the calibration anchor both Schneider segments pass through.
    #[test]
    fn water_anchor_is_exact() {
        let m = HuAcousticModel::default();
        assert!((m.density(0.0) - DENSITY_WATER_NOMINAL).abs() < 1e-9);
        assert!((m.sound_speed(0.0) - SOUND_SPEED_WATER_SIM).abs() < 1e-9);
    }

    // Distinct soft tissues map to DISTINCT properties — the whole point of
    // replacing the binary threshold. Fat < water < muscle < cortical bone.
    #[test]
    fn soft_tissues_are_resolved_distinctly() {
        let m = HuAcousticModel::default();
        let (rho_fat, rho_water, rho_muscle, rho_bone) = (
            m.density(-100.0),
            m.density(0.0),
            m.density(50.0),
            m.density(1000.0),
        );
        assert!(
            rho_fat < rho_water && rho_water < rho_muscle && rho_muscle < rho_bone,
            "densities not strictly ordered: fat={rho_fat} water={rho_water} muscle={rho_muscle} bone={rho_bone}"
        );
        // Fat at HU = −100 lands near the tabulated adipose density (928 kg/m³).
        assert!(
            (rho_fat - DENSITY_FAT).abs() < 30.0,
            "fat density {rho_fat} far from tabulated {DENSITY_FAT}"
        );
        // Cortical bone at HU = 1000 lands in the physiological band.
        assert!(
            (1900.0..=2000.0).contains(&rho_bone),
            "bone density {rho_bone} kg/m³"
        );
    }

    // Sound speed is continuous at the HU = 0 slope change (both Schneider
    // segments evaluate to c_water there).
    #[test]
    fn sound_speed_continuous_at_water_crossing() {
        let m = HuAcousticModel::default();
        let below = m.sound_speed(-1e-6);
        let above = m.sound_speed(1e-6);
        assert!(
            (below - above).abs() < 1e-3,
            "discontinuity at HU=0: {below} vs {above}"
        );
        // Bone is faster than soft tissue.
        assert!(m.sound_speed(1000.0) > m.sound_speed(0.0));
    }

    // Air floors clamp deep-negative (gas) voxels to physical air values rather
    // than the unphysical negative density / sub-300 m/s the raw line would give.
    #[test]
    fn deep_negative_hu_clamps_to_air_floor() {
        let m = HuAcousticModel::default();
        assert!((m.density(-5000.0) - DENSITY_AIR).abs() < 1e-9);
        assert_eq!(m.sound_speed(-5000.0), m.sound_speed_floor);
    }

    // Absorption blends monotonically from soft tissue (φ=0) to cortical bone
    // (φ=1) and the endpoints equal the calibrated prefactors exactly.
    #[test]
    fn absorption_blends_soft_to_bone() {
        let m = HuAcousticModel::default();
        assert!((m.absorption(0.0) - m.absorption_soft).abs() < 1e-12);
        assert!((m.absorption(1000.0) - m.absorption_bone).abs() < 1e-12);
        let mid = m.absorption(500.0);
        assert!(
            (mid - 0.5 * (m.absorption_soft + m.absorption_bone)).abs() < 1e-9,
            "midpoint absorption {mid} not the soft/bone average"
        );
        assert!(m.bone_fraction(-200.0) == 0.0 && m.bone_fraction(3000.0) == 1.0);
    }

    // Power-law exponent and B/A blend monotonically soft → bone, hitting the
    // calibrated endpoints exactly: a complete CT-derived medium needs per-voxel
    // y and B/A, not just ρ, c, α₀.
    #[test]
    fn exponent_and_nonlinearity_blend_soft_to_bone() {
        let m = HuAcousticModel::default();
        assert!((m.power_law_exponent(0.0) - m.exponent_soft).abs() < 1e-12);
        assert!((m.power_law_exponent(1000.0) - m.exponent_bone).abs() < 1e-12);
        assert!((m.nonlinearity(0.0) - m.bovera_soft).abs() < 1e-12);
        assert!((m.nonlinearity(1000.0) - m.bovera_bone).abs() < 1e-12);
        // Soft tissue (1.1) decreases toward skull (1.0) as bone fraction rises.
        assert!(m.power_law_exponent(500.0) < m.exponent_soft);
        assert!(m.power_law_exponent(500.0) > m.exponent_bone);
        // B/A rises soft → bone.
        assert!(m.nonlinearity(500.0) > m.bovera_soft && m.nonlinearity(500.0) < m.bovera_bone);
    }
}
