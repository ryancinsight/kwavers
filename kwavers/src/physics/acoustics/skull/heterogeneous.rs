//! Heterogeneous skull model with spatially varying properties
//!
//! ## Theorem: Bone Volume Fraction (BVF) — CT Mapping
//!
//! For a porous bone-water composite, the bone volume fraction (BVF) at a
//! voxel is linearly proportional to the Hounsfield unit (Marquet et al. 2009):
//! ```text
//!   φ(x) = clamp((HU(x) − HU_water) / (HU_cortical − HU_water), 0, 1)
//!   HU_water     = 0     (free water)
//!   HU_cortical  = 1000  (dense cortical bone)
//! ```
//! φ = 0: pure water/soft tissue; φ = 1: fully mineralised cortical bone.
//! Intermediate values (0 < φ < 1) represent the diploe (cancellous) layer.
//!
//! ## Voigt and Reuss Bounds (acoustic elastic moduli)
//!
//! Given BVF φ, the effective bulk modulus of the bone-water composite lies
//! between the Reuss lower bound and the Voigt upper bound:
//! ```text
//!   K_bone  = ρ_bone  · c_bone²
//!   K_water = ρ_water · c_water²
//!
//!   Voigt (upper): K_V = φ · K_bone  + (1−φ) · K_water
//!   Reuss (lower): 1/K_R = φ/K_bone + (1−φ)/K_water
//! ```
//! The Voigt average assumes the two phases experience equal strain (stiff bound).
//! The Reuss average assumes equal stress (compliant bound).
//!
//! ## Hill Averaging — Best Isotropic Estimate
//!
//! **Theorem** (Hill 1952): The true effective modulus of an isotropic polycrystal
//! lies between the Voigt and Reuss bounds.  The Hill (arithmetic) average
//! ```text
//!   K_Hill = (K_V + K_R) / 2
//! ```
//! minimises the deviation from the exact solution in the least-squares sense.
//!
//! The effective sound speed and density follow as:
//! ```text
//!   ρ_eff = φ · ρ_bone + (1−φ) · ρ_water         [Voigt density — always used]
//!   c_eff = sqrt(K_Hill / ρ_eff)
//! ```
//!
//! ## Skull Layer Classification
//!
//! The skull is a three-layer structure:
//! - **Outer cortical**: φ ≥ 0.75 (HU > 750)
//! - **Diploe (cancellous)**: 0.15 < φ < 0.75 (150 < HU < 750)
//! - **Inner cortical**: φ ≥ 0.75 at the inner surface
//!
//! The current implementation applies isotropic Hill-averaged properties; full
//! anisotropic elastic moduli (stiffness tensor C_ijkl) require a separate elastic
//! wave solver and are deferred to a future audit cycle.
//!
//! ## References
//! - Marquet F et al. (2009). Non-invasive transcranial ultrasound therapy based
//!   on a 3D CT scan. Phys. Med. Biol. 54(9), 2597–2613. DOI:10.1088/0031-9155/54/9/001
//! - Aubry J-F et al. (2003). Experimental demonstration of noninvasive transskull
//!   adaptive focusing. J. Acoust. Soc. Am. 113(1), 84–93. DOI:10.1121/1.1529663
//! - Clement GT, Hynynen K (2002). A non-invasive method for focusing ultrasound
//!   through the human skull. Phys. Med. Biol. 47(8), 1219–1236.
//! - Hill R (1952). The elastic behaviour of a crystalline aggregate.
//!   Proc. Phys. Soc. A 65(5), 349–354. DOI:10.1088/0370-1298/65/5/307
//! - Gassmann F (1951). Über die Elastizität poröser Medien.
//!   Vierteljahrsschr. Naturforsch. Ges. Zürich 96, 1–23.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::physics::acoustics::skull::SkullProperties;
use ndarray::{Array3, Zip};

// ── Physical constants for the skull model ────────────────────────────────────

/// Hounsfield unit of pure water (calibration reference).
pub const HU_WATER: f64 = 0.0;
/// Hounsfield unit of fully mineralised cortical bone.
pub const HU_CORTICAL: f64 = 1000.0;
/// Sound speed in water at 20 °C [m/s].
pub const C_WATER: f64 = 1500.0;
/// Density of water at 20 °C [kg/m³].
pub const RHO_WATER: f64 = 1000.0;
/// Water attenuation at clinical frequencies [Np/m/MHz].
pub const ALPHA_WATER: f64 = 0.002;

/// Skull layer classification derived from bone volume fraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkullLayer {
    /// Free water / soft tissue (φ < 0.15)
    SoftTissue,
    /// Diploe — cancellous (trabecular) bone (0.15 ≤ φ < 0.75)
    Diploe,
    /// Cortical bone (φ ≥ 0.75)
    Cortical,
}

/// Heterogeneous skull model with spatially varying acoustic properties.
///
/// Properties can be initialised from a binary mask, directly from CT data
/// using the legacy CTImageLoader pipeline, or from CT data via the more
/// physically accurate Hill-averaged BVF mixing model (see `from_ct_hill`).
#[derive(Debug, Clone)]
pub struct HeterogeneousSkull {
    /// Sound speed distribution (m/s)
    pub sound_speed: Array3<f64>,
    /// Density distribution (kg/m³)
    pub density: Array3<f64>,
    /// Attenuation coefficient distribution (Np/m/MHz)
    pub attenuation: Array3<f64>,
}

impl HeterogeneousSkull {
    /// Create heterogeneous skull from binary mask and scalar skull properties.
    ///
    /// All voxels with `mask > 0.5` receive the provided skull properties;
    /// remaining voxels are assigned water properties.
    pub fn from_mask(
        _grid: &Grid,
        mask: &Array3<f64>,
        props: &SkullProperties,
    ) -> KwaversResult<Self> {
        use crate::core::constants::thermodynamic::ROOM_TEMPERATURE_C;
        use crate::core::constants::water::WaterProperties;

        let water_c = WaterProperties::sound_speed(ROOM_TEMPERATURE_C);
        let water_rho = WaterProperties::density(ROOM_TEMPERATURE_C);

        let mut sound_speed = Array3::from_elem(mask.dim(), water_c);
        let mut density = Array3::from_elem(mask.dim(), water_rho);
        let mut attenuation = Array3::from_elem(mask.dim(), ALPHA_WATER);

        Zip::from(&mut sound_speed)
            .and(&mut density)
            .and(&mut attenuation)
            .and(mask)
            .for_each(|c, rho, atten, &m| {
                if m > 0.5 {
                    *c = props.sound_speed;
                    *rho = props.density;
                    *atten = props.attenuation_coeff;
                }
            });

        Ok(Self {
            sound_speed,
            density,
            attenuation,
        })
    }

    /// Create heterogeneous skull from CT data using the legacy CTImageLoader
    /// pipeline.  The binary threshold HU > 700 selects bone; attenuation
    /// uses the provided props value for bone voxels.
    pub fn from_ct(ct_data: &Array3<f64>, props: &SkullProperties) -> KwaversResult<Self> {
        use crate::domain::imaging::medical::CTImageLoader;

        let sound_speed = ct_data.mapv(CTImageLoader::hu_to_sound_speed);
        let density = ct_data.mapv(CTImageLoader::hu_to_density);
        let attenuation = ct_data.mapv(|hu| {
            if hu > 700.0 {
                props.attenuation_coeff
            } else {
                ALPHA_WATER
            }
        });

        Ok(Self {
            sound_speed,
            density,
            attenuation,
        })
    }

    /// Create heterogeneous skull from CT data using the Hill-averaged BVF
    /// mixing model (recommended for tFUS simulation).
    ///
    /// ## Algorithm
    ///
    /// 1. Compute BVF φ(HU) = clamp((HU − HU_water) / (HU_cortical − HU_water), 0, 1).
    /// 2. Compute Voigt density: ρ_eff = φ·ρ_bone + (1−φ)·ρ_water.
    /// 3. Compute Hill modulus: K_H = (K_V + K_R) / 2.
    /// 4. Effective speed: c_eff = sqrt(K_H / ρ_eff).
    /// 5. Attenuation: linear BVF interpolation α_eff = φ·α_bone + (1−φ)·α_water.
    ///
    /// ## References
    /// - Marquet et al. (2009) eq. (1)–(3). Phys. Med. Biol. 54(9):2597.
    /// - Hill R (1952). Proc. Phys. Soc. A 65(5):349.
    pub fn from_ct_hill(
        ct_data: &Array3<f64>,
        c_bone: f64,
        rho_bone: f64,
        alpha_bone: f64,
    ) -> KwaversResult<Self> {
        let k_bone = rho_bone * c_bone * c_bone;
        let k_water = RHO_WATER * C_WATER * C_WATER;

        let sound_speed = ct_data.mapv(|hu| {
            let phi = Self::bone_volume_fraction(hu);
            if phi <= 0.0 {
                return C_WATER;
            }
            if phi >= 1.0 {
                return c_bone;
            }
            let rho_eff = phi * rho_bone + (1.0 - phi) * RHO_WATER;
            let k_voigt = phi * k_bone + (1.0 - phi) * k_water;
            // Reuss modulus: 1/K_R = φ/K_bone + (1-φ)/K_water
            let k_reuss = 1.0 / (phi / k_bone + (1.0 - phi) / k_water);
            let k_hill = 0.5 * (k_voigt + k_reuss);
            (k_hill / rho_eff).sqrt()
        });

        let density = ct_data.mapv(|hu| {
            let phi = Self::bone_volume_fraction(hu);
            phi * rho_bone + (1.0 - phi) * RHO_WATER
        });

        let attenuation = ct_data.mapv(|hu| {
            let phi = Self::bone_volume_fraction(hu);
            phi * alpha_bone + (1.0 - phi) * ALPHA_WATER
        });

        Ok(Self {
            sound_speed,
            density,
            attenuation,
        })
    }

    /// Compute bone volume fraction (BVF) from Hounsfield unit.
    ///
    /// ## Formula
    /// ```text
    ///   φ = clamp((HU − HU_water) / (HU_cortical − HU_water), 0, 1)
    ///     = clamp(HU / 1000, 0, 1)
    /// ```
    /// φ = 0 for water/soft tissue; φ = 1 for fully mineralised cortical bone.
    ///
    /// ## Reference
    /// Marquet et al. (2009). Phys. Med. Biol. 54(9), eq. (1).
    #[inline]
    pub fn bone_volume_fraction(hu: f64) -> f64 {
        ((hu - HU_WATER) / (HU_CORTICAL - HU_WATER)).clamp(0.0, 1.0)
    }

    /// Classify a voxel into a skull layer based on its BVF.
    ///
    /// Thresholds follow Aubry et al. (2003) and Marquet et al. (2009):
    /// - φ < 0.15 → soft tissue / water
    /// - 0.15 ≤ φ < 0.75 → diploe (cancellous bone)
    /// - φ ≥ 0.75 → cortical bone
    #[inline]
    pub fn classify_layer(hu: f64) -> SkullLayer {
        let phi = Self::bone_volume_fraction(hu);
        if phi < 0.15 {
            SkullLayer::SoftTissue
        } else if phi < 0.75 {
            SkullLayer::Diploe
        } else {
            SkullLayer::Cortical
        }
    }

    /// Generate a binary mask from CT data (1.0 = bone, 0.0 = tissue).
    pub fn generate_mask_from_ct(ct_data: &Array3<f64>) -> Array3<f64> {
        ct_data.mapv(|hu| if hu > 700.0 { 1.0 } else { 0.0 })
    }

    /// Get acoustic impedance at position [Pa·s/m = Rayl]
    pub fn impedance_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.density[[i, j, k]] * self.sound_speed[[i, j, k]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    // ── BVF formula tests ──────────────────────────────────────────────────────

    #[test]
    fn test_bvf_water_is_zero() {
        assert_eq!(HeterogeneousSkull::bone_volume_fraction(0.0), 0.0);
    }

    #[test]
    fn test_bvf_cortical_is_one() {
        assert!((HeterogeneousSkull::bone_volume_fraction(1000.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_bvf_diploe_midpoint() {
        let phi = HeterogeneousSkull::bone_volume_fraction(500.0);
        assert!(
            (phi - 0.5).abs() < 1e-12,
            "BVF at HU=500 should be 0.5; got {phi:.6}"
        );
    }

    #[test]
    fn test_bvf_negative_hu_clamped_to_zero() {
        assert_eq!(HeterogeneousSkull::bone_volume_fraction(-100.0), 0.0);
    }

    #[test]
    fn test_bvf_high_hu_clamped_to_one() {
        assert!((HeterogeneousSkull::bone_volume_fraction(2000.0) - 1.0).abs() < 1e-12);
    }

    // ── Layer classification tests ─────────────────────────────────────────────

    #[test]
    fn test_classify_water_is_soft_tissue() {
        assert_eq!(
            HeterogeneousSkull::classify_layer(0.0),
            SkullLayer::SoftTissue
        );
    }

    #[test]
    fn test_classify_diploe() {
        assert_eq!(
            HeterogeneousSkull::classify_layer(400.0),
            SkullLayer::Diploe
        );
    }

    #[test]
    fn test_classify_cortical() {
        assert_eq!(
            HeterogeneousSkull::classify_layer(900.0),
            SkullLayer::Cortical
        );
    }

    // ── Hill mixing model tests ────────────────────────────────────────────────

    /// In pure water (φ=0): c_eff must equal C_WATER.
    #[test]
    fn test_hill_water_limit_gives_c_water() {
        let ct = Array3::from_elem((4, 4, 4), 0.0_f64);
        let skull = HeterogeneousSkull::from_ct_hill(&ct, 3100.0, 2100.0, 20.0).unwrap();
        for &c in skull.sound_speed.iter() {
            assert!(
                (c - C_WATER).abs() < 1.0,
                "water voxel speed {c:.1} should equal C_WATER={C_WATER}"
            );
        }
    }

    /// In pure cortical bone (φ=1): c_eff must equal c_bone.
    #[test]
    fn test_hill_bone_limit_gives_c_bone() {
        let c_bone = 3100.0_f64;
        let ct = Array3::from_elem((4, 4, 4), 1000.0_f64);
        let skull = HeterogeneousSkull::from_ct_hill(&ct, c_bone, 2100.0, 20.0).unwrap();
        for &c in skull.sound_speed.iter() {
            assert!(
                (c - c_bone).abs() < 1.0,
                "bone voxel speed {c:.1} should equal c_bone={c_bone}"
            );
        }
    }

    /// For intermediate φ, Hill speed must lie strictly between C_WATER and c_bone.
    #[test]
    fn test_hill_diploe_speed_between_water_and_bone() {
        let c_bone = 3100.0_f64;
        let ct = Array3::from_elem((4, 4, 4), 500.0_f64);
        let skull = HeterogeneousSkull::from_ct_hill(&ct, c_bone, 2100.0, 20.0).unwrap();
        for &c in skull.sound_speed.iter() {
            assert!(
                c > C_WATER && c < c_bone,
                "diploe speed {c:.1} must be strictly between {C_WATER} and {c_bone}"
            );
        }
    }

    /// Density must follow Voigt rule: ρ_eff = φ·ρ_bone + (1−φ)·ρ_water.
    #[test]
    fn test_hill_density_voigt_rule() {
        let rho_bone = 2100.0_f64;
        let phi = 0.5_f64;
        let hu = phi * HU_CORTICAL;
        let ct = Array3::from_elem((2, 2, 2), hu);
        let skull = HeterogeneousSkull::from_ct_hill(&ct, 3100.0, rho_bone, 20.0).unwrap();
        let expected_rho = phi * rho_bone + (1.0 - phi) * RHO_WATER;
        for &rho in skull.density.iter() {
            assert!(
                (rho - expected_rho).abs() < 1e-6,
                "density {rho:.3} != Voigt {expected_rho:.3}"
            );
        }
    }

    /// Attenuation must be a linear BVF interpolation.
    #[test]
    fn test_hill_attenuation_linear_interpolation() {
        let alpha_bone = 20.0_f64;
        let phi = 0.6_f64;
        let hu = phi * HU_CORTICAL;
        let ct = Array3::from_elem((2, 2, 2), hu);
        let skull = HeterogeneousSkull::from_ct_hill(&ct, 3100.0, 2100.0, alpha_bone).unwrap();
        let expected = phi * alpha_bone + (1.0 - phi) * ALPHA_WATER;
        for &a in skull.attenuation.iter() {
            assert!(
                (a - expected).abs() < 1e-9,
                "attenuation {a:.6} != linear BVF {expected:.6}"
            );
        }
    }

    /// Hill speed must not exceed the Voigt-modulus speed for any 0 < φ < 1.
    ///
    /// **Theorem** (Hill 1952): K_Hill = (K_V + K_R)/2 ≤ K_V since K_R ≤ K_V.
    /// With common Voigt density ρ_eff, this implies c_Hill = sqrt(K_H/ρ) ≤ sqrt(K_V/ρ).
    ///
    /// Note: the Voigt-modulus speed sqrt(K_V/ρ) is NOT the same as the linear
    /// speed average φ·c_bone + (1−φ)·c_water; both are upper bounds on c_Hill.
    #[test]
    fn test_hill_speed_does_not_exceed_voigt_modulus_speed() {
        let c_bone = 3100.0_f64;
        let rho_bone = 2100.0_f64;
        let k_bone = rho_bone * c_bone * c_bone;
        let k_water = RHO_WATER * C_WATER * C_WATER;
        for hu_int in 1_u32..10 {
            let hu = hu_int as f64 * 100.0;
            let phi = HeterogeneousSkull::bone_volume_fraction(hu);
            let rho_eff = phi * rho_bone + (1.0 - phi) * RHO_WATER;
            let k_voigt = phi * k_bone + (1.0 - phi) * k_water;
            let voigt_modulus_speed = (k_voigt / rho_eff).sqrt();
            let ct = Array3::from_elem((1, 1, 1), hu);
            let skull = HeterogeneousSkull::from_ct_hill(&ct, c_bone, rho_bone, 20.0).unwrap();
            let hill_speed = skull.sound_speed[[0, 0, 0]];
            assert!(
                hill_speed <= voigt_modulus_speed + 1e-6,
                "Hill speed {hill_speed:.2} exceeds Voigt-modulus speed {voigt_modulus_speed:.2} at HU={hu}"
            );
        }
    }
}
