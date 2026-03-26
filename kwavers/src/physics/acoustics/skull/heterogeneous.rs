//! Heterogeneous skull model with spatially varying properties
//!
//! # Theorem: Heterogeneous Acoustic Medium (Biot-Gassmann Framework)
//!
//! For a porous bone-water composite, the spatially-varying acoustic properties
//! can be estimated from the bone volume fraction (BVF) φ(x), which is linearly
//! related to the CT Hounsfield unit HU(x):
//! ```text
//!   φ(x) = clamp((HU(x) − HU_water) / (HU_bone − HU_water), 0, 1)
//! ```
//!
//! ## Voigt Mixing Rules (linear interpolation)
//!
//! For the heterogeneous skull, Voigt-Reuss bounds on sound speed and density
//! are approximated by linear mixing:
//! ```text
//!   c(x) = φ(x) · c_bone + (1 − φ(x)) · c_water
//!   ρ(x) = φ(x) · ρ_bone + (1 − φ(x)) · ρ_water
//!   α(x) = φ(x) · α_bone + (1 − φ(x)) · α_water
//! ```
//! This is the first-order Voigt average; higher-order Gassmann corrections
//! accounting for pore geometry and trabecular microstructure require CT-derived
//! porosity maps (not currently implemented; see TODO_AUDIT items).
//!
//! ## Anisotropy Note
//!
//! The skull consists of two cortical plates separated by cancellous (diploe) bone.
//! Compressional wave speed in the cortical layers is approximately:
//! - Longitudinal (perpendicular to plate): c_L ≈ 3100 m/s
//! - Shear: c_T ≈ 1700 m/s
//!
//! The current implementation assumes isotropic properties (scalar c, ρ, α).
//! Full anisotropic elasticity requires a separate elastic wave solver with the
//! stiffness tensor C_ijkl(x) — tracked in TODO_AUDIT items above.
//!
//! ## CT-to-Property Mapping Pipeline
//!
//! 1. Load CT scan → Hounsfield units HU(i,j,k).
//! 2. Compute BVF from HU using linear scaling.
//! 3. Apply Voigt mixing to obtain c(x), ρ(x), α(x) at each grid cell.
//! 4. Optionally smooth to remove CT noise (Gaussian convolution).
//!
//! ## References
//! - Marquet, F., Pernot, M., Aubry, J.-F. et al. (2009). Non-invasive transcranial
//!   ultrasound therapy based on a 3D CT scan: protocol validation and in vitro
//!   results. Phys. Med. Biol. 54(9), 2597–2613.
//! - Clement, G.T. & Hynynen, K. (2002). A non-invasive method for focusing
//!   ultrasound through the human skull. Phys. Med. Biol. 47(8), 1219–1236.
//! - Aubry, J.-F. et al. (2003). J. Acoust. Soc. Am. 113(1), 84–93.
//! - Gassmann, F. (1951). Über die Elastizität poröser Medien.
//!   Vierteljahrsschr. Naturforsch. Ges. Zürich 96, 1–23.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::physics::acoustics::skull::SkullProperties;
use ndarray::{Array3, Zip};

/// Heterogeneous skull model with spatially varying acoustic properties
/// TODO_AUDIT: P1 - Advanced Skull Modeling - Implement full patient-specific skull characterization with anisotropic properties and healing effects
/// DEPENDS ON: physics/acoustics/skull/anisotropic.rs, physics/acoustics/skull/healing.rs, physics/acoustics/skull/microstructure.rs
/// MISSING: Anisotropic acoustic properties accounting for diploe structure (trabecular bone)
/// MISSING: Age-related changes in skull acoustic properties
/// MISSING: Microstructural modeling of Haversian systems and osteons
/// MISSING: Healing and remodeling effects post-trauma
/// MISSING: Temperature-dependent skull properties for thermal therapy
/// THEOREM: Bone anisotropy: c_L > c_T where L is longitudinal, T is transverse wave speed
/// THEOREM: Osteon structure: Acoustic waves guided by cylindrical microstructure
/// REFERENCES: Marquet et al. (2009) Phys Med Biol; Clement et al. (2004) Ultrasound Med Biol
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
    /// Create heterogeneous skull from mask and properties
    pub fn from_mask(
        _grid: &Grid,
        mask: &Array3<f64>,
        props: &SkullProperties,
    ) -> KwaversResult<Self> {
        use crate::core::constants::thermodynamic::ROOM_TEMPERATURE_C;
        use crate::core::constants::water::WaterProperties;

        let water_c = WaterProperties::sound_speed(ROOM_TEMPERATURE_C);
        let water_rho = WaterProperties::density(ROOM_TEMPERATURE_C);
        let water_atten = 0.002; // Very low attenuation in water

        let mut sound_speed = Array3::from_elem(mask.dim(), water_c);
        let mut density = Array3::from_elem(mask.dim(), water_rho);
        let mut attenuation = Array3::from_elem(mask.dim(), water_atten);

        // Apply skull properties where mask is 1
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

    /// Create heterogeneous skull directly from CT Hounsfield Units
    pub fn from_ct(ct_data: &Array3<f64>, props: &SkullProperties) -> KwaversResult<Self> {
        use crate::domain::imaging::medical::CTImageLoader;
        
        let sound_speed = ct_data.mapv(CTImageLoader::hu_to_sound_speed);
        let density = ct_data.mapv(CTImageLoader::hu_to_density);
        
        // Attenuation relies on bone threshold (HU > 700)
        let water_atten = 0.002;
        let attenuation = ct_data.mapv(|hu| {
            if hu > 700.0 {
                props.attenuation_coeff
            } else {
                water_atten
            }
        });

        Ok(Self {
            sound_speed,
            density,
            attenuation,
        })
    }
    
    /// Generate a binary mask from CT data (1.0 = bone, 0.0 = tissue)
    pub fn generate_mask_from_ct(ct_data: &Array3<f64>) -> Array3<f64> {
        ct_data.mapv(|hu| if hu > 700.0 { 1.0 } else { 0.0 })
    }

    /// Get acoustic impedance at position
    pub fn impedance_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.density[[i, j, k]] * self.sound_speed[[i, j, k]]
    }
}

