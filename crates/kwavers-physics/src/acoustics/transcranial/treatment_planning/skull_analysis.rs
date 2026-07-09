//! Skull property characterization from CT density

use super::planner::TreatmentPlanner;
use super::types::TranscranialSkullProperties;
use kwavers_core::constants::ct_acoustics::{
    HU_BONE_THRESHOLD, PINTON_SKULL_ALPHA_BASE_DB_CM_MHZ,
    PINTON_SKULL_ALPHA_SLOPE_DB_CM_MHZ_PER_HU, PINTON_SKULL_DENSITY_BASE_KG_M3,
    PINTON_SKULL_DENSITY_SLOPE_KG_M3_PER_HU, PINTON_SKULL_SPEED_BASE_M_S,
    PINTON_SKULL_SPEED_SLOPE_M_S_PER_HU,
};
use kwavers_core::constants::fundamental::{
    ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL, SOUND_SPEED_AIR, SOUND_SPEED_WATER_SIM,
};
use kwavers_core::constants::tissue_acoustics::DENSITY_AIR;
use kwavers_core::error::KwaversResult;
use leto::Array3;

// ── Pinton et al. (2012) empirical skull CT model ─────────────────────────────
//
// Reference: Pinton G et al. (2012). "Attenuation, scattering, and absorption
// of ultrasound in the skull bone." *Med. Phys.* 39(1), 299–307.
// DOI: 10.1118/1.3668316.
//
// The model interpolates acoustic properties linearly above HU_BONE_THRESHOLD:
//   c    = PINTON_SKULL_SPEED_BASE   + PINTON_SKULL_SPEED_SLOPE   × (HU − HU_BONE_THRESHOLD)
//   ρ    = PINTON_SKULL_DENSITY_BASE + PINTON_SKULL_DENSITY_SLOPE × (HU − HU_BONE_THRESHOLD)
//   α    = PINTON_SKULL_ALPHA_BASE   + PINTON_SKULL_ALPHA_SLOPE   × (HU − HU_BONE_THRESHOLD)

impl TreatmentPlanner {
    /// Analyze skull acoustic properties from CT data
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn analyze_skull_properties(&self) -> KwaversResult<TranscranialSkullProperties> {
        let (nx, ny, nz) = self.skull_ct.dim();

        // Convert Hounsfield units to acoustic properties
        let mut speed_map = Array3::zeros((nx, ny, nz));
        let mut density_map = Array3::zeros((nx, ny, nz));
        let mut attenuation_map = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let hu = self.skull_ct[[i, j, k]];

                    // Empirical relationships: Pinton et al. (2012) linear CT model.
                    if hu > HU_BONE_THRESHOLD {
                        let delta_hu = hu - HU_BONE_THRESHOLD;
                        speed_map[[i, j, k]] = PINTON_SKULL_SPEED_SLOPE_M_S_PER_HU
                            .mul_add(delta_hu, PINTON_SKULL_SPEED_BASE_M_S); // m/s
                        density_map[[i, j, k]] = PINTON_SKULL_DENSITY_SLOPE_KG_M3_PER_HU
                            .mul_add(delta_hu, PINTON_SKULL_DENSITY_BASE_KG_M3); // kg/m³
                        attenuation_map[[i, j, k]] = PINTON_SKULL_ALPHA_SLOPE_DB_CM_MHZ_PER_HU
                            .mul_add(delta_hu, PINTON_SKULL_ALPHA_BASE_DB_CM_MHZ);
                    // dB/(cm·MHz)
                    } else if hu > -200.0 {
                        // Soft-tissue / water-like baseline (sourced from SSOT)
                        speed_map[[i, j, k]] = SOUND_SPEED_WATER_SIM;
                        density_map[[i, j, k]] = DENSITY_WATER_NOMINAL;
                        attenuation_map[[i, j, k]] = ACOUSTIC_ABSORPTION_TISSUE;
                    // 0.5 dB/(cm·MHz)
                    } else {
                        // Air (sourced from SSOT)
                        speed_map[[i, j, k]] = SOUND_SPEED_AIR;
                        density_map[[i, j, k]] = DENSITY_AIR;
                        attenuation_map[[i, j, k]] = 0.0;
                    }
                }
            }
        }

        Ok(TranscranialSkullProperties {
            sound_speed: speed_map,
            density: density_map,
            attenuation: attenuation_map,
        })
    }
}
