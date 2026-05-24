//! Skull property characterization from CT density

use super::planner::TreatmentPlanner;
use super::types::TranscranialSkullProperties;
use crate::core::constants::fundamental::{
    ACOUSTIC_ABSORPTION_TISSUE, DENSITY_AIR, DENSITY_WATER_NOMINAL, SOUND_SPEED_AIR,
    SOUND_SPEED_WATER_SIM,
};
use crate::core::error::KwaversResult;
use ndarray::Array3;

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

                    // Empirical relationships from literature
                    // Reference: Pinton et al. (2012)
                    if hu > 300.0 {
                        // Bone threshold
                        speed_map[[i, j, k]] = (hu - 300.0).mul_add(2.0, 3000.0); // m/s
                        density_map[[i, j, k]] = (hu - 300.0).mul_add(0.5, 1800.0); // kg/m³
                        attenuation_map[[i, j, k]] = (hu - 300.0).mul_add(0.01, 5.0);
                    // dB/MHz/cm
                    } else if hu > -200.0 {
                        // Soft-tissue / water-like baseline (sourced from SSOT)
                        speed_map[[i, j, k]] = SOUND_SPEED_WATER_SIM;
                        density_map[[i, j, k]] = DENSITY_WATER_NOMINAL;
                        attenuation_map[[i, j, k]] = ACOUSTIC_ABSORPTION_TISSUE; // 0.5 dB/(cm·MHz)
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
