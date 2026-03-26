//! Skull property characterization from CT density

use super::planner::TreatmentPlanner;
use super::types::SkullProperties;
use crate::core::error::KwaversResult;
use ndarray::Array3;

impl TreatmentPlanner {
    /// Analyze skull acoustic properties from CT data
    pub(crate) fn analyze_skull_properties(&self) -> KwaversResult<SkullProperties> {
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
                        speed_map[[i, j, k]] = 3000.0 + (hu - 300.0) * 2.0; // m/s
                        density_map[[i, j, k]] = 1800.0 + (hu - 300.0) * 0.5; // kg/m³
                        attenuation_map[[i, j, k]] = 5.0 + (hu - 300.0) * 0.01; // dB/MHz/cm
                    } else if hu > -200.0 {
                        // Tissue
                        speed_map[[i, j, k]] = 1500.0;
                        density_map[[i, j, k]] = 1000.0;
                        attenuation_map[[i, j, k]] = 0.5;
                    } else {
                        // Air
                        speed_map[[i, j, k]] = 340.0;
                        density_map[[i, j, k]] = 1.2;
                        attenuation_map[[i, j, k]] = 0.0;
                    }
                }
            }
        }

        Ok(SkullProperties {
            sound_speed: speed_map,
            density: density_map,
            attenuation: attenuation_map,
        })
    }
}
