//! CT-based skull model loader
//!
//! Reference: Marquet et al. (2009) "Non-invasive transcranial ultrasound
//! therapy based on a 3D CT scan"

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::physics::skull::HeterogeneousSkull;
use ndarray::Array3;

/// CT-based skull model
#[derive(Debug)]
pub struct CTBasedSkullModel {
    /// Hounsfield unit values from CT
    hounsfield: Array3<f64>,
}

impl CTBasedSkullModel {
    /// Load from NIFTI file
    pub fn from_file(path: &str) -> KwaversResult<Self> {
        // Placeholder - would use nifti crate in production
        Err(KwaversError::InvalidInput(format!(
            "CT loading not yet implemented for path: {}",
            path
        )))
    }

    /// Create from CT data array
    pub fn from_ct_data(ct_data: &Array3<f64>) -> KwaversResult<Self> {
        Ok(Self {
            hounsfield: ct_data.clone(),
        })
    }

    /// Generate binary skull mask from CT
    pub fn generate_mask(&self, grid: &Grid) -> KwaversResult<Array3<f64>> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Threshold HU values: bone typically > 700 HU
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if self.hounsfield[[i, j, k]] > 700.0 {
                        mask[[i, j, k]] = 1.0;
                    }
                }
            }
        }

        Ok(mask)
    }

    /// Convert to heterogeneous model
    pub fn to_heterogeneous(&self, grid: &Grid) -> KwaversResult<HeterogeneousSkull> {
        // Convert HU to acoustic properties using empirical relations
        // Reference: Aubry et al. (2003)
        let mut sound_speed = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut density = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut attenuation = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let hu = self.hounsfield[[i, j, k]];

                    if hu > 700.0 {
                        // Bone
                        sound_speed[[i, j, k]] = 2800.0 + (hu - 700.0) * 0.5;
                        density[[i, j, k]] = 1700.0 + (hu - 700.0) * 0.2;
                        attenuation[[i, j, k]] = 40.0 + (hu - 700.0) * 0.05;
                    } else {
                        // Soft tissue/water
                        sound_speed[[i, j, k]] = 1500.0;
                        density[[i, j, k]] = 1000.0;
                        attenuation[[i, j, k]] = 0.002;
                    }
                }
            }
        }

        Ok(HeterogeneousSkull {
            sound_speed,
            density,
            attenuation,
        })
    }

    /// Get CT data
    pub fn ct_data(&self) -> &Array3<f64> {
        &self.hounsfield
    }

    /// Get sound speed at specific voxel
    pub fn sound_speed(&self, i: usize, j: usize, k: usize) -> f64 {
        let hu = self.hounsfield[[i, j, k]];
        if hu > 700.0 {
            // Bone: empirical relation from Aubry et al. (2003)
            2800.0 + (hu - 700.0) * 2.0
        } else {
            // Soft tissue
            1500.0
        }
    }
}
