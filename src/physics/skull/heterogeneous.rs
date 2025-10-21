//! Heterogeneous skull model with spatially varying properties
//!
//! Reference: Marquet et al. (2009) "Non-invasive transcranial ultrasound
//! therapy based on a 3D CT scan"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::skull::SkullProperties;
use ndarray::Array3;

/// Heterogeneous skull model with spatially varying acoustic properties
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
        grid: &Grid,
        mask: &Array3<f64>,
        props: &SkullProperties,
    ) -> KwaversResult<Self> {
        let water_c = 1500.0;
        let water_rho = 1000.0;
        let water_atten = 0.002; // Very low attenuation in water

        let mut sound_speed = Array3::from_elem(mask.dim(), water_c);
        let mut density = Array3::from_elem(mask.dim(), water_rho);
        let mut attenuation = Array3::from_elem(mask.dim(), water_atten);

        // Apply skull properties where mask is 1
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if mask[[i, j, k]] > 0.5 {
                        sound_speed[[i, j, k]] = props.sound_speed;
                        density[[i, j, k]] = props.density;
                        attenuation[[i, j, k]] = props.attenuation_coeff;
                    }
                }
            }
        }

        Ok(Self {
            sound_speed,
            density,
            attenuation,
        })
    }

    /// Get acoustic impedance at position
    pub fn impedance_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.density[[i, j, k]] * self.sound_speed[[i, j, k]]
    }
}
