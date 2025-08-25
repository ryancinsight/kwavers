use crate::grid::Grid;
use crate::medium::{
    absorption::{tissue_specific, TissueType},
    acoustic::AcousticProperties,
    bubble::{BubbleProperties, BubbleState},
    core::{ArrayAccess, CoreMedium},
    elastic::{ElasticArrayAccess, ElasticProperties},
    optical::OpticalProperties,
    thermal::{TemperatureState, ThermalProperties},
    viscous::ViscousProperties,
};
use log::{debug, info};
use ndarray::{Array3, Axis, Zip};
use std::sync::OnceLock;
use crate::error::{ConfigError, KwaversResult};

/// Configuration for setting tissue in a specific region
#[derive(Debug, Clone)]
pub struct TissueRegion {
    pub tissue_type: TissueType,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub z_min: f64,
    pub z_max: f64,
}

impl TissueRegion {
    /// Create a new tissue region configuration
    pub fn new(
        tissue_type: TissueType,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_min: f64,
        z_max: f64,
    ) -> Self {
        Self {
            tissue_type,
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
        }
    }

    /// Validate the region bounds
    pub fn validate(&self) -> KwaversResult<()> {
        if self.x_min >= self.x_max {
            return Err(ConfigError::InvalidParameter(
                "x_min".to_string(),
                "x_min must be less than x_max".to_string(),
            )
            .into());
        }
        if self.y_min >= self.y_max {
            return Err(ConfigError::InvalidParameter(
                "y_min".to_string(),
                "y_min must be less than y_max".to_string(),
            )
            .into());
        }
        if self.z_min >= self.z_max {
            return Err(ConfigError::InvalidParameter(
                "z_min".to_string(),
                "z_min must be less than z_max".to_string(),
            )
            .into());
        }
        Ok(())
    }
}

/// A heterogeneous medium composed of multiple tissue types
#[derive(Debug, Clone)]
pub struct HeterogeneousTissueMedium {
    /// 3D array of tissue types
    pub tissue_map: Array3<TissueType>,
    /// Temperature distribution (K)
    pub temperature: Array3<f64>,
    /// Bubble radius distribution (m)
    pub bubble_radius: Array3<f64>,
    /// Bubble velocity distribution (m/s)
    pub bubble_velocity: Array3<f64>,
    /// Reference frequency for absorption calculations (Hz)
    pub reference_frequency: f64,
    /// Cached density array for performance
    density_array: OnceLock<Array3<f64>>,
    /// Cached sound speed array for performance
    sound_speed_array: OnceLock<Array3<f64>>,
    /// Optional cached pressure amplitude for nonlinear absorption effects
    pub pressure_amplitude: Option<Array3<f64>>,
    /// Cached shear sound speed array
    shear_sound_speed_array: OnceLock<Array3<f64>>,
    /// Cached shear viscosity coefficient array
    shear_viscosity_coeff_array: OnceLock<Array3<f64>>,
    /// Cached bulk viscosity coefficient array
    bulk_viscosity_coeff_array: OnceLock<Array3<f64>>,
    /// Cached Lamé's first parameter (lambda) array
    lame_lambda_array: OnceLock<Array3<f64>>,
    /// Cached Lamé's second parameter (mu) array
    lame_mu_array: OnceLock<Array3<f64>>,
}

impl HeterogeneousTissueMedium {
    /// Create a new heterogeneous tissue medium with default soft tissue
    pub fn new(grid: &Grid, reference_frequency: f64) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            tissue_map: Array3::from_elem(shape, TissueType::SoftTissue),
            temperature: Array3::from_elem(shape, 310.15), // Body temperature
            bubble_radius: Array3::zeros(shape),
            bubble_velocity: Array3::zeros(shape),
            reference_frequency,
            density_array: OnceLock::new(),
            sound_speed_array: OnceLock::new(),
            pressure_amplitude: None,
            shear_sound_speed_array: OnceLock::new(),
            shear_viscosity_coeff_array: OnceLock::new(),
            bulk_viscosity_coeff_array: OnceLock::new(),
            lame_lambda_array: OnceLock::new(),
            lame_mu_array: OnceLock::new(),
        }
    }

    /// Set tissue type in a specific region
    pub fn set_tissue_region(&mut self, region: &TissueRegion, grid: &Grid) -> KwaversResult<()> {
        region.validate()?;

        let i_min = ((region.x_min / grid.dx).floor() as usize).max(0);
        let i_max = ((region.x_max / grid.dx).ceil() as usize).min(grid.nx - 1);
        let j_min = ((region.y_min / grid.dy).floor() as usize).max(0);
        let j_max = ((region.y_max / grid.dy).ceil() as usize).min(grid.ny - 1);
        let k_min = ((region.z_min / grid.dz).floor() as usize).max(0);
        let k_max = ((region.z_max / grid.dz).ceil() as usize).min(grid.nz - 1);

        for i in i_min..=i_max {
            for j in j_min..=j_max {
                for k in k_min..=k_max {
                    self.tissue_map[[i, j, k]] = region.tissue_type;
                }
            }
        }

        // Clear caches when tissue map changes
        self.clear_caches();
        Ok(())
    }

    /// Clear all cached arrays
    pub fn clear_caches(&mut self) {
        // Take ownership and drop the old values
        self.density_array.take();
        self.sound_speed_array.take();
        self.shear_sound_speed_array.take();
        self.shear_viscosity_coeff_array.take();
        self.bulk_viscosity_coeff_array.take();
        self.lame_lambda_array.take();
        self.lame_mu_array.take();
    }

    /// Get tissue properties at a specific point
    fn get_tissue_properties(&self, x: f64, y: f64, z: f64, grid: &Grid) -> &'static tissue_specific::TissueProperties {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                })
        } else {
            tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap()
        }
    }
}

// Core medium properties
impl CoreMedium for HeterogeneousTissueMedium {
    fn density(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).density
    }

    fn sound_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).sound_speed
    }

    fn is_homogeneous(&self) -> bool {
        false
    }

    fn reference_frequency(&self) -> f64 {
        self.reference_frequency
    }
}

// Array-based access
impl ArrayAccess for HeterogeneousTissueMedium {
    fn density_array(&self) -> &Array3<f64> {
        self.density_array.get_or_init(|| {
            let mut arr = Array3::zeros(self.tissue_map.dim());
            Zip::indexed(&mut arr).for_each(|(i, j, k), val| {
                let tissue = self.tissue_map[[i, j, k]];
                let props = tissue_specific::tissue_database()
                    .get(&tissue)
                    .unwrap_or_else(|| {
                        tissue_specific::tissue_database()
                            .get(&TissueType::SoftTissue)
                            .unwrap()
                    });
                *val = props.density;
            });
            arr
        })
    }

    fn sound_speed_array(&self) -> &Array3<f64> {
        self.sound_speed_array.get_or_init(|| {
            let mut arr = Array3::zeros(self.tissue_map.dim());
            Zip::indexed(&mut arr).for_each(|(i, j, k), val| {
                let tissue = self.tissue_map[[i, j, k]];
                let props = tissue_specific::tissue_database()
                    .get(&tissue)
                    .unwrap_or_else(|| {
                        tissue_specific::tissue_database()
                            .get(&TissueType::SoftTissue)
                            .unwrap()
                    });
                *val = props.sound_speed;
            });
            arr
        })
    }
}

// Acoustic properties
impl AcousticProperties for HeterogeneousTissueMedium {
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let props = self.get_tissue_properties(x, y, z, grid);
        // Power law absorption: α = α₀ * (f/f₀)^y
        props.alpha0 * (frequency / self.reference_frequency).powf(props.y)
    }

    fn nonlinearity_parameter(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).beta
    }

    fn acoustic_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let props = self.get_tissue_properties(x, y, z, grid);
        props.thermal_conductivity / (props.density * props.specific_heat * props.sound_speed)
    }

    fn tissue_type(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Option<TissueType> {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            Some(self.tissue_map[indices])
        } else {
            None
        }
    }
}

// Elastic properties
impl ElasticProperties for HeterogeneousTissueMedium {
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).lame_lambda
    }

    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).lame_mu
    }
}

// Elastic array access
impl ElasticArrayAccess for HeterogeneousTissueMedium {
    fn lame_lambda_array(&self) -> Array3<f64> {
        self.lame_lambda_array
            .get_or_init(|| {
                let mut arr = Array3::zeros(self.tissue_map.dim());
                Zip::indexed(&mut arr).for_each(|(i, j, k), val| {
                    let tissue = self.tissue_map[[i, j, k]];
                    let props = tissue_specific::tissue_database()
                        .get(&tissue)
                        .unwrap_or_else(|| {
                            tissue_specific::tissue_database()
                                .get(&TissueType::SoftTissue)
                                .unwrap()
                        });
                    *val = props.lame_lambda;
                });
                arr
            })
            .clone()
    }

    fn lame_mu_array(&self) -> Array3<f64> {
        self.lame_mu_array
            .get_or_init(|| {
                let mut arr = Array3::zeros(self.tissue_map.dim());
                Zip::indexed(&mut arr).for_each(|(i, j, k), val| {
                    let tissue = self.tissue_map[[i, j, k]];
                    let props = tissue_specific::tissue_database()
                        .get(&tissue)
                        .unwrap_or_else(|| {
                            tissue_specific::tissue_database()
                                .get(&TissueType::SoftTissue)
                                .unwrap()
                        });
                    *val = props.lame_mu;
                });
                arr
            })
            .clone()
    }

    fn shear_sound_speed_array(&self) -> Array3<f64> {
        self.shear_sound_speed_array
            .get_or_init(|| {
                let mut arr = Array3::zeros(self.tissue_map.dim());
                let density_arr = self.density_array();
                Zip::indexed(&mut arr).and(density_arr).for_each(|(i, j, k), val, &rho| {
                    let tissue = self.tissue_map[[i, j, k]];
                    let props = tissue_specific::tissue_database()
                        .get(&tissue)
                        .unwrap_or_else(|| {
                            tissue_specific::tissue_database()
                                .get(&TissueType::SoftTissue)
                                .unwrap()
                        });
                    if rho > 0.0 && props.lame_mu > 0.0 {
                        *val = (props.lame_mu / rho).sqrt();
                    } else {
                        *val = 0.0;
                    }
                });
                arr
            })
            .clone()
    }
}

// Thermal properties
impl ThermalProperties for HeterogeneousTissueMedium {
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).specific_heat
    }

    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).thermal_conductivity
    }

    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let props = self.get_tissue_properties(x, y, z, grid);
        props.thermal_conductivity / (props.density * props.specific_heat)
    }

    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).thermal_expansion
    }
}

// Temperature state management
impl TemperatureState for HeterogeneousTissueMedium {
    fn update_temperature(&mut self, temperature: &Array3<f64>) {
        self.temperature = temperature.clone();
    }

    fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }
}

// Optical properties
impl OpticalProperties for HeterogeneousTissueMedium {
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).mu_a
    }

    fn optical_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).mu_s_prime
    }
}

// Viscous properties
impl ViscousProperties for HeterogeneousTissueMedium {
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).viscosity
    }

    fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // For biological tissues, shear viscosity is typically the same as dynamic viscosity
        self.get_tissue_properties(x, y, z, grid).viscosity
    }

    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Stokes' hypothesis: bulk viscosity = 2.5 * shear viscosity
        2.5 * self.get_tissue_properties(x, y, z, grid).viscosity
    }
}

// Bubble properties
impl BubbleProperties for HeterogeneousTissueMedium {
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).surface_tension
    }

    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        101325.0 // Standard atmospheric pressure
    }

    fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).vapor_pressure
    }

    fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).polytropic_index
    }

    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.get_tissue_properties(x, y, z, grid).gas_diffusion_coeff
    }
}

// Bubble state management
impl BubbleState for HeterogeneousTissueMedium {
    fn bubble_radius(&self) -> &Array3<f64> {
        &self.bubble_radius
    }

    fn bubble_velocity(&self) -> &Array3<f64> {
        &self.bubble_velocity
    }

    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        self.bubble_radius = radius.clone();
        self.bubble_velocity = velocity.clone();
    }
}

// Re-export the type for backward compatibility
pub use HeterogeneousTissueMedium as HeterogeneousTissue;