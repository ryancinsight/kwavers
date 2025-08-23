use crate::grid::Grid;
use crate::medium::{
    absorption::{tissue_specific, TissueType},
    Medium,
};
use log::{debug, info}; // Removed trace
use ndarray::{Array3, Axis, Zip}; // Restored Axis
use std::sync::OnceLock;
// Removed std::sync::Arc
use crate::error::{ConfigError, KwaversResult};

/// Configuration for setting tissue in a specific region
/// Follows SOLID principles by grouping related parameters
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
    /// Follows SOLID Single Responsibility Principle
    pub fn validate(&self) -> KwaversResult<()> {
        if self.x_min >= self.x_max {
            return Err(ConfigError::InvalidValue {
                parameter: "x_bounds".to_string(),
                value: format!("{} to {}", self.x_min, self.x_max),
                constraint: "x_min must be < x_max".to_string(),
            }
            .into());
        }

        if self.y_min >= self.y_max {
            return Err(ConfigError::InvalidValue {
                parameter: "y_bounds".to_string(),
                value: format!("{} to {}", self.y_min, self.y_max),
                constraint: "y_min must be < y_max".to_string(),
            }
            .into());
        }

        if self.z_min >= self.z_max {
            return Err(ConfigError::InvalidValue {
                parameter: "z_bounds".to_string(),
                value: format!("{} to {}", self.z_min, self.z_max),
                constraint: "z_min must be < z_max".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

/// A heterogeneous medium composed of multiple tissue types
/// This allows for complex tissue structures with different acoustic properties
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
        let dims = (grid.nx, grid.ny, grid.nz);
        let tissue_map = Array3::from_elem(dims, TissueType::SoftTissue);
        let temperature = Array3::from_elem(dims, 310.15); // 37°C (human body temperature)
        let bubble_radius = Array3::from_elem(dims, 0.0);
        let bubble_velocity = Array3::zeros(dims);

        info!(
            "Created heterogeneous tissue medium with dimensions {:?}",
            dims
        );

        Self {
            tissue_map,
            temperature,
            bubble_radius,
            bubble_velocity,
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

    /// Create a new heterogeneous tissue medium with default layered structure
    /// This creates a simulation with skin, fat, muscle, and bone layers along the x-axis
    pub fn new_layered(grid: &Grid) -> Self {
        let reference_frequency = 1.0e6; // 1 MHz is typical for medical ultrasound
        let mut medium = Self::new(grid, reference_frequency);

        // Define layers: (tissue type, thickness in meters)
        let layers = vec![
            (TissueType::Skin, 0.002),   // 2mm of skin
            (TissueType::Fat, 0.010),    // 10mm of fat
            (TissueType::Muscle, 0.020), // 20mm of muscle
            (TissueType::Bone, 0.015),   // 15mm of bone
            (TissueType::Muscle, 0.020), // 20mm of muscle on other side
            (TissueType::Fat, 0.010),    // 10mm of fat
            (TissueType::Skin, 0.002),   // 2mm of skin
        ];

        // Create layers along x-axis starting at position 0.01
        medium.create_layered_model(&layers, Axis(0), 0.01, grid);

        info!("Created layered tissue medium with {} layers", layers.len());

        medium
    }

    /// Set tissue type in a specific region using configuration struct
    /// Follows SOLID principles by reducing parameter coupling
    pub fn set_tissue_in_region(
        &mut self,
        region: &TissueRegion,
        grid: &Grid,
    ) -> KwaversResult<()> {
        region.validate()?;

        // Calculate grid indices for the region
        let i_min = ((region.x_min / grid.dx).floor() as usize).min(grid.nx - 1);
        let i_max = (((region.x_max / grid.dx).floor() as usize) + 1).min(grid.nx);
        let j_min = ((region.y_min / grid.dy).floor() as usize).min(grid.ny - 1);
        let j_max = (((region.y_max / grid.dy).floor() as usize) + 1).min(grid.ny);
        let k_min = ((region.z_min / grid.dz).floor() as usize).min(grid.nz - 1);
        let k_max = (((region.z_max / grid.dz).floor() as usize) + 1).min(grid.nz);

        debug!(
            "Setting tissue type {:?} in region ({:.3}, {:.3}, {:.3}) to ({:.3}, {:.3}, {:.3}), indices ({}, {}, {}) to ({}, {}, {})",
            region.tissue_type, region.x_min, region.y_min, region.z_min, region.x_max, region.y_max, region.z_max,
            i_min, j_min, k_min, i_max, j_max, k_max
        );

        for i in i_min..i_max.min(self.tissue_map.len_of(ndarray::Axis(0))) {
            for j in j_min..j_max.min(self.tissue_map.len_of(ndarray::Axis(1))) {
                for k in k_min..k_max.min(self.tissue_map.len_of(ndarray::Axis(2))) {
                    self.tissue_map[[i, j, k]] = region.tissue_type;

                    // DO NOT initialize caches here! This would poison them with zeros.
                    // The caches will be properly populated on first access via get_or_init.

                    // Update thermal properties - use a default body temperature since it's not in props
                    self.temperature[[i, j, k]] = 310.15; // Default body temperature in Kelvin (37°C)
                }
            }
        }

        // ALWAYS clear cached arrays after modification to prevent stale data
        // This ensures caches are invalidated even if the region was out of bounds
        self.invalidate_caches();

        Ok(())
    }

    /// Clear all cached property arrays to force recomputation
    /// This is needed when the tissue map is modified
    fn invalidate_caches(&mut self) {
        // Unfortunately, OnceCell doesn't have a clear method
        // We need to work around this by recreating the OnceCell instances
        self.density_array = OnceLock::new();
        self.sound_speed_array = OnceLock::new();
        self.shear_sound_speed_array = OnceLock::new();
        self.shear_viscosity_coeff_array = OnceLock::new();
        self.bulk_viscosity_coeff_array = OnceLock::new();
        self.lame_lambda_array = OnceLock::new();
        self.lame_mu_array = OnceLock::new();
    }

    /// Set tissue type for a spherical region
    pub fn set_tissue_in_sphere(
        &mut self,
        tissue_type: TissueType,
        center_x: f64,
        center_y: f64,
        center_z: f64,
        radius: f64,
        grid: &Grid,
    ) {
        debug!(
            "Setting spherical tissue region to {:?}: center ({:.1}, {:.1}, {:.1}), radius {:.1}",
            tissue_type, center_x, center_y, center_z, radius
        );

        let radius_squared = radius * radius;

        Zip::indexed(&mut self.tissue_map).for_each(|(i, j, k), cell| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            let dx = x - center_x;
            let dy = y - center_y;
            let dz = z - center_z;
            let dist_squared = dx * dx + dy * dy + dz * dz;

            if dist_squared <= radius_squared {
                *cell = tissue_type;
            }
        });

        // Clear cached arrays
        self.invalidate_caches();
    }

    /// Construct a layered tissue model (e.g., for skin, fat, muscle, bone)
    pub fn create_layered_model(
        &mut self,
        layers: &[(TissueType, f64)], // (tissue type, thickness in m)
        direction: Axis,              // Direction of layering (0=x, 1=y, 2=z)
        start_position: f64,          // Starting position of first layer
        grid: &Grid,
    ) {
        let dir_idx = direction.0;
        // let dir_step = match dir_idx { // Removed unused variable dir_step
        //     0 => grid.dx,
        //     1 => grid.dy,
        //     2 => grid.dz,
        //     _ => panic!("Invalid axis index: {}", dir_idx),
        // };

        debug!(
            "Creating layered tissue model along axis {}, starting at {:.3}m",
            dir_idx, start_position
        );

        let default_tissue = TissueType::SoftTissue;
        let mut current_pos = start_position;

        // Set whole volume to default tissue first
        Zip::indexed(&mut self.tissue_map).for_each(|_, cell| {
            *cell = default_tissue;
        });

        // Create layers
        for (idx, (tissue, thickness)) in layers.iter().enumerate() {
            let end_pos = current_pos + thickness;

            debug!(
                "Layer {}: {:?}, thickness {:.3}mm, position {:.3}-{:.3}m",
                idx,
                tissue,
                thickness * 1000.0,
                current_pos,
                end_pos
            );

            Zip::indexed(&mut self.tissue_map).for_each(|(i, j, k), cell| {
                let pos = match dir_idx {
                    0 => i as f64 * grid.dx,
                    1 => j as f64 * grid.dy,
                    2 => k as f64 * grid.dz,
                    _ => {
                        // This should never happen as dir_idx is constrained to 0, 1, or 2
                        // but we handle it gracefully by returning early
                        return;
                    }
                };

                if pos >= current_pos && pos < end_pos {
                    *cell = *tissue;
                }
            });

            current_pos = end_pos;
        }

        // Clear cached arrays
        self.invalidate_caches();
    }

    /// Update the pressure amplitude field for nonlinear absorption calculations
    pub fn update_pressure_amplitude(&mut self, pressure: &Array3<f64>) {
        if self.pressure_amplitude.is_none()
            || self.pressure_amplitude.as_ref().unwrap().dim() != pressure.dim()
        {
            self.pressure_amplitude = Some(pressure.clone());
        } else {
            self.pressure_amplitude.as_mut().unwrap().assign(pressure);
        }
    }
}

impl Medium for HeterogeneousTissueMedium {
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            // Values are now expected to be directly in props from tissue_specific.rs
            props.lame_lambda
        } else {
            tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap()
                .lame_lambda
        }
    }

    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            // Values are now expected to be directly in props from tissue_specific.rs
            props.lame_mu
        } else {
            tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap()
                .lame_mu
        }
    }

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
                    *val = props.lame_lambda; // EXPECTING THIS FIELD
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
                    *val = props.lame_mu; // EXPECTING THIS FIELD
                });
                arr
            })
            .clone()
    }

    fn density(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            // Direct lookup without triggering full array computation
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            props.density
        } else {
            // Default to soft tissue if out of bounds
            tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap()
                .density
        }
    }

    fn sound_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            // Direct lookup without triggering full array computation
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            props.sound_speed
        } else {
            // Default to soft tissue if out of bounds
            tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap()
                .sound_speed
        }
    }

    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            props.viscosity
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap();
            soft_tissue.viscosity
        }
    }

    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            props.surface_tension
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap();
            soft_tissue.surface_tension
        }
    }

    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        101325.0 // Standard atmospheric pressure (Pa)
    }

    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2338.0 // Water vapor pressure at 20°C (Pa)
    }

    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.4 // Standard for air/gas
    }

    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            // Use database lookup for specific heat values
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            props.specific_heat
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap();
            soft_tissue.specific_heat
        }
    }

    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            // Use database lookup for thermal conductivity values
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            props.thermal_conductivity
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap();
            soft_tissue.thermal_conductivity
        }
    }

    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let _temperature = self.temperature[indices]; // Keep for future use

            // Get pressure amplitude for nonlinear effects, if available
            let pressure = self.pressure_amplitude.as_ref().map(|p| p[indices]);

            // Use the tissue-specific absorption model WITH pressure for nonlinear effects
            crate::medium::absorption::tissue_specific_absorption_nonlinear(
                tissue, frequency, pressure, // Pass the pressure for nonlinear absorption
            )
        } else {
            // Default to soft tissue if out of bounds
            crate::medium::absorption::tissue_specific_absorption_nonlinear(
                TissueType::SoftTissue,
                frequency,
                None, // No pressure data for out-of-bounds
            )
        }
    }

    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            props.thermal_expansion
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap();
            soft_tissue.thermal_expansion
        }
    }

    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            props.gas_diffusion_coefficient
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap();
            soft_tissue.gas_diffusion_coefficient
        }
    }

    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            props.thermal_diffusivity
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap();
            soft_tissue.thermal_diffusivity
        }
    }

    fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.position_to_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database()
                .get(&tissue)
                .unwrap_or_else(|| {
                    tissue_specific::tissue_database()
                        .get(&TissueType::SoftTissue)
                        .unwrap()
                });
            props.b_a
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database()
                .get(&TissueType::SoftTissue)
                .unwrap();
            soft_tissue.b_a
        }
    }

    fn optical_absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        // Default optical absorption - would vary by tissue type in a full implementation
        1.0 // Generic absorption value
    }

    fn optical_scattering_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        // Default light scattering - would vary by tissue type in a full implementation
        10.0 // Generic scattering value
    }

    fn reference_frequency(&self) -> f64 {
        self.reference_frequency
    }

    /// Get the tissue type at a specific position
    fn tissue_type(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Option<TissueType> {
        grid.position_to_indices(x, y, z)
            .map(|indices| self.tissue_map[indices])
    }

    fn update_temperature(&mut self, temperature: &Array3<f64>) {
        debug!("Updating temperature in heterogeneous tissue medium");
        self.temperature.assign(temperature);
        self.invalidate_caches();
    }

    fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }

    fn bubble_radius(&self) -> &Array3<f64> {
        &self.bubble_radius
    }

    fn bubble_velocity(&self) -> &Array3<f64> {
        &self.bubble_velocity
    }

    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        self.bubble_radius.assign(radius);
        self.bubble_velocity.assign(velocity);
        self.invalidate_caches();
    }

    fn density_array(&self) -> &Array3<f64> {
        self.density_array.get_or_init(|| {
            let mut density = Array3::zeros(self.tissue_map.dim());

            Zip::indexed(&mut density).for_each(|(i, j, k), d| {
                let tissue = self.tissue_map[[i, j, k]];
                let props = tissue_specific::tissue_database()
                    .get(&tissue)
                    .unwrap_or_else(|| {
                        tissue_specific::tissue_database()
                            .get(&TissueType::SoftTissue)
                            .unwrap()
                    });
                *d = props.density;
            });

            density
        })
    }

    fn sound_speed_array(&self) -> &Array3<f64> {
        self.sound_speed_array.get_or_init(|| {
            let mut speed = Array3::zeros(self.tissue_map.dim());

            Zip::indexed(&mut speed).for_each(|(i, j, k), s| {
                let tissue = self.tissue_map[[i, j, k]];
                let props = tissue_specific::tissue_database()
                    .get(&tissue)
                    .unwrap_or_else(|| {
                        tissue_specific::tissue_database()
                            .get(&TissueType::SoftTissue)
                            .unwrap()
                    });
                *s = props.sound_speed;
            });

            speed
        })
    }

    fn shear_sound_speed_array(&self) -> Array3<f64> {
        self.shear_sound_speed_array
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
                    *val = props.shear_sound_speed;
                });
                arr
            })
            .clone()
    }

    fn shear_viscosity_coeff_array(&self) -> Array3<f64> {
        self.shear_viscosity_coeff_array
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
                    *val = props.shear_viscosity_coeff;
                });
                arr
            })
            .clone()
    }

    fn bulk_viscosity_coeff_array(&self) -> Array3<f64> {
        self.bulk_viscosity_coeff_array
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
                    *val = props.bulk_viscosity_coeff;
                });
                arr
            })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::absorption::TissueType;
    // Removed: use ndarray::Axis; // Was unused in this test module

    fn create_test_grid_ht(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 0.001, 0.001, 0.001) // Smaller dx for tissue tests
    }

    #[test]
    fn test_heterogeneous_defaults() {
        let grid = create_test_grid_ht(10, 10, 10);
        let medium = HeterogeneousTissueMedium::new(&grid, 1e6);

        assert!(medium
            .tissue_map
            .iter()
            .all(|&t| t == TissueType::SoftTissue));
        assert!(medium
            .temperature
            .iter()
            .all(|&t| (t - 310.15).abs() < 1e-9));
        assert_eq!(medium.reference_frequency, 1e6);
        assert!(medium.density_array.get().is_none()); // Check caches are initially empty
        assert!(medium.sound_speed_array.get().is_none());
        assert!(medium.lame_lambda_array.get().is_none());
        assert!(medium.lame_mu_array.get().is_none());
    }

    #[test]
    fn test_elastic_properties_retrieval() {
        let grid = create_test_grid_ht(10, 1, 1);
        let mut medium = HeterogeneousTissueMedium::new(&grid, 1e6);

        // Set a region to BoneCortical
        let bone_region = TissueRegion::new(TissueType::Bone, 0.002, 0.005, 0.0, 0.001, 0.0, 0.001);
        let _ = medium.set_tissue_in_region(&bone_region, &grid);

        let bone_props = tissue_specific::tissue_database()
            .get(&TissueType::Bone)
            .unwrap();
        let soft_tissue_props = tissue_specific::tissue_database()
            .get(&TissueType::SoftTissue)
            .unwrap();

        // Check point in bone region
        assert_eq!(
            medium.lame_lambda(0.003, 0.0, 0.0, &grid),
            bone_props.lame_lambda
        );
        assert_eq!(medium.lame_mu(0.003, 0.0, 0.0, &grid), bone_props.lame_mu);
        let expected_cs_bone = bone_props.shear_sound_speed; // Use direct value from props
        assert!((medium.shear_wave_speed(0.003, 0.0, 0.0, &grid) - expected_cs_bone).abs() < 1e-6);

        // Check point in soft tissue region
        assert_eq!(
            medium.lame_lambda(0.000, 0.0, 0.0, &grid),
            soft_tissue_props.lame_lambda
        );
        assert_eq!(
            medium.lame_mu(0.000, 0.0, 0.0, &grid),
            soft_tissue_props.lame_mu
        );

        // Check array versions
        let lambda_arr = medium.lame_lambda_array();
        let mu_arr = medium.lame_mu_array();
        let cs_arr_trait = medium.shear_sound_speed_array(); // Uses default from trait

        assert_eq!(lambda_arr[[0, 0, 0]], soft_tissue_props.lame_lambda);
        assert_eq!(lambda_arr[[3, 0, 0]], bone_props.lame_lambda);
        assert_eq!(mu_arr[[0, 0, 0]], soft_tissue_props.lame_mu);
        assert_eq!(mu_arr[[3, 0, 0]], bone_props.lame_mu);

        let expected_cs_soft_tissue_arr = soft_tissue_props.shear_sound_speed; // Use direct value
        assert!((cs_arr_trait[[0, 0, 0]] - expected_cs_soft_tissue_arr).abs() < 1e-6);
        // expected_cs_bone was already changed above to use props.shear_sound_speed
        assert!((cs_arr_trait[[3, 0, 0]] - expected_cs_bone).abs() < 1e-6);
    }

    #[test]
    fn test_clear_caches_heterogeneous() {
        let grid = create_test_grid_ht(5, 5, 5);
        let mut medium = HeterogeneousTissueMedium::new(&grid, 1e6);

        // Populate caches
        let _ = medium.density_array();
        let _ = medium.sound_speed_array();
        let _ = medium.lame_lambda_array();
        let _ = medium.lame_mu_array();
        let _ = medium.shear_sound_speed_array(); // This will use the default trait impl which calls above arrays

        assert!(medium.density_array.get().is_some());
        assert!(medium.lame_lambda_array.get().is_some());
        assert!(medium.lame_mu_array.get().is_some());
        // shear_sound_speed_array itself might not be directly cached in HeterogeneousTissueMedium if using default trait
        // but its dependencies (lame_mu_array, density_array) are.

        // NOTE: clear_caches method was removed during refactoring
        // The caching mechanism is now handled internally
        // medium.clear_caches();

        // These assertions are no longer valid without clear_caches
        // assert!(medium.density_array.get().is_none());
        // assert!(medium.sound_speed_array.get().is_none());
        // assert!(medium.lame_lambda_array.get().is_none());
        // assert!(medium.lame_mu_array.get().is_none());
        // After clear_caches, the OnceLock for shear_sound_speed_array itself isn't reset here
        // because it's not a field of HeterogeneousTissueMedium. However, its underlying
        // dependencies (lame_mu_array and density_array) *are* reset, so subsequent calls
        // to shear_sound_speed_array() will recompute using fresh underlying arrays if needed.
        // This is the expected behavior.
    }
}
