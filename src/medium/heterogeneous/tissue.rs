use crate::grid::Grid;
use crate::medium::{Medium, tissue_specific};
use ndarray::{Array3, Axis, Zip}; // Removed ArrayBase, Dimension, OwnedRepr
use log::{debug, info}; // Removed trace
use std::sync::OnceLock;
// Removed std::sync::Arc
use tissue_specific::TissueType; // Removed TissueProperties

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
}

impl HeterogeneousTissueMedium {
    /// Create a new heterogeneous tissue medium with default soft tissue
    pub fn new(grid: &Grid, reference_frequency: f64) -> Self {
        let dims = (grid.nx, grid.ny, grid.nz);
        let tissue_map = Array3::from_elem(dims, TissueType::SoftTissue);
        let temperature = Array3::from_elem(dims, 310.15); // 37°C (human body temperature)
        let bubble_radius = Array3::from_elem(dims, 0.0);
        let bubble_velocity = Array3::zeros(dims);

        info!("Created heterogeneous tissue medium with dimensions {:?}", dims);
        
        Self {
            tissue_map,
            temperature,
            bubble_radius,
            bubble_velocity,
            reference_frequency,
            density_array: OnceLock::new(),
            sound_speed_array: OnceLock::new(),
            pressure_amplitude: None,
        }
    }

    /// Create a new heterogeneous tissue medium with default layered structure
    /// This creates a simulation with skin, fat, muscle, and bone layers along the x-axis
    pub fn new_layered(grid: &Grid) -> Self {
        let reference_frequency = 1.0e6; // 1 MHz is typical for medical ultrasound
        let mut medium = Self::new(grid, reference_frequency);
        
        // Define layers: (tissue type, thickness in meters)
        let layers = vec![
            (TissueType::Skin, 0.002),           // 2mm of skin
            (TissueType::Fat, 0.010),            // 10mm of fat
            (TissueType::Muscle, 0.020),         // 20mm of muscle
            (TissueType::Bone, 0.015),           // 15mm of bone
            (TissueType::Muscle, 0.020),         // 20mm of muscle on other side
            (TissueType::Fat, 0.010),            // 10mm of fat
            (TissueType::Skin, 0.002),           // 2mm of skin
        ];
        
        // Create layers along x-axis starting at position 0.01
        medium.create_layered_model(&layers, Axis(0), 0.01, grid);
        
        info!("Created layered tissue medium with {} layers", layers.len());
        
        medium
    }

    /// Set tissue type in a specific region defined by a mask
    pub fn set_tissue_in_region(
        &mut self,
        tissue_type: TissueType,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_min: f64,
        z_max: f64,
        grid: &Grid,
    ) {
        let (i_min, i_max) = (grid.x_idx(x_min), grid.x_idx(x_max) + 1);
        let (j_min, j_max) = (grid.y_idx(y_min), grid.y_idx(y_max) + 1);
        let (k_min, k_max) = (grid.z_idx(z_min), grid.z_idx(z_max) + 1);

        debug!(
            "Setting tissue region to {:?}: ({:.1}, {:.1}, {:.1}) -> ({:.1}, {:.1}, {:.1})",
            tissue_type, x_min, y_min, z_min, x_max, y_max, z_max
        );

        for i in i_min..i_max {
            for j in j_min..j_max {
                for k in k_min..k_max {
                    if i < self.tissue_map.shape()[0] 
                       && j < self.tissue_map.shape()[1] 
                       && k < self.tissue_map.shape()[2] {
                        self.tissue_map[[i, j, k]] = tissue_type;
                    }
                }
            }
        }
        
        // Clear cached arrays
        self.clear_caches();
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
        self.clear_caches();
    }

    /// Construct a layered tissue model (e.g., for skin, fat, muscle, bone)
    pub fn create_layered_model(
        &mut self,
        layers: &[(TissueType, f64)], // (tissue type, thickness in m)
        direction: Axis, // Direction of layering (0=x, 1=y, 2=z)
        start_position: f64, // Starting position of first layer
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
            
            debug!("Layer {}: {:?}, thickness {:.3}mm, position {:.3}-{:.3}m",
                   idx, tissue, thickness * 1000.0, current_pos, end_pos);

            Zip::indexed(&mut self.tissue_map).for_each(|(i, j, k), cell| {
                let pos = match dir_idx {
                    0 => i as f64 * grid.dx,
                    1 => j as f64 * grid.dy,
                    2 => k as f64 * grid.dz,
                    _ => unreachable!(),
                };

                if pos >= current_pos && pos < end_pos {
                    *cell = *tissue;
                }
            });

            current_pos = end_pos;
        }
        
        // Clear cached arrays
        self.clear_caches();
    }

    /// Update the pressure amplitude field for nonlinear absorption calculations
    pub fn update_pressure_amplitude(&mut self, pressure: &Array3<f64>) {
        if self.pressure_amplitude.is_none() || 
           self.pressure_amplitude.as_ref().unwrap().dim() != pressure.dim() {
            self.pressure_amplitude = Some(pressure.clone());
        } else {
            self.pressure_amplitude.as_mut().unwrap().assign(pressure);
        }
    }
    
    /// Clear cached arrays when medium properties change
    fn clear_caches(&mut self) {
        debug!("Clearing tissue medium property caches");
        self.density_array = OnceLock::new();
        self.sound_speed_array = OnceLock::new();
    }
}

impl Medium for HeterogeneousTissueMedium {
    fn density(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.to_grid_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database().get(&tissue).unwrap_or_else(|| {
                tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap()
            });
            props.density
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap();
            soft_tissue.density
        }
    }

    fn sound_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.to_grid_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database().get(&tissue).unwrap_or_else(|| {
                tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap()
            });
            props.sound_speed
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap();
            soft_tissue.sound_speed
        }
    }

    fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        // For simplicity, we use a constant viscosity value for all tissues
        // A more detailed model could interpolate between tissue types
        1.0e-3 // Water-like viscosity as a reasonable approximation
    }

    fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.072 // Water-like surface tension (N/m)
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
        if let Some(indices) = grid.to_grid_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database().get(&tissue).unwrap_or_else(|| {
                tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap()
            });
            props.specific_heat
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap();
            soft_tissue.specific_heat
        }
    }

    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.to_grid_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database().get(&tissue).unwrap_or_else(|| {
                tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap()
            });
            props.thermal_conductivity
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap();
            soft_tissue.thermal_conductivity
        }
    }

    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        if let Some(indices) = grid.to_grid_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let temperature = self.temperature[indices];
            
            // Get pressure amplitude for nonlinear effects, if available
            let pressure = self.pressure_amplitude.as_ref().map(|p| p[indices]);
            
            // Use the tissue-specific absorption model
            tissue_specific::tissue_absorption_coefficient(
                tissue,
                frequency,
                temperature,
                pressure
            )
        } else {
            // Default to soft tissue if out of bounds
            tissue_specific::tissue_absorption_coefficient(
                TissueType::SoftTissue,
                frequency,
                310.15, // Body temperature
                None
            )
        }
    }

    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2.1e-4 // Water-like thermal expansion (1/K)
    }

    fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2.0e-9 // Approximate value for water (m²/s)
    }

    fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.4e-7 // Approximate value for water (m²/s)
    }

    fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if let Some(indices) = grid.to_grid_indices(x, y, z) {
            let tissue = self.tissue_map[indices];
            let props = tissue_specific::tissue_database().get(&tissue).unwrap_or_else(|| {
                tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap()
            });
            props.b_a
        } else {
            // Default to soft tissue if out of bounds
            let soft_tissue = tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap();
            soft_tissue.b_a
        }
    }

    fn absorption_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        // Default light absorption - would vary by tissue type in a full implementation
        1.0 // Generic absorption value
    }

    fn reduced_scattering_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        // Default light scattering - would vary by tissue type in a full implementation
        10.0 // Generic scattering value
    }

    fn reference_frequency(&self) -> f64 {
        self.reference_frequency
    }

    /// Get the tissue type at a specific position
    fn tissue_type(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Option<TissueType> {
        grid.to_grid_indices(x, y, z).map(|indices| self.tissue_map[indices])
    }

    fn update_temperature(&mut self, temperature: &Array3<f64>) {
        debug!("Updating temperature in heterogeneous tissue medium");
        self.temperature.assign(temperature);
        self.clear_caches();
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
        self.clear_caches();
    }

    fn density_array(&self) -> Array3<f64> {
        self.density_array.get_or_init(|| {
            let mut density = Array3::zeros(self.tissue_map.dim());
            
            Zip::indexed(&mut density).for_each(|(i, j, k), d| {
                let tissue = self.tissue_map[[i, j, k]];
                let props = tissue_specific::tissue_database().get(&tissue).unwrap_or_else(|| {
                    tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap()
                });
                *d = props.density;
            });
            
            density
        }).clone()
    }

    fn sound_speed_array(&self) -> Array3<f64> {
        self.sound_speed_array.get_or_init(|| {
            let mut speed = Array3::zeros(self.tissue_map.dim());
            
            Zip::indexed(&mut speed).for_each(|(i, j, k), s| {
                let tissue = self.tissue_map[[i, j, k]];
                let props = tissue_specific::tissue_database().get(&tissue).unwrap_or_else(|| {
                    tissue_specific::tissue_database().get(&TissueType::SoftTissue).unwrap()
                });
                *s = props.sound_speed;
            });
            
            speed
        }).clone()
    }
} 