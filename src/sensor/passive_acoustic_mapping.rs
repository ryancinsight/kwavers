//! Passive Acoustic Mapping (PAM) for cavitation detection and sonoluminescence
//!
//! This module implements passive acoustic mapping techniques for detecting and
//! mapping cavitation fields and sonoluminescence events using arbitrary sensor
//! array geometries.
//!
//! ## Literature References
//!
//! 1. **Gyöngy & Coussios (2010)**: "Passive spatial mapping of inertial cavitation
//!    during HIFU exposure", IEEE Trans. Biomed. Eng.
//! 2. **Haworth et al. (2012)**: "Passive imaging with pulsed ultrasound insonations",
//!    J. Acoust. Soc. Am.
//! 3. **Coviello et al. (2015)**: "Passive acoustic mapping utilizing optimal beamforming
//!    in ultrasound therapy monitoring", J. Acoust. Soc. Am.

use crate::error::{KwaversResult, KwaversError};
use crate::grid::Grid;
use crate::physics::plugin::{PhysicsPlugin, PluginMetadata, PluginContext};
use crate::physics::field_mapping::UnifiedFieldType;
use ndarray::{Array1, Array2, Array3, Array4, Axis};
use std::f64::consts::PI;
use std::any::Any;

/// Array geometry types for different sensor configurations
#[derive(Debug, Clone)]
pub enum ArrayGeometry {
    /// Linear array (1D)
    Linear {
        elements: usize,
        pitch: f64, // Element spacing in meters
        center: [f64; 3],
        orientation: [f64; 3], // Direction vector
    },
    /// Planar array (2D)
    Planar {
        elements_x: usize,
        elements_y: usize,
        pitch_x: f64,
        pitch_y: f64,
        center: [f64; 3],
        normal: [f64; 3], // Normal to plane
    },
    /// Circular/Ring array
    Circular {
        elements: usize,
        radius: f64,
        center: [f64; 3],
        normal: [f64; 3],
    },
    /// Hemispherical bowl array
    Hemispherical {
        rings: usize,
        elements_per_ring: Vec<usize>,
        radius: f64,
        center: [f64; 3],
        focus: [f64; 3],
    },
    /// Phased array with arbitrary element positions
    Phased {
        elements: Vec<[f64; 3]>, // Individual element positions
        aperture: f64, // Effective aperture size
    },
    /// Custom arbitrary geometry
    Custom {
        positions: Vec<[f64; 3]>,
        weights: Vec<f64>, // Element sensitivity weights
        description: String,
    },
}

impl ArrayGeometry {
    /// Get all sensor element positions
    pub fn get_element_positions(&self) -> Vec<[f64; 3]> {
        match self {
            ArrayGeometry::Linear { elements, pitch, center, orientation } => {
                let mut positions = Vec::with_capacity(*elements);
                let norm = (orientation[0].powi(2) + orientation[1].powi(2) + orientation[2].powi(2)).sqrt();
                let dir = [orientation[0]/norm, orientation[1]/norm, orientation[2]/norm];
                
                for i in 0..*elements {
                    let offset = (i as f64 - (*elements as f64 - 1.0) / 2.0) * pitch;
                    positions.push([
                        center[0] + offset * dir[0],
                        center[1] + offset * dir[1],
                        center[2] + offset * dir[2],
                    ]);
                }
                positions
            },
            ArrayGeometry::Planar { elements_x, elements_y, pitch_x, pitch_y, center, normal } => {
                let mut positions = Vec::with_capacity(elements_x * elements_y);
                // Create orthonormal basis
                let n = normalize_vector(*normal);
                let u = get_perpendicular_vector(n);
                let v = cross_product(n, u);
                
                for i in 0..*elements_x {
                    for j in 0..*elements_y {
                        let x_offset = (i as f64 - (*elements_x as f64 - 1.0) / 2.0) * pitch_x;
                        let y_offset = (j as f64 - (*elements_y as f64 - 1.0) / 2.0) * pitch_y;
                        positions.push([
                            center[0] + x_offset * u[0] + y_offset * v[0],
                            center[1] + x_offset * u[1] + y_offset * v[1],
                            center[2] + x_offset * u[2] + y_offset * v[2],
                        ]);
                    }
                }
                positions
            },
            ArrayGeometry::Circular { elements, radius, center, normal } => {
                let mut positions = Vec::with_capacity(*elements);
                let n = normalize_vector(*normal);
                let u = get_perpendicular_vector(n);
                let v = cross_product(n, u);
                
                for i in 0..*elements {
                    let angle = 2.0 * PI * i as f64 / *elements as f64;
                    let x = radius * angle.cos();
                    let y = radius * angle.sin();
                    positions.push([
                        center[0] + x * u[0] + y * v[0],
                        center[1] + x * u[1] + y * v[1],
                        center[2] + x * u[2] + y * v[2],
                    ]);
                }
                positions
            },
            ArrayGeometry::Hemispherical { rings, elements_per_ring, radius, center, focus } => {
                let mut positions = Vec::new();
                
                for ring in 0..*rings {
                    let theta = (ring + 1) as f64 * PI / (2.0 * (*rings + 1) as f64);
                    let ring_radius = radius * theta.sin();
                    let z = radius * theta.cos();
                    
                    let elements_in_ring = elements_per_ring.get(ring).copied().unwrap_or(32);
                    for elem in 0..elements_in_ring {
                        let phi = 2.0 * PI * elem as f64 / elements_in_ring as f64;
                        positions.push([
                            center[0] + ring_radius * phi.cos(),
                            center[1] + ring_radius * phi.sin(),
                            center[2] + z,
                        ]);
                    }
                }
                positions
            },
            ArrayGeometry::Phased { elements, .. } => elements.clone(),
            ArrayGeometry::Custom { positions, .. } => positions.clone(),
        }
    }
    
    /// Get element count
    pub fn element_count(&self) -> usize {
        match self {
            ArrayGeometry::Linear { elements, .. } => *elements,
            ArrayGeometry::Planar { elements_x, elements_y, .. } => elements_x * elements_y,
            ArrayGeometry::Circular { elements, .. } => *elements,
            ArrayGeometry::Hemispherical { elements_per_ring, .. } => elements_per_ring.iter().sum(),
            ArrayGeometry::Phased { elements, .. } => elements.len(),
            ArrayGeometry::Custom { positions, .. } => positions.len(),
        }
    }
}

/// Passive Acoustic Mapping configuration
#[derive(Debug, Clone)]
pub struct PAMConfig {
    /// Sensor array geometry
    pub array_geometry: ArrayGeometry,
    /// Frequency bands for analysis (Hz)
    pub frequency_bands: Vec<(f64, f64)>,
    /// Beamforming algorithm
    pub beamforming: BeamformingMethod,
    /// Time window for integration (seconds)
    pub integration_time: f64,
    /// Enable cavitation detection
    pub detect_cavitation: bool,
    /// Enable sonoluminescence detection
    pub detect_sonoluminescence: bool,
    /// Spatial resolution for reconstruction (meters)
    pub spatial_resolution: f64,
}

/// Beamforming methods for PAM
#[derive(Debug, Clone)]
pub enum BeamformingMethod {
    /// Delay-and-sum beamforming
    DelayAndSum,
    /// Robust Capon beamforming
    RobustCapon { diagonal_loading: f64 },
    /// MUSIC algorithm
    MUSIC { signal_subspace_dim: usize },
    /// Time exposure acoustics
    TimeExposureAcoustics,
    /// Passive cavitation imaging
    PassiveCavitationImaging,
}

/// PAM plugin for real-time cavitation field mapping
#[derive(Debug)]
pub struct PassiveAcousticMappingPlugin {
    metadata: PluginMetadata,
    config: PAMConfig,
    sensor_positions: Vec<[f64; 3]>,
    sensor_indices: Vec<[usize; 3]>,
    recorded_signals: Vec<Vec<f64>>,
    cavitation_map: Array3<f64>,
    sonoluminescence_map: Array3<f64>,
    frequency_domain_data: Vec<Array1<f64>>,
}

impl PassiveAcousticMappingPlugin {
    pub fn new(config: PAMConfig, grid: &Grid) -> KwaversResult<Self> {
        let sensor_positions = config.array_geometry.get_element_positions();
        let mut sensor_indices = Vec::with_capacity(sensor_positions.len());
        
        // Convert physical positions to grid indices
        for pos in &sensor_positions {
            let i = (pos[0] / grid.dx).round() as usize;
            let j = (pos[1] / grid.dy).round() as usize;
            let k = (pos[2] / grid.dz).round() as usize;
            
            if i >= grid.nx || j >= grid.ny || k >= grid.nz {
                return Err(KwaversError::Config(crate::error::ConfigError::InvalidValue {
                    parameter: "sensor_position".to_string(),
                    value: format!("{:?}", pos),
                    constraint: "Sensor position must be within grid bounds".to_string(),
                }));
            }
            
            sensor_indices.push([i, j, k]);
        }
        
        let num_sensors = sensor_positions.len();
        
        Ok(Self {
            metadata: PluginMetadata {
                id: "pam_plugin".to_string(),
                name: "Passive Acoustic Mapping".to_string(),
                version: "1.0.0".to_string(),
                description: "Real-time cavitation and sonoluminescence mapping".to_string(),
                author: "Kwavers Team".to_string(),
                license: "MIT".to_string(),
            },
            config,
            sensor_positions,
            sensor_indices,
            recorded_signals: vec![Vec::new(); num_sensors],
            cavitation_map: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            sonoluminescence_map: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            frequency_domain_data: vec![Array1::zeros(1024); num_sensors],
        })
    }
    
    /// Perform beamforming to reconstruct spatial field
    fn beamform(&self, grid: &Grid, frequency: f64) -> Array3<f64> {
        let mut reconstructed = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        match &self.config.beamforming {
            BeamformingMethod::DelayAndSum => {
                self.delay_and_sum_beamforming(&mut reconstructed, grid, frequency);
            },
            BeamformingMethod::RobustCapon { diagonal_loading } => {
                self.capon_beamforming_with_diagonal_loading(&mut reconstructed, grid, frequency, *diagonal_loading);
            },
            BeamformingMethod::MUSIC { signal_subspace_dim } => {
                self.music_beamforming(&mut reconstructed, grid, frequency, *signal_subspace_dim);
            },
            BeamformingMethod::TimeExposureAcoustics => {
                self.time_exposure_acoustics(&mut reconstructed, grid);
            },
            BeamformingMethod::PassiveCavitationImaging => {
                self.passive_cavitation_imaging(&mut reconstructed, grid, frequency);
            },
        }
        
        reconstructed
    }
    
    /// Delay-and-sum beamforming
    fn delay_and_sum_beamforming(&self, output: &mut Array3<f64>, grid: &Grid, frequency: f64) {
        let c = 1500.0; // Speed of sound in water (m/s)
        let wavelength = c / frequency;
        let k = 2.0 * PI / wavelength;
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k_idx in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k_idx as f64 * grid.dz;
                    
                    let mut sum = 0.0;
                    for (sensor_idx, sensor_pos) in self.sensor_positions.iter().enumerate() {
                        let dx = x - sensor_pos[0];
                        let dy = y - sensor_pos[1];
                        let dz = z - sensor_pos[2];
                        let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                        
                        // Apply time delay based on distance
                        let phase = k * distance;
                        
                        if let Some(signal_value) = self.frequency_domain_data[sensor_idx].get(0) {
                            sum += signal_value * phase.cos();
                        }
                    }
                    
                    output[[i, j, k_idx]] = sum / self.sensor_positions.len() as f64;
                }
            }
        }
    }
    
    /// Capon beamforming with diagonal loading for regularization
    fn capon_beamforming_with_diagonal_loading(&self, output: &mut Array3<f64>, grid: &Grid, frequency: f64, diagonal_loading: f64) {
        // Implementation of Capon beamformer with diagonal loading
        // Reference: Li et al. (2003) "Robust Capon beamforming"
        
        let c = 1500.0;
        let wavelength = c / frequency;
        let k = 2.0 * PI / wavelength;
        
        // Compute covariance matrix with diagonal loading
        let num_sensors = self.sensor_positions.len();
        let mut covariance = Array2::<f64>::zeros((num_sensors, num_sensors));
        
        // Add diagonal loading for robustness
        for i in 0..num_sensors {
            covariance[[i, i]] = diagonal_loading;
        }
        
        // Spatial spectrum computation
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k_idx in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k_idx as f64 * grid.dz;
                    
                    // Compute steering vector
                    let mut steering = Array1::<f64>::zeros(num_sensors);
                    for (sensor_idx, sensor_pos) in self.sensor_positions.iter().enumerate() {
                        let dx = x - sensor_pos[0];
                        let dy = y - sensor_pos[1];
                        let dz = z - sensor_pos[2];
                        let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                        steering[sensor_idx] = (k * distance).cos();
                    }
                    
                    // Capon spatial spectrum
                    // P = 1 / (a^H * R^-1 * a)
                    // For now, use simplified version
                    output[[i, j, k_idx]] = steering.sum() / num_sensors as f64;
                }
            }
        }
    }
    
    /// MUSIC algorithm for high-resolution imaging
    fn music_beamforming(&self, output: &mut Array3<f64>, grid: &Grid, frequency: f64, signal_subspace_dim: usize) {
        // MUSIC (Multiple Signal Classification) algorithm
        // Reference: Schmidt (1986) "Multiple emitter location and signal parameter estimation"
        
        // This is a simplified implementation
        // Full MUSIC requires eigenvalue decomposition of covariance matrix
        self.delay_and_sum_beamforming(output, grid, frequency);
    }
    
    /// Time exposure acoustics for cavitation mapping
    fn time_exposure_acoustics(&self, output: &mut Array3<f64>, grid: &Grid) {
        // Time exposure acoustics (TEA) for passive cavitation mapping
        // Reference: Gyöngy & Coussios (2010)
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    // Integrate acoustic energy over time window
                    let mut energy = 0.0;
                    for sensor_idx in 0..self.sensor_positions.len() {
                        if let Some(signal) = self.recorded_signals.get(sensor_idx) {
                            energy += signal.iter().map(|&s| s * s).sum::<f64>();
                        }
                    }
                    output[[i, j, k]] = energy / (self.sensor_positions.len() as f64);
                }
            }
        }
    }
    
    /// Passive cavitation imaging
    fn passive_cavitation_imaging(&self, output: &mut Array3<f64>, grid: &Grid, frequency: f64) {
        // Passive cavitation imaging based on broadband emissions
        // Reference: Haworth et al. (2012)
        
        // Use delay-and-sum as base, then apply cavitation-specific processing
        self.delay_and_sum_beamforming(output, grid, frequency);
        
        // Apply threshold for cavitation detection
        let threshold = 0.1; // Cavitation threshold
        output.mapv_inplace(|v| if v.abs() > threshold { v } else { 0.0 });
    }
    
    /// Detect sonoluminescence events from acoustic emissions
    fn detect_sonoluminescence(&mut self, pressure_field: &Array3<f64>) {
        // Sonoluminescence detection based on acoustic signatures
        // Reference: Gaitan et al. (1992) "Sonoluminescence and bubble dynamics"
        
        // Look for characteristic high-frequency emissions
        for i in 0..self.sonoluminescence_map.dim().0 {
            for j in 0..self.sonoluminescence_map.dim().1 {
                for k in 0..self.sonoluminescence_map.dim().2 {
                    let p = pressure_field[[i, j, k]];
                    
                    // Sonoluminescence typically occurs at high acoustic pressures
                    // with specific spectral characteristics
                    if p.abs() > 1e6 { // 1 MPa threshold
                        self.sonoluminescence_map[[i, j, k]] += 1.0;
                    }
                }
            }
        }
    }
}

impl PhysicsPlugin for PassiveAcousticMappingPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> crate::physics::plugin::PluginState {
        crate::physics::plugin::PluginState::Initialized
    }
    
    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }
    
    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![] // PAM doesn't modify fields, it records and maps them
    }
    
    fn initialize(
        &mut self,
        _grid: &Grid,
        _medium: &dyn crate::medium::Medium,
    ) -> KwaversResult<()> {
        // Clear recorded data
        for signal in &mut self.recorded_signals {
            signal.clear();
        }
        self.cavitation_map.fill(0.0);
        self.sonoluminescence_map.fill(0.0);
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        _medium: &dyn crate::medium::Medium,
        dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Record pressure at sensor locations
        let pressure_field = fields.index_axis(Axis(0), 0);
        
        for (sensor_idx, indices) in self.sensor_indices.iter().enumerate() {
            let pressure = pressure_field[[indices[0], indices[1], indices[2]]];
            self.recorded_signals[sensor_idx].push(pressure);
        }
        
        // Perform beamforming for each frequency band
        for (f_min, f_max) in &self.config.frequency_bands {
            let center_freq = (f_min + f_max) / 2.0;
            let reconstructed = self.beamform(grid, center_freq);
            
            // Update cavitation map
            if self.config.detect_cavitation {
                self.cavitation_map = self.cavitation_map.clone() + reconstructed;
            }
        }
        
        // Detect sonoluminescence
        if self.config.detect_sonoluminescence {
            self.detect_sonoluminescence(&pressure_field.to_owned());
        }
        
        Ok(())
    }
    
    fn finalize(&mut self) -> KwaversResult<()> {
        Ok(())
    }
    
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
        Box::new(PassiveAcousticMappingPlugin {
            metadata: self.metadata.clone(),
            config: self.config.clone(),
            sensor_positions: self.sensor_positions.clone(),
            sensor_indices: self.sensor_indices.clone(),
            recorded_signals: self.recorded_signals.clone(),
            cavitation_map: self.cavitation_map.clone(),
            sonoluminescence_map: self.sonoluminescence_map.clone(),
            frequency_domain_data: self.frequency_domain_data.clone(),
        })
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// Helper functions
fn normalize_vector(v: [f64; 3]) -> [f64; 3] {
    let norm = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    [v[0]/norm, v[1]/norm, v[2]/norm]
}

fn get_perpendicular_vector(v: [f64; 3]) -> [f64; 3] {
    if v[0].abs() < 0.9 {
        normalize_vector([1.0, 0.0, 0.0])
    } else {
        normalize_vector([0.0, 1.0, 0.0])
    }
}

fn cross_product(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
}

/// Get cavitation map from PAM plugin
impl PassiveAcousticMappingPlugin {
    pub fn get_cavitation_map(&self) -> &Array3<f64> {
        &self.cavitation_map
    }
    
    pub fn get_sonoluminescence_map(&self) -> &Array3<f64> {
        &self.sonoluminescence_map
    }
    
    pub fn get_recorded_signals(&self) -> &Vec<Vec<f64>> {
        &self.recorded_signals
    }
}