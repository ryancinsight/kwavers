// src/source/phased_array.rs
//! Phased Array Transducer Implementation
//! 
//! This module implements phased array transducers with electronic beam steering,
//! focusing capabilities, and realistic element modeling. Follows SOLID, CUPID, GRASP,
//! and CLEAN principles for maintainable, extensible architecture.
//!
//! Key Features:
//! - Multi-element array configurations with independent control
//! - Electronic beam focusing and steering algorithms  
//! - Phase delay calculations for desired beam patterns
//! - Element cross-talk modeling for realistic behavior
//! - Frequency-dependent sensitivity patterns

use crate::grid::Grid;
use crate::medium::Medium;
use crate::signal::Signal;
use crate::source::{Apodization, Source};
use crate::error::KwaversResult;
use ndarray::Array2;
use std::fmt::Debug;
use std::sync::Arc;

/// Configuration for phased array transducer geometry and behavior
#[derive(Debug, Clone)]
pub struct PhasedArrayConfig {
    /// Number of elements in the array
    pub num_elements: usize,
    /// Element spacing [m]
    pub element_spacing: f64,
    /// Element width [m] 
    pub element_width: f64,
    /// Element height [m]
    pub element_height: f64,
    /// Array center position (x, y, z) [m]
    pub center_position: (f64, f64, f64),
    /// Operating frequency [Hz]
    pub frequency: f64,
    /// Enable element cross-talk modeling
    pub enable_crosstalk: bool,
    /// Cross-talk coupling coefficient [0.0-1.0] (0.0 = no coupling, 1.0 = full coupling)
    pub crosstalk_coefficient: f64,
}

impl Default for PhasedArrayConfig {
    fn default() -> Self {
        Self {
            num_elements: 64,
            element_spacing: 0.3e-3, // 0.3mm (λ/2 at 2.5MHz in tissue)
            element_width: 0.25e-3,  // 0.25mm
            element_height: 10e-3,   // 10mm
            center_position: (0.0, 0.0, 0.0),
            frequency: 2.5e6,        // 2.5 MHz
            enable_crosstalk: true,
            crosstalk_coefficient: 0.1, // 10% coupling
        }
    }
}

/// Individual transducer element with physics-based modeling
#[derive(Debug, Clone)]
pub struct TransducerElement {
    /// Element ID
    pub id: usize,
    /// Position (x, y, z) [m]
    pub position: (f64, f64, f64),
    /// Width [m]
    pub width: f64,
    /// Height [m] 
    pub height: f64,
    /// Phase delay [rad]
    pub phase_delay: f64,
    /// Amplitude weight [0.0-1.0]
    pub amplitude_weight: f64,
    /// Element sensitivity pattern
    pub sensitivity: ElementSensitivity,
}

/// Element sensitivity pattern modeling
#[derive(Debug, Clone)]
pub struct ElementSensitivity {
    /// Main lobe width [rad]
    pub main_lobe_width: f64,
    /// Side lobe level [dB]
    pub side_lobe_level: f64,
    /// Frequency response coefficients
    pub frequency_response: Vec<f64>,
}

impl Default for ElementSensitivity {
    fn default() -> Self {
        Self {
            main_lobe_width: 0.5, // ~30 degrees
            side_lobe_level: -20.0, // -20 dB
            frequency_response: vec![1.0, 0.8, 0.6, 0.4], // Simple rolloff
        }
    }
}

/// Beamforming algorithms for phased array control
#[derive(Debug, Clone)]
pub enum BeamformingMode {
    /// Focus at specific point (x, y, z) [m]
    Focus { target: (f64, f64, f64) },
    /// Steer beam to angle (theta, phi) [rad]
    Steer { theta: f64, phi: f64 },
    /// Custom phase delays [rad]
    Custom { delays: Vec<f64> },
    /// Plane wave transmission
    PlaneWave { direction: (f64, f64, f64) },
}

/// Advanced phased array transducer with electronic beam control
#[derive(Debug)]
pub struct PhasedArrayTransducer {
    /// Array configuration
    config: PhasedArrayConfig,
    /// Individual elements
    elements: Vec<TransducerElement>,
    /// Signal generator
    signal: Arc<dyn Signal>,
    /// Beamforming mode
    beamforming_mode: BeamformingMode,
    /// Sound speed in medium [m/s]
    sound_speed: f64,
    /// Cross-talk coupling matrix
    crosstalk_matrix: Option<Array2<f64>>,
}

impl Clone for PhasedArrayTransducer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            elements: self.elements.clone(),
            signal: self.signal.clone(),
            beamforming_mode: self.beamforming_mode.clone(),
            sound_speed: self.sound_speed,
            crosstalk_matrix: self.crosstalk_matrix.clone(),
        }
    }
}

impl PhasedArrayTransducer {
    /// Create new phased array transducer
    pub fn new(
        config: PhasedArrayConfig,
        signal: Arc<dyn Signal>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Self> {
        // Validate configuration
        Self::validate_config(&config)?;
        
        // Get sound speed from medium
        let sound_speed = medium.sound_speed(
            config.center_position.0,
            config.center_position.1, 
            config.center_position.2,
            grid
        );
        
        // Create elements
        let elements = Self::create_elements(&config)?;
        
        // Initialize with focus at 50mm depth
        let default_focus = (
            config.center_position.0,
            config.center_position.1,
            config.center_position.2 + 50e-3
        );
        
        let mut transducer = Self {
            config: config.clone(),
            elements,
            signal,
            beamforming_mode: BeamformingMode::Focus { target: default_focus },
            sound_speed,
            crosstalk_matrix: None,
        };
        
        // Calculate phase delays for initial beamforming
        transducer.update_beamforming()?;
        
        // Initialize cross-talk matrix if enabled
        if config.enable_crosstalk {
            transducer.initialize_crosstalk_matrix()?;
        }
        
        Ok(transducer)
    }
    
    /// Validate configuration parameters
    fn validate_config(config: &PhasedArrayConfig) -> KwaversResult<()> {
        if config.num_elements == 0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "num_elements".to_string(),
                value: config.num_elements.to_string(),
                constraint: "Must be greater than 0".to_string(),
            }.into());
        }
        
        if config.element_spacing <= 0.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "element_spacing".to_string(),
                value: config.element_spacing.to_string(),
                constraint: "Must be positive".to_string(),
            }.into());
        }
        
        if config.frequency <= 0.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "frequency".to_string(),
                value: config.frequency.to_string(),
                constraint: "Must be positive".to_string(),
            }.into());
        }
        
        if !(0.0..=1.0).contains(&config.crosstalk_coefficient) {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "crosstalk_coefficient".to_string(),
                value: config.crosstalk_coefficient.to_string(),
                constraint: "Must be between 0.0 and 1.0".to_string(),
            }.into());
        }
        
        Ok(())
    }
    
    /// Create array elements based on configuration
    fn create_elements(config: &PhasedArrayConfig) -> KwaversResult<Vec<TransducerElement>> {
        let mut elements = Vec::with_capacity(config.num_elements);
        
        // Calculate element positions (linear array along x-axis)
        let array_length = (config.num_elements - 1) as f64 * config.element_spacing;
        let start_x = config.center_position.0 - array_length / 2.0;
        
        for i in 0..config.num_elements {
            let x = start_x + i as f64 * config.element_spacing;
            let position = (x, config.center_position.1, config.center_position.2);
            
            let element = TransducerElement {
                id: i,
                position,
                width: config.element_width,
                height: config.element_height,
                phase_delay: 0.0, // Will be set by beamforming
                amplitude_weight: 1.0, // Default uniform weighting
                sensitivity: ElementSensitivity::default(),
            };
            
            elements.push(element);
        }
        
        Ok(elements)
    }
    
    /// Update beamforming parameters
    pub fn update_beamforming(&mut self) -> KwaversResult<()> {
        match &self.beamforming_mode {
            BeamformingMode::Focus { target } => {
                self.calculate_focus_delays(*target)?;
            },
            BeamformingMode::Steer { theta, phi } => {
                self.calculate_steering_delays(*theta, *phi)?;
            },
            BeamformingMode::Custom { delays } => {
                self.set_custom_delays(delays.clone())?;
            },
            BeamformingMode::PlaneWave { direction } => {
                self.calculate_plane_wave_delays(*direction)?;
            },
        }
        Ok(())
    }
    
    /// Calculate phase delays for focusing at target point
    fn calculate_focus_delays(&mut self, target: (f64, f64, f64)) -> KwaversResult<()> {
        let wavelength = self.sound_speed / self.config.frequency;
        let k = 2.0 * std::f64::consts::PI / wavelength;
        
        // Find reference distance (usually center element)
        let center_idx = self.elements.len() / 2;
        let ref_pos = self.elements[center_idx].position;
        let ref_distance = Self::distance(ref_pos, target);
        
        // Calculate delays for each element
        for element in &mut self.elements {
            let distance = Self::distance(element.position, target);
            let path_difference = distance - ref_distance;
            element.phase_delay = -k * path_difference; // Negative for focusing
        }
        
        Ok(())
    }
    
    /// Calculate phase delays for beam steering
    fn calculate_steering_delays(&mut self, theta: f64, phi: f64) -> KwaversResult<()> {
        let wavelength = self.sound_speed / self.config.frequency;
        let k = 2.0 * std::f64::consts::PI / wavelength;
        
        // Steering direction vector
        let kx = k * theta.sin() * phi.cos();
        let ky = k * theta.sin() * phi.sin();
        
        // Reference position (center element)
        let center_idx = self.elements.len() / 2;
        let ref_pos = self.elements[center_idx].position;
        
        for element in &mut self.elements {
            let dx = element.position.0 - ref_pos.0;
            let dy = element.position.1 - ref_pos.1;
            element.phase_delay = -(kx * dx + ky * dy);
        }
        
        Ok(())
    }
    
    /// Set custom phase delays
    fn set_custom_delays(&mut self, delays: Vec<f64>) -> KwaversResult<()> {
        if delays.len() != self.elements.len() {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "custom_delays".to_string(),
                value: delays.len().to_string(),
                constraint: format!("Must have {} elements", self.elements.len()),
            }.into());
        }
        
        for (element, &delay) in self.elements.iter_mut().zip(delays.iter()) {
            element.phase_delay = delay;
        }
        
        Ok(())
    }
    
    /// Calculate plane wave delays
    fn calculate_plane_wave_delays(&mut self, direction: (f64, f64, f64)) -> KwaversResult<()> {
        let wavelength = self.sound_speed / self.config.frequency;
        let k = 2.0 * std::f64::consts::PI / wavelength;
        
        // Normalize direction vector
        let norm = (direction.0.powi(2) + direction.1.powi(2) + direction.2.powi(2)).sqrt();
        let dir = (direction.0 / norm, direction.1 / norm, direction.2 / norm);
        
        // Reference position
        let ref_pos = self.elements[0].position;
        
        for element in &mut self.elements {
            let dx = element.position.0 - ref_pos.0;
            let dy = element.position.1 - ref_pos.1;
            let dz = element.position.2 - ref_pos.2;
            
            let projection = dx * dir.0 + dy * dir.1 + dz * dir.2;
            element.phase_delay = -k * projection;
        }
        
        Ok(())
    }
    
    /// Initialize cross-talk coupling matrix
    fn initialize_crosstalk_matrix(&mut self) -> KwaversResult<()> {
        let n = self.elements.len();
        let mut matrix = Array2::zeros((n, n));
        
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[[i, j]] = 1.0; // Self-coupling
                } else {
                    // Distance-based coupling (exponential decay)
                    let distance = Self::distance(
                        self.elements[i].position,
                        self.elements[j].position
                    );
                    let coupling = self.config.crosstalk_coefficient * 
                                 (-distance / self.config.element_spacing).exp();
                    matrix[[i, j]] = coupling;
                }
            }
        }
        
        self.crosstalk_matrix = Some(matrix);
        Ok(())
    }
    
    /// Calculate distance between two points
    fn distance(p1: (f64, f64, f64), p2: (f64, f64, f64)) -> f64 {
        ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2) + (p1.2 - p2.2).powi(2)).sqrt()
    }
    
    /// Set beamforming mode
    pub fn set_beamforming_mode(&mut self, mode: BeamformingMode) -> KwaversResult<()> {
        self.beamforming_mode = mode;
        self.update_beamforming()
    }
    
    /// Get element positions
    pub fn element_positions(&self) -> Vec<(f64, f64, f64)> {
        self.elements.iter().map(|e| e.position).collect()
    }
    
    /// Get element phase delays
    pub fn element_delays(&self) -> Vec<f64> {
        self.elements.iter().map(|e| e.phase_delay).collect()
    }
    
    /// Apply apodization weighting
    pub fn apply_apodization<A: Apodization>(&mut self, apodization: A) {
        let num_elements = self.elements.len();
        for (i, element) in self.elements.iter_mut().enumerate() {
            element.amplitude_weight = apodization.weight(i, num_elements);
        }
    }
}

impl Source for PhasedArrayTransducer {
    fn create_mask(&self, grid: &Grid) -> ndarray::Array3<f64> {
        let mut mask = ndarray::Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        for element in &self.elements {
            if let Some((ix, iy, iz)) = grid.to_grid_indices(element.position.0, element.position.1, element.position.2) {
                mask[(ix, iy, iz)] = element.amplitude_weight;
            }
        }
        
        mask
    }
    
    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }
    
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
        let mut total_source = 0.0;
        
        // Calculate contribution from each element
        for element in &self.elements {
            let _distance = Self::distance((x, y, z), element.position);
            
            // Element spatial response (rectangular aperture model)
            let spatial_response = self.calculate_element_response(element, x, y, z);
            
            // Time-delayed signal
            let time_delay = element.phase_delay / (2.0 * std::f64::consts::PI * self.config.frequency);
            let delayed_time = t - time_delay;
            let temporal_response = self.signal.amplitude(delayed_time) * element.amplitude_weight;
            
            total_source += spatial_response * temporal_response;
        }
        
        // Apply cross-talk if enabled
        if let Some(ref crosstalk_matrix) = self.crosstalk_matrix {
            // Apply cross-talk effects using the crosstalk_matrix
            for (i, element) in self.elements.iter().enumerate() {
                for (j, other_element) in self.elements.iter().enumerate() {
                    if i != j {
                        let coupling = crosstalk_matrix[[i, j]] * self.config.crosstalk_coefficient;
                        let spatial_response = self.calculate_element_response(other_element, x, y, z);
                        let time_delay = other_element.phase_delay / (2.0 * std::f64::consts::PI * self.config.frequency);
                        let delayed_time = t - time_delay;
                        let temporal_response = self.signal.amplitude(delayed_time) * other_element.amplitude_weight;
                        total_source += coupling * spatial_response * temporal_response;
                    }
                }
            }
        }
        
        total_source
    }
    
    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.element_positions()
    }
    
    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }
}

impl PhasedArrayTransducer {
    /// Calculate spatial response of individual element
    fn calculate_element_response(&self, element: &TransducerElement, x: f64, y: f64, z: f64) -> f64 {
        let dx = x - element.position.0;
        let dy = y - element.position.1;
        let dz = z - element.position.2;
        
        // Distance from element center
        let r = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();
        
        // Rectangular aperture response (sinc function calculation)
        let kx = 2.0 * std::f64::consts::PI * dx / (self.sound_speed / self.config.frequency);
        let ky = 2.0 * std::f64::consts::PI * dy / (self.sound_speed / self.config.frequency);
        
        let sinc_x = self.sinc(kx, element.width);
        let sinc_y = self.sinc(ky, element.height);
        
        // Distance-based attenuation
        let distance_factor = 1.0 / (r + 1e-9); // Add a small epsilon to prevent instability
        
        sinc_x * sinc_y * distance_factor
    }

    /// Helper method to calculate the sinc function for a given k and dimension
    fn sinc(&self, k: f64, dimension: f64) -> f64 {
        let argument = k * dimension / 2.0;
        if argument.abs() < 1e-10 {
            1.0
        } else {
            argument.sin() / argument
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::homogeneous::HomogeneousMedium;
    use crate::signal::SineWave;
    
    use crate::utils::test_helpers::{create_default_test_grid, create_test_medium};
    
    #[test]
    fn test_phased_array_creation() {
        let grid = create_default_test_grid();
        let medium = create_test_medium(&grid);
        let config = PhasedArrayConfig::default();
        let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));
        
        let array = PhasedArrayTransducer::new(config.clone(), signal, &medium, &grid);
        assert!(array.is_ok());
        
        let array = array.unwrap();
        assert_eq!(array.elements.len(), config.num_elements);
        assert_eq!(array.config.frequency, 2.5e6);
    }
    
    #[test]
    fn test_focus_beamforming() {
        let grid = create_default_test_grid();
        let medium = create_test_medium(&grid);
        let config = PhasedArrayConfig::default();
        let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));
        
        let mut array = PhasedArrayTransducer::new(config, signal, &medium, &grid).unwrap();
        
        // Test focusing at 50mm depth
        let focus_target = (0.0, 0.0, 50e-3);
        let result = array.set_beamforming_mode(BeamformingMode::Focus { target: focus_target });
        assert!(result.is_ok());
        
        // Check that delays are calculated
        let delays = array.element_delays();
        assert_eq!(delays.len(), array.elements.len());
        
        // Center elements should have minimum delay
        let center_idx = array.elements.len() / 2;
        let center_delay = delays[center_idx];
        
        // Edge elements should have larger delays for focusing
        assert!(delays[0].abs() >= center_delay.abs());
        assert!(delays[delays.len() - 1].abs() >= center_delay.abs());
    }
    
    #[test]
    fn test_steering_beamforming() {
        let grid = create_default_test_grid();
        let medium = create_test_medium(&grid);
        let config = PhasedArrayConfig::default();
        let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));
        
        let mut array = PhasedArrayTransducer::new(config, signal, &medium, &grid).unwrap();
        
        // Test steering to 30 degrees
        let theta = 30.0_f64.to_radians();
        let phi = 0.0;
        let result = array.set_beamforming_mode(BeamformingMode::Steer { theta, phi });
        assert!(result.is_ok());
        
        let delays = array.element_delays();
        
        // Delays should vary linearly across array for steering
        let delay_diff = delays[delays.len() - 1] - delays[0];
        assert!(delay_diff.abs() > 0.0);
    }
    
    #[test]
    fn test_custom_delays() {
        let grid = create_default_test_grid();
        let medium = create_test_medium(&grid);
        let config = PhasedArrayConfig::default();
        let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));
        
        let mut array = PhasedArrayTransducer::new(config.clone(), signal, &medium, &grid).unwrap();
        
        // Test custom delays
        let custom_delays: Vec<f64> = (0..config.num_elements).map(|i| i as f64 * 0.1).collect();
        let result = array.set_beamforming_mode(BeamformingMode::Custom { 
            delays: custom_delays.clone() 
        });
        assert!(result.is_ok());
        
        let applied_delays = array.element_delays();
        assert_eq!(applied_delays, custom_delays);
    }
    
    #[test]
    fn test_source_term_calculation() {
        let grid = create_default_test_grid();
        let medium = create_test_medium(&grid);
        let config = PhasedArrayConfig::default();
        let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));
        
        let array = PhasedArrayTransducer::new(config, signal, &medium, &grid).unwrap();
        
        // Test source term at various points
        let source_term_center = array.get_source_term(0.0, 0.0, 0.0, 10e-3, &grid);
        let source_term_off_axis = array.get_source_term(0.0, 5e-3, 0.0, 10e-3, &grid);
        
        // Source term should be non-zero
        assert!(source_term_center.abs() > 0.0);
        assert!(source_term_off_axis.abs() > 0.0);
        
        // On-axis should typically be stronger than off-axis
        assert!(source_term_center.abs() >= source_term_off_axis.abs());
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = PhasedArrayConfig::default();
        
        // Test invalid num_elements
        config.num_elements = 0;
        assert!(PhasedArrayTransducer::validate_config(&config).is_err());
        
        config.num_elements = 64;
        
        // Test invalid element_spacing
        config.element_spacing = -1.0;
        assert!(PhasedArrayTransducer::validate_config(&config).is_err());
        
        config.element_spacing = 0.3e-3;
        
        // Test invalid frequency
        config.frequency = -1000.0;
        assert!(PhasedArrayTransducer::validate_config(&config).is_err());
        
        config.frequency = 2.5e6;
        
        // Test invalid crosstalk_coefficient
        config.crosstalk_coefficient = 1.5;
        assert!(PhasedArrayTransducer::validate_config(&config).is_err());
        
        config.crosstalk_coefficient = 0.1;
        
        // Valid configuration should pass
        assert!(PhasedArrayTransducer::validate_config(&config).is_ok());
    }
    
    #[test]
    fn test_element_positions() {
        let grid = create_default_test_grid();
        let medium = create_test_medium(&grid);
        let config = PhasedArrayConfig {
            num_elements: 4,
            element_spacing: 1e-3,
            center_position: (0.0, 0.0, 0.0),
            ..Default::default()
        };
        let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));
        
        let array = PhasedArrayTransducer::new(config, signal, &medium, &grid).unwrap();
        let positions = array.element_positions();
        
        assert_eq!(positions.len(), 4);
        
        // Check symmetric positioning around center
        assert_eq!(positions[0].0, -1.5e-3); // First element
        assert_eq!(positions[1].0, -0.5e-3); // Second element
        assert_eq!(positions[2].0, 0.5e-3);  // Third element
        assert_eq!(positions[3].0, 1.5e-3);  // Fourth element
    }
    
    #[test]
    fn test_plane_wave_beamforming() {
        let grid = create_default_test_grid();
        let medium = create_test_medium(&grid);
        let config = PhasedArrayConfig::default();
        let signal = Arc::new(SineWave::new(2.5e6, 1.0, 0.0));
        
        let mut array = PhasedArrayTransducer::new(config, signal, &medium, &grid).unwrap();
        
        // Test plane wave in z-direction
        let direction = (0.0, 0.0, 1.0);
        let result = array.set_beamforming_mode(BeamformingMode::PlaneWave { direction });
        assert!(result.is_ok());
        
        let delays = array.element_delays();
        
        // For plane wave in z-direction, all delays should be equal (array is in x-y plane)
        let first_delay = delays[0];
        for &delay in &delays {
            assert!((delay - first_delay).abs() < 1e-10);
        }
    }
}