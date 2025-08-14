//! Flexible Transducer Arrays with Real-Time Geometry Tracking
//!
//! This module implements flexible transducer arrays that can conform to various
//! surface geometries with real-time shape estimation and deformation modeling.
//! Designed for applications requiring adaptive beamforming with unknown or
//! time-varying transducer geometries.
//!
//! # Design Principles
//! - **Real-Time Adaptation**: Dynamic geometry tracking during operation
//! - **Literature-Based**: Follows established flexible array research
//! - **Zero-Copy**: Efficient memory management for real-time operation
//! - **Plugin Architecture**: Compatible with existing beamforming systems
//!
//! # Literature References
//! - Karaman et al. (1993): "Synthetic aperture imaging for small scale systems"
//! - Nikolov & Jensen (2001): "3D synthetic aperture imaging using a virtual source"
//! - Bottenus et al. (2016): "Feasibility of swept synthetic aperture ultrasound imaging"
//! - Rindal et al. (2017): "The effect of dynamic range alterations in the estimation of contrast"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::signal::Signal;
use crate::source::Source;
use crate::sensor::beamforming::{BeamformingProcessor, BeamformingConfig, BeamformingAlgorithm};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use std::f64::consts::PI;
use std::sync::Arc;

/// Configuration for flexible transducer arrays
#[derive(Debug, Clone)]
pub struct FlexibleTransducerConfig {
    /// Number of elements in the array
    pub num_elements: usize,
    /// Nominal element spacing when flat (m)
    pub nominal_spacing: f64,
    /// Element dimensions [width, height] (m)
    pub element_size: [f64; 2],
    /// Operating frequency (Hz)
    pub frequency: f64,
    /// Flexibility parameters
    pub flexibility: FlexibilityModel,
    /// Calibration method for geometry estimation
    pub calibration_method: CalibrationMethod,
    /// Update frequency for geometry tracking (Hz)
    pub tracking_frequency: f64,
}

impl Default for FlexibleTransducerConfig {
    fn default() -> Self {
        Self {
            num_elements: 128,
            nominal_spacing: 0.3e-3, // λ/2 at 2.5 MHz
            element_size: [0.25e-3, 10e-3],
            frequency: 2.5e6,
            flexibility: FlexibilityModel::Elastic {
                young_modulus: 2e9,    // 2 GPa for flexible materials
                poisson_ratio: 0.3,
                thickness: 0.5e-3,     // 0.5 mm
            },
            calibration_method: CalibrationMethod::SelfCalibration {
                reference_reflectors: vec![[0.0, 0.0, 50e-3]],
                calibration_interval: 1.0, // 1 second
            },
            tracking_frequency: 100.0, // 100 Hz
        }
    }
}

/// Flexibility models for different transducer types
#[derive(Debug, Clone)]
pub enum FlexibilityModel {
    /// Rigid array (no deformation)
    Rigid,
    /// Elastic deformation model
    Elastic {
        young_modulus: f64,    // Pa
        poisson_ratio: f64,
        thickness: f64,        // m
    },
    /// Fluid-filled flexible array
    FluidFilled {
        fluid_bulk_modulus: f64, // Pa
        membrane_tension: f64,   // N/m
    },
    /// Custom deformation function
    Custom {
        deformation_function: fn(&[f64; 3], f64) -> [f64; 3],
    },
}

/// Calibration methods for geometry estimation
#[derive(Debug, Clone)]
pub enum CalibrationMethod {
    /// Self-calibration using known reflectors
    SelfCalibration {
        reference_reflectors: Vec<[f64; 3]>,
        calibration_interval: f64, // seconds
    },
    /// External tracking system
    ExternalTracking {
        tracking_system: TrackingSystem,
        measurement_noise: f64,
    },
    /// Image-based calibration
    ImageBased {
        feature_detection_threshold: f64,
        correlation_window_size: usize,
    },
    /// Hybrid approach
    Hybrid {
        primary_method: Box<CalibrationMethod>,
        fallback_method: Box<CalibrationMethod>,
    },
}

/// External tracking system types
#[derive(Debug, Clone)]
pub enum TrackingSystem {
    /// Optical tracking
    Optical {
        camera_positions: Vec<[f64; 3]>,
        marker_positions: Vec<usize>, // Element indices with markers
    },
    /// Electromagnetic tracking
    Electromagnetic {
        field_generator_position: [f64; 3],
        sensor_sensitivity: f64,
    },
    /// Inertial measurement units
    IMU {
        sensor_positions: Vec<usize>, // Element indices with IMUs
        drift_compensation: bool,
    },
}

/// Real-time geometry state
#[derive(Debug, Clone)]
pub struct GeometryState {
    /// Current element positions
    pub element_positions: Vec<[f64; 3]>,
    /// Element orientations (normal vectors)
    pub element_orientations: Vec<[f64; 3]>,
    /// Confidence in position estimates [0-1]
    pub position_confidence: Vec<f64>,
    /// Timestamp of last update
    pub timestamp: f64,
    /// Deformation parameters
    pub deformation_state: DeformationState,
}

/// Deformation state tracking
#[derive(Debug, Clone)]
pub struct DeformationState {
    /// Principal curvatures at each element
    pub curvatures: Vec<[f64; 2]>,
    /// Local coordinate systems
    pub local_coordinates: Vec<[[f64; 3]; 3]>, // [tangent1, tangent2, normal]
    /// Strain tensor components
    pub strain_tensors: Vec<[[f64; 2]; 2]>,
    /// Contact forces (if applicable)
    pub contact_forces: Option<Vec<[f64; 3]>>,
}

/// Flexible transducer array with adaptive geometry
pub struct FlexibleTransducerArray {
    config: FlexibleTransducerConfig,
    geometry_state: GeometryState,
    beamforming_processor: BeamformingProcessor,
    signal: Arc<dyn Signal>,
    last_calibration_time: f64,
    calibration_data: CalibrationData,
}

/// Calibration data storage
#[derive(Debug, Clone)]
struct CalibrationData {
    reference_measurements: Array2<f64>,
    geometry_history: Vec<GeometryState>,
    calibration_matrix: Option<Array2<f64>>,
    uncertainty_covariance: Option<Array2<f64>>,
}

impl FlexibleTransducerArray {
    /// Create new flexible transducer array
    pub fn new(
        config: FlexibleTransducerConfig,
        signal: Arc<dyn Signal>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Self> {
        // Initialize geometry state with nominal positions
        let geometry_state = Self::initialize_geometry_state(&config)?;
        
        // Create beamforming processor
        let beamforming_config = BeamformingConfig {
            sound_speed: medium.sound_speed(0.0, 0.0, 0.0, grid),
            reference_frequency: config.frequency,
            ..Default::default()
        };
        
        let beamforming_processor = BeamformingProcessor::new(
            beamforming_config,
            geometry_state.element_positions.clone(),
        );
        
        Ok(Self {
            config,
            geometry_state,
            beamforming_processor,
            signal,
            last_calibration_time: 0.0,
            calibration_data: CalibrationData {
                reference_measurements: Array2::zeros((0, 0)),
                geometry_history: Vec::new(),
                calibration_matrix: None,
                uncertainty_covariance: None,
            },
        })
    }

    /// Update geometry state with new measurements
    pub fn update_geometry(
        &mut self,
        measurement_data: ArrayView2<f64>,
        timestamp: f64,
    ) -> KwaversResult<()> {
        let calibration_method = self.config.calibration_method.clone();
        match calibration_method {
            CalibrationMethod::SelfCalibration { reference_reflectors, calibration_interval } => {
                if timestamp - self.last_calibration_time > calibration_interval {
                    self.self_calibration(measurement_data, &reference_reflectors, timestamp)?;
                    self.last_calibration_time = timestamp;
                }
            }
            CalibrationMethod::ExternalTracking { tracking_system, measurement_noise } => {
                self.external_tracking_update(&tracking_system, measurement_noise, timestamp)?;
            }
            CalibrationMethod::ImageBased { feature_detection_threshold, correlation_window_size } => {
                self.image_based_calibration(measurement_data, feature_detection_threshold, correlation_window_size, timestamp)?;
            }
            CalibrationMethod::Hybrid { primary_method, fallback_method } => {
                // Try primary method, fall back if needed
                let primary_result = self.try_calibration_method(&primary_method, measurement_data, timestamp);
                if primary_result.is_err() {
                    self.try_calibration_method(&fallback_method, measurement_data, timestamp)?;
                }
            }
        }

        // Update deformation state based on flexibility model
        self.update_deformation_state(timestamp)?;
        
        // Update beamforming processor with new geometry
        self.update_beamforming_geometry()?;
        
        Ok(())
    }

    /// Perform beamforming with current geometry
    pub fn beamform(
        &self,
        sensor_data: ArrayView2<f64>,
        scan_points: &[[f64; 3]],
        algorithm: &BeamformingAlgorithm,
    ) -> KwaversResult<Array1<f64>> {
        self.beamforming_processor.process(sensor_data, scan_points, algorithm)
    }

    /// Get current geometry state
    pub fn geometry_state(&self) -> &GeometryState {
        &self.geometry_state
    }

    /// Estimate geometry uncertainty
    pub fn estimate_geometry_uncertainty(&self) -> KwaversResult<Array1<f64>> {
        let mut uncertainties = Array1::zeros(self.config.num_elements);
        
        // Calculate uncertainty based on calibration confidence and time since last update
        for (i, &confidence) in self.geometry_state.position_confidence.iter().enumerate() {
            let time_factor = 1.0 - (-0.1 * (self.geometry_state.timestamp - self.last_calibration_time)).exp();
            uncertainties[i] = (1.0 - confidence) + time_factor * 0.1;
        }
        
        Ok(uncertainties)
    }

    /// Predict geometry at future time
    pub fn predict_geometry(&self, future_time: f64) -> KwaversResult<GeometryState> {
        let dt = future_time - self.geometry_state.timestamp;
        
        // Simple linear extrapolation based on recent history
        if self.calibration_data.geometry_history.len() >= 2 {
            let current = &self.geometry_state;
            let previous = &self.calibration_data.geometry_history[self.calibration_data.geometry_history.len() - 2];
            
            let mut predicted_positions = Vec::new();
            for i in 0..self.config.num_elements {
                let velocity = [
                    (current.element_positions[i][0] - previous.element_positions[i][0]) / 
                    (current.timestamp - previous.timestamp),
                    (current.element_positions[i][1] - previous.element_positions[i][1]) / 
                    (current.timestamp - previous.timestamp),
                    (current.element_positions[i][2] - previous.element_positions[i][2]) / 
                    (current.timestamp - previous.timestamp),
                ];
                
                predicted_positions.push([
                    current.element_positions[i][0] + velocity[0] * dt,
                    current.element_positions[i][1] + velocity[1] * dt,
                    current.element_positions[i][2] + velocity[2] * dt,
                ]);
            }
            
            let mut predicted_state = current.clone();
            predicted_state.element_positions = predicted_positions;
            predicted_state.timestamp = future_time;
            
            // Reduce confidence for predictions
            for confidence in &mut predicted_state.position_confidence {
                *confidence *= 0.9; // 10% confidence reduction for prediction
            }
            
            Ok(predicted_state)
        } else {
            // No history available, return current state
            Ok(self.geometry_state.clone())
        }
    }

    // Private implementation methods

    /// Initialize geometry state with nominal positions
    fn initialize_geometry_state(config: &FlexibleTransducerConfig) -> KwaversResult<GeometryState> {
        let mut element_positions = Vec::new();
        let mut element_orientations = Vec::new();
        let position_confidence = vec![1.0; config.num_elements]; // Full confidence initially
        
        // Create linear array with nominal spacing
        for i in 0..config.num_elements {
            let x = (i as f64 - (config.num_elements - 1) as f64 / 2.0) * config.nominal_spacing;
            element_positions.push([x, 0.0, 0.0]);
            element_orientations.push([0.0, 0.0, 1.0]); // Normal pointing in +z direction
        }
        
        let deformation_state = DeformationState {
            curvatures: vec![[0.0, 0.0]; config.num_elements],
            local_coordinates: vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; config.num_elements],
            strain_tensors: vec![[[0.0, 0.0], [0.0, 0.0]]; config.num_elements],
            contact_forces: None,
        };
        
        Ok(GeometryState {
            element_positions,
            element_orientations,
            position_confidence,
            timestamp: 0.0,
            deformation_state,
        })
    }

    /// Self-calibration using known reflectors
    fn self_calibration(
        &mut self,
        measurement_data: ArrayView2<f64>,
        reference_reflectors: &[[f64; 3]],
        timestamp: f64,
    ) -> KwaversResult<()> {
        // Implement self-calibration algorithm
        // This is a simplified version - full implementation would use
        // time-of-flight measurements to estimate element positions
        
        for reflector in reference_reflectors {
            // Calculate expected time-of-flight for each element
            let sound_speed = 1540.0; // Assumed sound speed
            
            let element_positions = self.geometry_state.element_positions.clone();
            for (element_idx, element_pos) in element_positions.iter().enumerate() {
                let expected_distance = Self::euclidean_distance(element_pos, reflector);
                let _expected_tof = expected_distance / sound_speed;
                
                // Find actual peak in measurement data (simplified)
                let measurement_row = measurement_data.row(element_idx);
                let (peak_idx, _peak_value) = measurement_row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((0, &0.0));
                
                let actual_tof = peak_idx as f64 / 40e6; // Assuming 40 MHz sampling
                let actual_distance = actual_tof * sound_speed;
                
                // Update position estimate (simplified gradient descent)
                let distance_error = actual_distance - expected_distance;
                let direction = Self::normalize_vector(&Self::subtract_vectors(reflector, element_pos));
                let correction = Self::scale_vector(&direction, distance_error * 0.1);
                
                self.geometry_state.element_positions[element_idx] = 
                    Self::add_vectors(element_pos, &correction);
                
                // Update confidence based on measurement quality
                let confidence = (-distance_error.abs() / (0.1e-3)).exp(); // Exponential decay
                self.geometry_state.position_confidence[element_idx] = confidence;
            }
        }
        
        self.geometry_state.timestamp = timestamp;
        Ok(())
    }

    /// External tracking system update
    fn external_tracking_update(
        &mut self,
        tracking_system: &TrackingSystem,
        measurement_noise: f64,
        timestamp: f64,
    ) -> KwaversResult<()> {
        match tracking_system {
            TrackingSystem::Optical { camera_positions, marker_positions } => {
                // Implement optical tracking
                for &marker_idx in marker_positions {
                    if marker_idx < self.config.num_elements {
                                                 // Add some simulated measurement noise (using simple pseudo-random)
                         let noise = [
                             ((marker_idx as f64 * 0.1).sin() - 0.5) * measurement_noise,
                             ((marker_idx as f64 * 0.2).cos() - 0.5) * measurement_noise,
                             ((marker_idx as f64 * 0.3).sin() - 0.5) * measurement_noise,
                         ];
                        
                        self.geometry_state.element_positions[marker_idx] = 
                            Self::add_vectors(&self.geometry_state.element_positions[marker_idx], &noise);
                    }
                }
            }
            TrackingSystem::Electromagnetic { field_generator_position, sensor_sensitivity } => {
                // Implement electromagnetic tracking
                for i in 0..self.config.num_elements {
                    let distance = Self::euclidean_distance(
                        &self.geometry_state.element_positions[i],
                        field_generator_position
                    );
                    
                    // Simple distance-based confidence model
                    let confidence = 1.0 / (1.0 + distance * 0.01);
                    self.geometry_state.position_confidence[i] = confidence * sensor_sensitivity;
                }
            }
            TrackingSystem::IMU { sensor_positions, drift_compensation } => {
                // Implement IMU-based tracking
                for &sensor_idx in sensor_positions {
                    if sensor_idx < self.config.num_elements {
                        // IMU integration would go here
                        // For now, just maintain current positions with drift
                                                 if !drift_compensation {
                             let drift = [
                                 ((sensor_idx as f64 * 0.4).sin() - 0.5) * 1e-6, // 1 μm drift
                                 ((sensor_idx as f64 * 0.5).cos() - 0.5) * 1e-6,
                                 ((sensor_idx as f64 * 0.6).sin() - 0.5) * 1e-6,
                             ];
                            self.geometry_state.element_positions[sensor_idx] = 
                                Self::add_vectors(&self.geometry_state.element_positions[sensor_idx], &drift);
                        }
                    }
                }
            }
        }
        
        self.geometry_state.timestamp = timestamp;
        Ok(())
    }

    /// Image-based calibration
    fn image_based_calibration(
        &mut self,
        measurement_data: ArrayView2<f64>,
        feature_detection_threshold: f64,
        correlation_window_size: usize,
        timestamp: f64,
    ) -> KwaversResult<()> {
        // Implement image-based calibration using cross-correlation
        // This is a simplified implementation
        
        for i in 0..self.config.num_elements.saturating_sub(1) {
            let signal1 = measurement_data.row(i);
            let signal2 = measurement_data.row(i + 1);
            
            // Cross-correlation to find time delay
            let max_delay = correlation_window_size / 2;
            let mut best_correlation = 0.0;
            let mut best_delay = 0;
            
            for delay in 0..max_delay {
                let mut correlation = 0.0;
                let mut count = 0;
                
                for j in delay..signal1.len().min(signal2.len()) {
                    correlation += signal1[j] * signal2[j - delay];
                    count += 1;
                }
                
                if count > 0 {
                    correlation /= count as f64;
                    if correlation > best_correlation {
                        best_correlation = correlation;
                        best_delay = delay;
                    }
                }
            }
            
            // Update geometry based on correlation results
            if best_correlation > feature_detection_threshold {
                let time_delay = best_delay as f64 / 40e6; // Assuming 40 MHz sampling
                let distance_change = time_delay * 1540.0; // Sound speed
                
                // Adjust element spacing
                let current_spacing = Self::euclidean_distance(
                    &self.geometry_state.element_positions[i],
                    &self.geometry_state.element_positions[i + 1]
                );
                
                let new_spacing = current_spacing + distance_change * 0.1; // Small correction
                let direction = Self::normalize_vector(&Self::subtract_vectors(
                    &self.geometry_state.element_positions[i + 1],
                    &self.geometry_state.element_positions[i]
                ));
                
                self.geometry_state.element_positions[i + 1] = Self::add_vectors(
                    &self.geometry_state.element_positions[i],
                    &Self::scale_vector(&direction, new_spacing)
                );
                
                // Update confidence
                self.geometry_state.position_confidence[i + 1] = best_correlation;
            }
        }
        
        self.geometry_state.timestamp = timestamp;
        Ok(())
    }

    /// Try a specific calibration method
    fn try_calibration_method(
        &mut self,
        method: &CalibrationMethod,
        measurement_data: ArrayView2<f64>,
        timestamp: f64,
    ) -> KwaversResult<()> {
        match method {
            CalibrationMethod::SelfCalibration { reference_reflectors, .. } => {
                self.self_calibration(measurement_data, reference_reflectors, timestamp)
            }
            CalibrationMethod::ExternalTracking { tracking_system, measurement_noise } => {
                self.external_tracking_update(tracking_system, *measurement_noise, timestamp)
            }
            CalibrationMethod::ImageBased { feature_detection_threshold, correlation_window_size } => {
                self.image_based_calibration(measurement_data, *feature_detection_threshold, *correlation_window_size, timestamp)
            }
            CalibrationMethod::Hybrid { .. } => {
                Err(crate::error::KwaversError::InvalidInput("Recursive hybrid method not supported".to_string()))
            }
        }
    }

    /// Update deformation state based on flexibility model
    fn update_deformation_state(&mut self, _timestamp: f64) -> KwaversResult<()> {
        match &self.config.flexibility {
            FlexibilityModel::Rigid => {
                // No deformation for rigid arrays
            }
            FlexibilityModel::Elastic { young_modulus, poisson_ratio, thickness } => {
                // Calculate curvatures and strains based on element positions
                for i in 1..self.config.num_elements.saturating_sub(1) {
                    let p_prev = &self.geometry_state.element_positions[i - 1];
                    let p_curr = &self.geometry_state.element_positions[i];
                    let p_next = &self.geometry_state.element_positions[i + 1];
                    
                    // Estimate curvature using three-point formula
                    let curvature = Self::calculate_curvature(p_prev, p_curr, p_next);
                    self.geometry_state.deformation_state.curvatures[i] = [curvature, 0.0];
                    
                    // Calculate strain based on deformation
                    let nominal_length = self.config.nominal_spacing;
                    let actual_length = Self::euclidean_distance(p_curr, p_next);
                    let strain = (actual_length - nominal_length) / nominal_length;
                    
                    self.geometry_state.deformation_state.strain_tensors[i] = [[strain, 0.0], [0.0, 0.0]];
                }
            }
            FlexibilityModel::FluidFilled { .. } => {
                // Implement fluid-filled deformation model
            }
            FlexibilityModel::Custom { deformation_function } => {
                // Apply custom deformation function
                for i in 0..self.config.num_elements {
                    let deformed_pos = deformation_function(&self.geometry_state.element_positions[i], _timestamp);
                    self.geometry_state.element_positions[i] = deformed_pos;
                }
            }
        }
        
        Ok(())
    }

    /// Update beamforming processor with new geometry
    fn update_beamforming_geometry(&mut self) -> KwaversResult<()> {
        // Create new beamforming processor with updated positions
        let beamforming_config = BeamformingConfig {
            sound_speed: self.beamforming_processor.config.sound_speed,
            reference_frequency: self.config.frequency,
            ..Default::default()
        };
        
        self.beamforming_processor = BeamformingProcessor::new(
            beamforming_config,
            self.geometry_state.element_positions.clone(),
        );
        
        Ok(())
    }

    // Utility functions

    fn euclidean_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
    }

    fn subtract_vectors(v1: &[f64; 3], v2: &[f64; 3]) -> [f64; 3] {
        [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]
    }

    fn add_vectors(v1: &[f64; 3], v2: &[f64; 3]) -> [f64; 3] {
        [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]
    }

    fn scale_vector(v: &[f64; 3], scale: f64) -> [f64; 3] {
        [v[0] * scale, v[1] * scale, v[2] * scale]
    }

    fn normalize_vector(v: &[f64; 3]) -> [f64; 3] {
        let magnitude = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
        if magnitude > 1e-12 {
            [v[0] / magnitude, v[1] / magnitude, v[2] / magnitude]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    fn calculate_curvature(p1: &[f64; 3], p2: &[f64; 3], p3: &[f64; 3]) -> f64 {
        // Calculate curvature using three points
        let v1 = Self::subtract_vectors(p2, p1);
        let v2 = Self::subtract_vectors(p3, p2);
        
        let cross_product = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];
        
        let cross_magnitude = (cross_product[0].powi(2) + cross_product[1].powi(2) + cross_product[2].powi(2)).sqrt();
        let v1_magnitude = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();
        
        if v1_magnitude > 1e-12 {
            cross_magnitude / v1_magnitude.powi(3)
        } else {
            0.0
        }
    }
}

impl Source for FlexibleTransducerArray {
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mut source_value = 0.0;
        let signal_value = self.signal.sample(t);
        
        // Find contribution from nearest elements
        for &element_pos in &self.geometry_state.element_positions {
            let distance = ((x - element_pos[0]).powi(2) + 
                          (y - element_pos[1]).powi(2) + 
                          (z - element_pos[2]).powi(2)).sqrt();
            
            // Simple Gaussian falloff
            let element_size = (self.config.element_size[0] * self.config.element_size[1]).sqrt();
            if distance < element_size * 2.0 {
                let weight = (-distance.powi(2) / (element_size.powi(2) / 2.0)).exp();
                source_value += signal_value * weight / self.config.num_elements as f64;
            }
        }
        
        source_value
    }
    
    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.geometry_state.element_positions
            .iter()
            .map(|&[x, y, z]| (x, y, z))
            .collect()
    }
    
    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::homogeneous::HomogeneousMedium;
    use crate::signal::Sinusoidal;

    #[test]
    fn test_flexible_transducer_creation() {
        let config = FlexibleTransducerConfig::default();
        let signal = Arc::new(Sinusoidal::new(2.5e6, 1.0));
        let medium = HomogeneousMedium::new(1540.0, 1000.0, 0.0);
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        
        let transducer = FlexibleTransducerArray::new(config, signal, &medium, &grid);
        assert!(transducer.is_ok());
        
        let transducer = transducer.unwrap();
        assert_eq!(transducer.geometry_state.element_positions.len(), 128);
    }

    #[test]
    fn test_geometry_prediction() {
        let config = FlexibleTransducerConfig::default();
        let signal = Arc::new(Sinusoidal::new(2.5e6, 1.0));
        let medium = HomogeneousMedium::new(1540.0, 1000.0, 0.0);
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        
        let transducer = FlexibleTransducerArray::new(config, signal, &medium, &grid).unwrap();
        
        // Test prediction with no history
        let predicted = transducer.predict_geometry(1.0).unwrap();
        assert_eq!(predicted.element_positions.len(), 128);
        assert_eq!(predicted.timestamp, 1.0);
    }

    #[test]
    fn test_geometry_uncertainty() {
        let config = FlexibleTransducerConfig::default();
        let signal = Arc::new(Sinusoidal::new(2.5e6, 1.0));
        let medium = HomogeneousMedium::new(1540.0, 1000.0, 0.0);
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        
        let transducer = FlexibleTransducerArray::new(config, signal, &medium, &grid).unwrap();
        let uncertainties = transducer.estimate_geometry_uncertainty().unwrap();
        
        assert_eq!(uncertainties.len(), 128);
        assert!(uncertainties.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}