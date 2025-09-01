//! Main flexible transducer array implementation
//!
//! This module provides the main FlexibleTransducerArray struct that
//! integrates configuration, geometry, and calibration components.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::signal::Signal;
use crate::source::Source;
use ndarray::{Array3, ArrayView2};
use std::sync::Arc;

use super::calibration::CalibrationManager;
use super::config::{CalibrationMethod, FlexibilityModel, FlexibleTransducerConfig};
use super::geometry::{DeformationState, GeometryState};

/// Flexible transducer array with real-time geometry tracking
#[derive(Debug)]
pub struct FlexibleTransducerArray {
    /// Configuration
    config: FlexibleTransducerConfig,
    /// Current geometry state
    geometry_state: GeometryState,
    /// Calibration processor
    calibration_processor: CalibrationManager,
    /// Signal generator
    signal: Arc<dyn Signal>,
    /// Last update timestamp
    last_update_time: f64,
}

impl FlexibleTransducerArray {
    /// Create a new flexible transducer array
    pub fn new(config: FlexibleTransducerConfig, signal: Arc<dyn Signal>) -> KwaversResult<Self> {
        let geometry_state = GeometryState::flat_array(config.num_elements, config.nominal_spacing);

        let calibration_processor = CalibrationManager::new();

        Ok(Self {
            config,
            geometry_state,
            calibration_processor,
            signal,
            last_update_time: 0.0,
        })
    }

    /// Update geometry based on measurements
    pub fn update_geometry(
        &mut self,
        measurement_data: ArrayView2<f64>,
        timestamp: f64,
    ) -> KwaversResult<()> {
        // Process calibration based on configured method
        let new_positions = match &self.config.calibration_method {
            CalibrationMethod::SelfCalibration {
                reference_reflectors,
                calibration_interval,
            } => {
                if timestamp - self.last_update_time > *calibration_interval {
                    // For 2D measurement data, we use external tracking instead of self-calibration
                    // Self-calibration requires 3D pressure field data
                    // Measurement noise level based on typical ultrasound tracking accuracy
                    // Reference: Mercier et al. (2012) IEEE Trans. Ultrason. Ferroelectr. Freq. Control
                    const TRACKING_NOISE_LEVEL: f64 = 1e-3; // 1mm position uncertainty
                    self.calibration_processor.process_external_tracking(
                        &measurement_data.to_owned(),
                        TRACKING_NOISE_LEVEL,
                        timestamp,
                    )?
                } else {
                    return Ok(());
                }
            }
            CalibrationMethod::ExternalTracking {
                tracking_system,
                measurement_noise,
            } => {
                // In real implementation, would interface with tracking system
                self.calibration_processor.process_external_tracking(
                    &measurement_data.to_owned(),
                    *measurement_noise,
                    timestamp,
                )?
            }
            _ => {
                // Other calibration methods would be implemented here
                return Ok(());
            }
        };

        // Update geometry state
        let normals = self.calculate_normals(&new_positions);
        self.geometry_state.update_positions(new_positions, normals);
        self.geometry_state.timestamp = timestamp;

        // Update deformation state
        self.update_deformation_state()?;

        self.last_update_time = timestamp;

        Ok(())
    }

    /// Calculate element normals from positions
    fn calculate_normals(&self, positions: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
        let n = positions.nrows();
        let mut normals = ndarray::Array2::zeros((n, 3));

        for i in 0..n {
            // Simple normal calculation based on neighboring elements
            let (v1, v2) = if i == 0 {
                // First element
                let p0 = positions.row(0);
                let p1 = positions.row(1.min(n - 1));
                let v = &p1 - &p0;
                (v.clone(), v)
            } else if i == n - 1 {
                // Last element
                let p0 = positions.row(n - 2);
                let p1 = positions.row(n - 1);
                let v = &p1 - &p0;
                (v.clone(), v)
            } else {
                // Middle elements
                let p0 = positions.row(i - 1);
                let p1 = positions.row(i);
                let p2 = positions.row(i + 1);
                (&p1 - &p0, &p2 - &p1)
            };

            // Cross product to get normal (assuming array lies roughly in x-y plane)
            let normal = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ];

            // Normalize
            let mag = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
            if mag > 0.0 {
                normals[[i, 0]] = normal[0] / mag;
                normals[[i, 1]] = normal[1] / mag;
                normals[[i, 2]] = normal[2] / mag;
            } else {
                // Default to z-direction if undefined
                normals[[i, 2]] = 1.0;
            }
        }

        normals
    }

    /// Update deformation state based on current geometry
    fn update_deformation_state(&mut self) -> KwaversResult<()> {
        let curvature = self.geometry_state.calculate_curvature();

        // Update deformation based on flexibility model
        match &self.config.flexibility {
            FlexibilityModel::Rigid => {
                // No deformation for rigid arrays
                self.geometry_state.deformation = DeformationState {
                    curvature_radius: None,
                    strain: vec![0.0; self.config.num_elements],
                    stress: vec![0.0; self.config.num_elements],
                    deformation_energy: 0.0,
                    max_safe_deformation: 1.0,
                };
            }
            FlexibilityModel::Elastic {
                young_modulus,
                poisson_ratio,
                thickness,
            } => {
                // Calculate strain and stress for elastic deformation
                let strain = self.calculate_strain(curvature, *thickness);
                let stress = self.calculate_stress(&strain, *young_modulus);

                let mut deformation = DeformationState {
                    curvature_radius: if curvature > 0.0 {
                        Some(1.0 / curvature)
                    } else {
                        None
                    },
                    strain: strain.clone(),
                    stress: stress.clone(),
                    deformation_energy: 0.0,
                    max_safe_deformation: 0.1, // 10% strain limit
                };

                deformation.calculate_energy();
                self.geometry_state.deformation = deformation;
            }
            FlexibilityModel::FluidFilled {
                fluid_bulk_modulus,
                membrane_tension,
            } => {
                // Simplified fluid-filled model
                self.geometry_state.deformation = DeformationState {
                    curvature_radius: if curvature > 0.0 {
                        Some(1.0 / curvature)
                    } else {
                        None
                    },
                    strain: vec![curvature * 0.001; self.config.num_elements],
                    stress: vec![*membrane_tension; self.config.num_elements],
                    deformation_energy: membrane_tension * curvature,
                    max_safe_deformation: 0.2,
                };
            }
        }

        Ok(())
    }

    /// Calculate strain from curvature
    fn calculate_strain(&self, curvature: f64, thickness: f64) -> Vec<f64> {
        // Simple bending strain calculation
        vec![curvature * thickness / 2.0; self.config.num_elements]
    }

    /// Calculate stress from strain
    fn calculate_stress(&self, strain: &[f64], young_modulus: f64) -> Vec<f64> {
        strain.iter().map(|&s| s * young_modulus).collect()
    }

    /// Get current geometry state
    pub fn geometry_state(&self) -> &GeometryState {
        &self.geometry_state
    }

    /// Get configuration
    pub fn config(&self) -> &FlexibleTransducerConfig {
        &self.config
    }

    /// Get calibration confidence
    pub fn calibration_confidence(&self) -> f64 {
        self.calibration_processor.get_confidence()
    }
}

impl Source for FlexibleTransducerArray {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Add mask for each element position
        for position in self.geometry_state.element_positions.rows() {
            // Find nearest grid point
            let i = ((position[0] + grid.nx as f64 * grid.dx / 2.0) / grid.dx).round() as usize;
            let j = ((position[1] + grid.ny as f64 * grid.dy / 2.0) / grid.dy).round() as usize;
            let k = ((position[2] + grid.nz as f64 * grid.dz / 2.0) / grid.dz).round() as usize;

            // Check bounds and set mask
            if i < grid.nx && j < grid.ny && k < grid.nz {
                mask[[i, j, k]] = 1.0;
            }
        }

        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.geometry_state
            .element_positions
            .rows()
            .into_iter()
            .map(|row| (row[0], row[1], row[2]))
            .collect()
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }
}
