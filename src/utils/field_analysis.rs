//! Field Analysis and Beam Pattern Calculation Module
//!
//! This module provides comprehensive field analysis tools equivalent to k-Wave's 
//! beam pattern calculation and field analysis functions.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for field analysis functions
//! - **DRY**: Reusable field analysis implementations
//! - **Zero-Copy**: Uses ArrayView and efficient iterators
//! - **KISS**: Clear, well-documented field analysis interfaces
//!
//! # Literature References
//! - O'Neil (1949): "Theory of focusing radiators"
//! - Cobbold (2007): "Foundations of Biomedical Ultrasound"
//! - Treeby & Cox (2010): "k-Wave: MATLAB toolbox"
//! - Szabo (2014): "Diagnostic Ultrasound Imaging"

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array1, Array2, ArrayView2, ArrayView3, Zip};
use std::f64::consts::PI;

/// Beam pattern analysis configuration
#[derive(Debug, Clone)]
pub struct BeamPatternConfig {
    /// Analysis frequency (Hz)
    pub frequency: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Far-field distance calculation method
    pub far_field_method: FarFieldMethod,
    /// Angular resolution (radians)
    pub angular_resolution: f64,
}

/// Far-field calculation methods
#[derive(Debug, Clone)]
pub enum FarFieldMethod {
    /// Fresnel approximation
    Fresnel,
    /// Fraunhofer approximation
    Fraunhofer,
    /// Full calculation
    Exact,
}

/// Field analysis metrics
#[derive(Debug, Clone)]
pub struct FieldMetrics {
    /// Peak pressure location [x, y, z] (m)
    pub peak_location: [f64; 3],
    /// Peak pressure value (Pa)
    pub peak_pressure: f64,
    /// Focal depth (m)
    pub focal_depth: f64,
    /// Beam width at focus (m)
    pub beam_width_focus: f64,
    /// Depth of field (m)
    pub depth_of_field: f64,
    /// Total acoustic power (W)
    pub total_power: f64,
    /// Beam efficiency
    pub beam_efficiency: f64,
}

/// Comprehensive field analyzer
pub struct FieldAnalyzer {
    config: BeamPatternConfig,
}

impl FieldAnalyzer {
    /// Create new field analyzer
    pub fn new(config: BeamPatternConfig) -> Self {
        Self { config }
    }

    /// Calculate comprehensive beam pattern for arbitrary source distribution
    pub fn calculate_beam_pattern(
        &self,
        source_field: ArrayView3<f64>,
        grid: &Grid,
        analysis_points: &[[f64; 3]],
    ) -> KwaversResult<Array1<f64>> {
        let wavelength = self.config.sound_speed / self.config.frequency;
        let wavenumber = 2.0 * PI / wavelength;
        
        let mut beam_pattern = Array1::zeros(analysis_points.len());
        
        Zip::indexed(beam_pattern.view_mut()).for_each(|idx, pattern_value| {
            let analysis_point = analysis_points[idx];
            let mut field_sum = 0.0;
            
            // Integrate over all source points
            Zip::indexed(source_field).for_each(|(i, j, k), &source_amplitude| {
                if source_amplitude.abs() > 1e-12 {
                    let source_pos = [
                        grid.x_coordinates()[i],
                        grid.y_coordinates()[j], 
                        grid.z_coordinates()[k],
                    ];
                    
                    let distance = Self::euclidean_distance(&source_pos, &analysis_point);
                    
                    // Apply appropriate approximation
                    let field_contribution = match self.config.far_field_method {
                        FarFieldMethod::Fraunhofer => {
                            self.fraunhofer_contribution(source_amplitude, &source_pos, &analysis_point, wavenumber)
                        }
                        FarFieldMethod::Fresnel => {
                            self.fresnel_contribution(source_amplitude, &source_pos, &analysis_point, wavenumber)
                        }
                        FarFieldMethod::Exact => {
                            self.exact_contribution(source_amplitude, distance, wavenumber)
                        }
                    };
                    
                    field_sum += field_contribution;
                }
            });
            
            *pattern_value = field_sum.abs();
        });
        
        Ok(beam_pattern)
    }

    /// Calculate field metrics from pressure field
    pub fn analyze_field_metrics(
        &self,
        pressure_field: ArrayView3<f64>,
        grid: &Grid,
    ) -> KwaversResult<FieldMetrics> {
        // Find peak pressure location
        let (peak_location, peak_pressure) = self.find_peak_location(pressure_field, grid)?;
        
        // Calculate focal depth
        let focal_depth = peak_location[2]; // Assuming z is depth
        
        // Calculate beam width at focus
        let beam_width_focus = self.calculate_beam_width_at_depth(pressure_field, grid, focal_depth)?;
        
        // Calculate depth of field (FWHM region along axis)
        let depth_of_field = self.calculate_depth_of_field(pressure_field, grid, &peak_location)?;
        
        // Calculate total acoustic power
        let total_power = self.calculate_total_power(pressure_field, grid)?;
        
        // Calculate beam efficiency (power in focal region / total power)
        let beam_efficiency = self.calculate_beam_efficiency(pressure_field, grid, &peak_location)?;
        
        Ok(FieldMetrics {
            peak_location,
            peak_pressure,
            focal_depth,
            beam_width_focus,
            depth_of_field,
            total_power,
            beam_efficiency,
        })
    }

    /// Calculate directivity pattern for array transducers
    pub fn calculate_array_directivity(
        &self,
        element_positions: &[[f64; 3]],
        element_phases: &[f64],
        element_amplitudes: &[f64],
        steering_angles: &[f64; 2], // [azimuth, elevation] in radians
    ) -> KwaversResult<Array2<f64>> {
        let wavelength = self.config.sound_speed / self.config.frequency;
        let wavenumber = 2.0 * PI / wavelength;
        
        let n_azimuth = (2.0 * PI / self.config.angular_resolution) as usize;
        let n_elevation = (PI / self.config.angular_resolution) as usize;
        
        let mut directivity = Array2::zeros((n_azimuth, n_elevation));
        
        Zip::indexed(directivity.view_mut()).for_each(|(az_idx, el_idx), pattern_value| {
            let azimuth = (az_idx as f64) * self.config.angular_resolution - PI;
            let elevation = (el_idx as f64) * self.config.angular_resolution - PI/2.0;
            
            let direction = [
                azimuth.cos() * elevation.cos(),
                azimuth.sin() * elevation.cos(),
                elevation.sin(),
            ];
            
            let mut array_factor = 0.0_f64;
            
            // Sum contributions from all elements
            for (i, &element_pos) in element_positions.iter().enumerate() {
                let phase_delay = wavenumber * Self::dot_product(&element_pos, &direction);
                let total_phase = element_phases[i] + phase_delay;
                let contribution = element_amplitudes[i] * total_phase.cos();
                array_factor += contribution;
            }
            
            *pattern_value = array_factor.abs();
        });
        
        Ok(directivity)
    }

    /// Calculate near-field to far-field transformation
    pub fn near_to_far_field_transform(
        &self,
        near_field: ArrayView2<f64>, // 2D field at measurement plane
        measurement_grid: &Grid,
        far_field_distances: &[f64],
    ) -> KwaversResult<Array2<f64>> {
        let wavelength = self.config.sound_speed / self.config.frequency;
        let wavenumber = 2.0 * PI / wavelength;
        
        let n_angles = (2.0 * PI / self.config.angular_resolution) as usize;
        let mut far_field = Array2::zeros((far_field_distances.len(), n_angles));
        
        Zip::indexed(far_field.view_mut()).for_each(|(dist_idx, angle_idx), field_value| {
            let distance = far_field_distances[dist_idx];
            let angle = (angle_idx as f64) * self.config.angular_resolution;
            
            let direction = [angle.cos(), angle.sin(), 0.0];
            let mut field_sum = 0.0;
            
                         // Integrate over measurement plane
             Zip::indexed(near_field).for_each(|(i, j), &field_amplitude| {
                 let source_pos = [
                     measurement_grid.x_coordinates()[i],
                     measurement_grid.y_coordinates()[j],
                     0.0,
                 ];
                
                let phase = wavenumber * Self::dot_product(&source_pos, &direction);
                field_sum += field_amplitude * phase.cos();
            });
            
            // Apply far-field scaling
            *field_value = field_sum / distance;
        });
        
        Ok(far_field)
    }

    /// Calculate field uniformity metrics
    pub fn calculate_field_uniformity(
        &self,
        pressure_field: ArrayView3<f64>,
        analysis_region: &[[f64; 3]; 2], // [min_corner, max_corner]
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let mut values_in_region = Vec::new();
        
        Zip::indexed(pressure_field).for_each(|(i, j, k), &pressure| {
            let pos = [
                grid.x_coordinates()[i],
                grid.y_coordinates()[j],
                grid.z_coordinates()[k],
            ];
            
            if Self::point_in_box(&pos, analysis_region) {
                values_in_region.push(pressure.abs());
            }
        });
        
        if values_in_region.is_empty() {
            return Ok(0.0);
        }
        
        let mean = values_in_region.iter().sum::<f64>() / values_in_region.len() as f64;
        let variance = values_in_region.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values_in_region.len() as f64;
        
        let coefficient_of_variation = variance.sqrt() / mean;
        let uniformity = (1.0 - coefficient_of_variation).max(0.0);
        
        Ok(uniformity)
    }

    // Private helper methods
    
    fn fraunhofer_contribution(
        &self,
        amplitude: f64,
        source_pos: &[f64; 3],
        target_pos: &[f64; 3],
        wavenumber: f64,
    ) -> f64 {
        let direction = Self::normalize_vector(&Self::subtract_vectors(target_pos, source_pos));
        let phase = wavenumber * Self::dot_product(source_pos, &direction);
        amplitude * phase.cos()
    }

    fn fresnel_contribution(
        &self,
        amplitude: f64,
        source_pos: &[f64; 3],
        target_pos: &[f64; 3],
        wavenumber: f64,
    ) -> f64 {
        let distance = Self::euclidean_distance(source_pos, target_pos);
        let phase = wavenumber * distance;
        amplitude * phase.cos() / distance
    }

    fn exact_contribution(
        &self,
        amplitude: f64,
        distance: f64,
        wavenumber: f64,
    ) -> f64 {
        let phase = wavenumber * distance;
        amplitude * phase.cos() / distance
    }

    fn find_peak_location(
        &self,
        field: ArrayView3<f64>,
        grid: &Grid,
    ) -> KwaversResult<([f64; 3], f64)> {
        let mut max_pressure = 0.0;
        let mut max_indices = (0, 0, 0);
        
        Zip::indexed(field).for_each(|(i, j, k), &pressure| {
            if pressure.abs() > max_pressure {
                max_pressure = pressure.abs();
                max_indices = (i, j, k);
            }
        });
        
        let location = [
            grid.x_coordinates()[max_indices.0],
            grid.y_coordinates()[max_indices.1],
            grid.z_coordinates()[max_indices.2],
        ];
        
        Ok((location, max_pressure))
    }

    fn calculate_beam_width_at_depth(
        &self,
        field: ArrayView3<f64>,
        grid: &Grid,
        depth: f64,
    ) -> KwaversResult<f64> {
        // Find closest z-index to depth
        let z_idx = grid.z_coordinates().iter()
            .enumerate()
            .min_by(|(_, &z1), (_, &z2)| {
                (z1 - depth).abs().partial_cmp(&(z2 - depth).abs()).unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        // Extract 2D slice at this depth
        let slice = field.slice(ndarray::s![.., .., z_idx]);
        
        // Find beam width (FWHM) in this slice
        self.calculate_fwhm_width(&slice, grid)
    }

    fn calculate_fwhm_width(
        &self,
        field_2d: &ArrayView2<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        // Find maximum in the slice
        let max_val = field_2d.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        let half_max = max_val * 0.5;
        
        // Find FWHM by measuring width at half maximum
        let mut width_measurements = Vec::new();
        
        for j in 0..grid.ny {
            let mut first_half_max = None;
            let mut last_half_max = None;
            
            for i in 0..grid.nx {
                if field_2d[[i, j]].abs() >= half_max {
                    if first_half_max.is_none() {
                        first_half_max = Some(i);
                    }
                    last_half_max = Some(i);
                }
            }
            
            if let (Some(first), Some(last)) = (first_half_max, last_half_max) {
                let width = (last - first) as f64 * grid.dx;
                width_measurements.push(width);
            }
        }
        
        if width_measurements.is_empty() {
            Ok(0.0)
        } else {
            Ok(width_measurements.iter().sum::<f64>() / width_measurements.len() as f64)
        }
    }

    fn calculate_depth_of_field(
        &self,
        field: ArrayView3<f64>,
        grid: &Grid,
        peak_location: &[f64; 3],
    ) -> KwaversResult<f64> {
        // Extract axial profile through peak
        let x_idx = grid.x_coordinates().iter()
            .position(|&x| (x - peak_location[0]).abs() < grid.dx / 2.0)
            .unwrap_or(grid.nx / 2);
        let y_idx = grid.y_coordinates().iter()
            .position(|&y| (y - peak_location[1]).abs() < grid.dy / 2.0)
            .unwrap_or(grid.ny / 2);
        
        let axial_profile: Array1<f64> = field.slice(ndarray::s![x_idx, y_idx, ..]).to_owned();
        
        // Find FWHM along axial direction
        let max_val = axial_profile.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        let half_max = max_val * 0.5;
        
        let mut first_half_max = None;
        let mut last_half_max = None;
        
        for (k, &val) in axial_profile.iter().enumerate() {
            if val.abs() >= half_max {
                if first_half_max.is_none() {
                    first_half_max = Some(k);
                }
                last_half_max = Some(k);
            }
        }
        
        if let (Some(first), Some(last)) = (first_half_max, last_half_max) {
            Ok((last - first) as f64 * grid.dz)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_total_power(
        &self,
        field: ArrayView3<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let volume_element = grid.dx * grid.dy * grid.dz;
        let impedance = 1500.0 * 1000.0; // Approximate water impedance
        
        let power = field.iter()
            .map(|&pressure| pressure.powi(2) / impedance)
            .sum::<f64>() * volume_element;
        
        Ok(power)
    }

    fn calculate_beam_efficiency(
        &self,
        field: ArrayView3<f64>,
        grid: &Grid,
        peak_location: &[f64; 3],
    ) -> KwaversResult<f64> {
        let total_power = self.calculate_total_power(field, grid)?;
        
        // Define focal region as sphere around peak
        let focal_radius = 2.0 * (self.config.sound_speed / self.config.frequency); // 2 wavelengths
        let mut focal_power = 0.0;
        let volume_element = grid.dx * grid.dy * grid.dz;
        let impedance = 1500.0 * 1000.0;
        
        Zip::indexed(field).for_each(|(i, j, k), &pressure| {
            let pos = [
                grid.x_coordinates()[i],
                grid.y_coordinates()[j],
                grid.z_coordinates()[k],
            ];
            
            let distance = Self::euclidean_distance(&pos, peak_location);
            if distance <= focal_radius {
                focal_power += pressure.powi(2) / impedance * volume_element;
            }
        });
        
        if total_power > 0.0 {
            Ok(focal_power / total_power)
        } else {
            Ok(0.0)
        }
    }

    // Utility functions
    
    fn euclidean_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
    }

    fn dot_product(v1: &[f64; 3], v2: &[f64; 3]) -> f64 {
        v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    }

    fn subtract_vectors(v1: &[f64; 3], v2: &[f64; 3]) -> [f64; 3] {
        [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]
    }

    fn normalize_vector(v: &[f64; 3]) -> [f64; 3] {
        let magnitude = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
        if magnitude > 1e-12 {
            [v[0] / magnitude, v[1] / magnitude, v[2] / magnitude]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    fn point_in_box(point: &[f64; 3], bbox: &[[f64; 3]; 2]) -> bool {
        point[0] >= bbox[0][0] && point[0] <= bbox[1][0] &&
        point[1] >= bbox[0][1] && point[1] <= bbox[1][1] &&
        point[2] >= bbox[0][2] && point[2] <= bbox[1][2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    
    #[test]
    fn test_field_analyzer_creation() {
        let config = BeamPatternConfig {
            frequency: 1e6,
            sound_speed: 1500.0,
            far_field_method: FarFieldMethod::Fraunhofer,
            angular_resolution: 0.01,
        };
        
        let analyzer = FieldAnalyzer::new(config);
        assert_eq!(analyzer.config.frequency, 1e6);
    }
    
    #[test]
    fn test_beam_pattern_calculation() {
        let config = BeamPatternConfig {
            frequency: 1e6,
            sound_speed: 1500.0,
            far_field_method: FarFieldMethod::Fraunhofer,
            angular_resolution: 0.1,
        };
        
        let analyzer = FieldAnalyzer::new(config);
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3);
        let source_field = Array3::zeros((16, 16, 16));
        let analysis_points = vec![[0.0, 0.0, 1.0]];
        
        let result = analyzer.calculate_beam_pattern(source_field.view(), &grid, &analysis_points);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_field_metrics_calculation() {
        let config = BeamPatternConfig {
            frequency: 1e6,
            sound_speed: 1500.0,
            far_field_method: FarFieldMethod::Exact,
            angular_resolution: 0.1,
        };
        
        let analyzer = FieldAnalyzer::new(config);
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3);
        let mut pressure_field = Array3::zeros((16, 16, 16));
        pressure_field[[8, 8, 8]] = 1.0; // Peak at center
        
        let result = analyzer.analyze_field_metrics(pressure_field.view(), &grid);
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(metrics.peak_pressure > 0.0);
    }
}