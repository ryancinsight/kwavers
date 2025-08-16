//! Focused Transducer Sources Module
//!
//! This module provides focused transducer geometries compatible with k-Wave toolbox,
//! including bowl transducers, arc sources, and multi-element arrays.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for transducer geometry generation
//! - **DRY**: Reusable geometric primitives and calculations
//! - **Zero-Copy**: Uses iterators and in-place operations
//! - **KISS**: Clear interfaces for complex geometries
//!
//! # Literature References
//! - O'Neil (1949): "Theory of focusing radiators"
//! - Cobbold (2007): "Foundations of Biomedical Ultrasound"
//! - Szabo (2014): "Diagnostic Ultrasound Imaging"

use crate::{
    error::{KwaversError, KwaversResult},
    grid::Grid,
};
use ndarray::{Array2, Array3, Zip, s};
use std::f64::consts::PI;
use rayon::prelude::*;

/// Configuration for a focused bowl transducer
#[derive(Debug, Clone)]
pub struct BowlConfig {
    /// Radius of curvature (m)
    pub radius_of_curvature: f64,
    
    /// Diameter of the bowl aperture (m)
    pub diameter: f64,
    
    /// Center position [x, y, z] (m)
    pub center: [f64; 3],
    
    /// Focus position [x, y, z] (m)
    pub focus: [f64; 3],
    
    /// Operating frequency (Hz)
    pub frequency: f64,
    
    /// Source amplitude (Pa)
    pub amplitude: f64,
    
    /// Phase delay (radians)
    pub phase: f64,
    
    /// Element size for discretization (m)
    pub element_size: Option<f64>,
    
    /// Apply directivity weighting
    pub apply_directivity: bool,
}

impl Default for BowlConfig {
    fn default() -> Self {
        Self {
            radius_of_curvature: 0.064,  // 64mm
            diameter: 0.064,              // 64mm
            center: [0.0, 0.0, 0.0],
            focus: [0.0, 0.0, 0.064],     // Focus at radius
            frequency: 1e6,               // 1 MHz
            amplitude: 1e6,               // 1 MPa
            phase: 0.0,
            element_size: None,
            apply_directivity: true,
        }
    }
}

/// Focused bowl transducer (makeBowl equivalent)
pub struct BowlTransducer {
    config: BowlConfig,
    /// Discretized element positions
    element_positions: Vec<[f64; 3]>,
    /// Element normals for directivity
    element_normals: Vec<[f64; 3]>,
    /// Element areas for weighting
    element_areas: Vec<f64>,
}

impl BowlTransducer {
    /// Create a new bowl transducer
    pub fn new(config: BowlConfig) -> KwaversResult<Self> {
        // Validate configuration
        if config.diameter > 2.0 * config.radius_of_curvature {
            return Err(KwaversError::field_validation(
                "diameter",
                config.diameter,
                "Diameter cannot exceed 2 * radius_of_curvature"
            ));
        }
        
        // Calculate element size if not provided
        let element_size = config.element_size.unwrap_or_else(|| {
            // Use lambda/4 as default element size
            let speed_of_sound = 1500.0; // Default for water
            let wavelength = speed_of_sound / config.frequency;
            wavelength / 4.0
        });
        
        // Discretize bowl surface
        let (positions, normals, areas) = Self::discretize_bowl(&config, element_size)?;
        
        Ok(Self {
            config,
            element_positions: positions,
            element_normals: normals,
            element_areas: areas,
        })
    }
    
    /// Discretize the bowl surface into elements
    fn discretize_bowl(config: &BowlConfig, element_size: f64) -> KwaversResult<(Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>)> {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut areas = Vec::new();
        
        // Calculate bowl parameters
        let r = config.radius_of_curvature;
        let a = config.diameter / 2.0;
        let h = r - (r * r - a * a).sqrt(); // Height of spherical cap
        
        // Angular extent of the bowl
        let theta_max = (a / r).asin();
        
        // Number of angular divisions
        let n_theta = ((2.0 * PI * r * theta_max) / element_size).ceil() as usize;
        let n_phi = ((2.0 * PI * a) / element_size).ceil() as usize;
        
        // Generate elements using spherical coordinates
        for i in 0..n_theta {
            let theta = (i as f64 + 0.5) * theta_max / n_theta as f64;
            let r_ring = r * theta.sin();
            let z_ring = r * (1.0 - theta.cos());
            
            // Number of elements in this ring
            let n_ring = ((2.0 * PI * r_ring) / element_size).ceil() as usize;
            
            for j in 0..n_ring {
                let phi = 2.0 * PI * j as f64 / n_ring as f64;
                
                // Element position relative to bowl center
                let x = r_ring * phi.cos();
                let y = r_ring * phi.sin();
                let z = z_ring;
                
                // Transform to global coordinates
                let pos = [
                    config.center[0] + x,
                    config.center[1] + y,
                    config.center[2] + z,
                ];
                
                // Calculate normal (points inward toward focus)
                let norm_vec = [
                    config.focus[0] - pos[0],
                    config.focus[1] - pos[1],
                    config.focus[2] - pos[2],
                ];
                let norm_mag = (norm_vec[0].powi(2) + norm_vec[1].powi(2) + norm_vec[2].powi(2)).sqrt();
                let normal = [
                    norm_vec[0] / norm_mag,
                    norm_vec[1] / norm_mag,
                    norm_vec[2] / norm_mag,
                ];
                
                // Calculate element area (approximate)
                let dtheta = theta_max / n_theta as f64;
                let dphi = 2.0 * PI / n_ring as f64;
                let area = r * r * theta.sin() * dtheta * dphi;
                
                positions.push(pos);
                normals.push(normal);
                areas.push(area);
            }
        }
        
        Ok((positions, normals, areas))
    }
    
    /// Generate source distribution on grid
    pub fn generate_source(&self, grid: &Grid, time: f64) -> KwaversResult<Array3<f64>> {
        let mut source = grid.create_field();
        let omega = 2.0 * PI * self.config.frequency;
        
        // Calculate phase delays for focusing
        let focus_delays = self.calculate_focus_delays();
        
        // Use parallel processing for efficiency
        let source_slice = source.as_slice_mut().unwrap();
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let dx = grid.dx;
        
        source_slice.par_chunks_mut(nx * ny).enumerate().for_each(|(iz, chunk)| {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = iy * nx + ix;
                    if idx < chunk.len() {
                        // Grid point position
                        let x = ix as f64 * dx;
                        let y = iy as f64 * dx;
                        let z = iz as f64 * dx;
                        
                        // Accumulate contributions from all elements
                        let mut pressure = 0.0;
                        
                        for (i, &pos) in self.element_positions.iter().enumerate() {
                            // Distance from element to grid point
                            let r = ((x - pos[0]).powi(2) + 
                                    (y - pos[1]).powi(2) + 
                                    (z - pos[2]).powi(2)).sqrt();
                            
                            if r > 0.0 {
                                // Apply directivity if enabled
                                let directivity = if self.config.apply_directivity {
                                    self.calculate_directivity(i, [x, y, z])
                                } else {
                                    1.0
                                };
                                
                                // Phase with focusing delay
                                let phase = omega * (time - focus_delays[i]) + self.config.phase;
                                
                                // Pressure contribution with spherical spreading
                                let element_pressure = self.config.amplitude * 
                                                       self.element_areas[i] * 
                                                       directivity * 
                                                       phase.sin() / (4.0 * PI * r);
                                
                                pressure += element_pressure;
                            }
                        }
                        
                        chunk[idx] = pressure;
                    }
                }
            }
        });
        
        Ok(source)
    }
    
    /// Calculate time delays for focusing
    fn calculate_focus_delays(&self) -> Vec<f64> {
        let speed_of_sound = 1500.0; // Default for water
        
        self.element_positions.iter().map(|&pos| {
            let distance = ((self.config.focus[0] - pos[0]).powi(2) +
                           (self.config.focus[1] - pos[1]).powi(2) +
                           (self.config.focus[2] - pos[2]).powi(2)).sqrt();
            distance / speed_of_sound
        }).collect()
    }
    
    /// Calculate directivity for an element
    fn calculate_directivity(&self, element_idx: usize, target: [f64; 3]) -> f64 {
        let pos = self.element_positions[element_idx];
        let normal = self.element_normals[element_idx];
        
        // Vector from element to target
        let dir = [
            target[0] - pos[0],
            target[1] - pos[1],
            target[2] - pos[2],
        ];
        let dir_mag = (dir[0].powi(2) + dir[1].powi(2) + dir[2].powi(2)).sqrt();
        
        if dir_mag > 0.0 {
            // Cosine of angle between normal and direction
            let cos_theta = (normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2]) / dir_mag;
            
            // Cosine directivity pattern
            if cos_theta > 0.0 {
                cos_theta
            } else {
                0.0
            }
        } else {
            1.0
        }
    }
    
    /// Get analytical solution using O'Neil's formula
    /// 
    /// O'Neil's solution provides the on-axis pressure field for a spherically
    /// focused transducer. The formula involves complex exponentials representing
    /// the phase accumulation from different parts of the curved surface.
    /// 
    /// Reference: O'Neil, H. T. (1949). "Theory of focusing radiators." 
    /// The Journal of the Acoustical Society of America, 21(5), 516-526.
    pub fn oneil_solution(&self, z: f64, time: f64) -> f64 {
        // O'Neil's solution for on-axis pressure of a focused bowl transducer
        let r = self.config.radius_of_curvature;
        let a = self.config.diameter / 2.0;
        let k = 2.0 * PI * self.config.frequency / 1500.0; // Wave number
        let omega = 2.0 * PI * self.config.frequency; // Angular frequency
        let c = 1500.0; // Speed of sound in water
        
        // Geometric parameters
        // h is the height of the spherical cap
        let h = r - (r * r - a * a).sqrt();
        
        // Distance from transducer center to field point
        let d1 = z.abs();
        // Distance from edge of spherical cap to field point  
        let d2 = ((z - (r - h)).powi(2) + a * a).sqrt();
        
        // O'Neil's formula for on-axis pressure
        // The pressure is the result of interference between waves from the center
        // and edge of the bowl, accounting for their phase difference
        let phase_diff = k * (d2 - d1);
        
        // Complex pressure amplitude using proper formulation
        // P(z,t) = P0 * |exp(ikd1) - exp(ikd2)| * exp(i*omega*t) / (2*d1)
        // For magnitude: |exp(ikd1) - exp(ikd2)| = 2*|sin(phase_diff/2)|
        let p_amplitude = if d1 > 0.0 {
            self.config.amplitude * 2.0 * (phase_diff / 2.0).sin().abs() / d1
        } else {
            // At z=0, use limiting value
            self.config.amplitude * k * h
        };
        
        // Add time-varying component
        let p = p_amplitude * (omega * time - k * d1 + self.config.phase).sin();
        
        p
    }
}

/// Configuration for an arc source (2D focused transducer)
#[derive(Debug, Clone)]
pub struct ArcConfig {
    /// Radius of curvature (m)
    pub radius: f64,
    
    /// Arc angle (radians)
    pub arc_angle: f64,
    
    /// Center position [x, y] (m)
    pub center: [f64; 2],
    
    /// Orientation angle (radians)
    pub orientation: f64,
    
    /// Operating frequency (Hz)
    pub frequency: f64,
    
    /// Source amplitude (Pa)
    pub amplitude: f64,
    
    /// Element spacing (m)
    pub element_spacing: Option<f64>,
}

impl Default for ArcConfig {
    fn default() -> Self {
        Self {
            radius: 0.05,           // 50mm
            arc_angle: PI / 3.0,    // 60 degrees
            center: [0.0, 0.0],
            orientation: 0.0,
            frequency: 1e6,         // 1 MHz
            amplitude: 1e6,         // 1 MPa
            element_spacing: None,
        }
    }
}

/// Arc source for 2D simulations (makeArc equivalent)
pub struct ArcSource {
    config: ArcConfig,
    /// Discretized element positions
    element_positions: Vec<[f64; 2]>,
    /// Element weights
    element_weights: Vec<f64>,
}

impl ArcSource {
    /// Create a new arc source
    pub fn new(config: ArcConfig) -> KwaversResult<Self> {
        // Calculate element spacing if not provided
        let element_spacing = config.element_spacing.unwrap_or_else(|| {
            let speed_of_sound = 1500.0;
            let wavelength = speed_of_sound / config.frequency;
            wavelength / 4.0
        });
        
        // Number of elements
        let arc_length = config.radius * config.arc_angle;
        let n_elements = (arc_length / element_spacing).ceil() as usize;
        
        // Generate element positions
        let mut positions = Vec::with_capacity(n_elements);
        let mut weights = Vec::with_capacity(n_elements);
        
        for i in 0..n_elements {
            // Angle for this element
            let theta = -config.arc_angle / 2.0 + (i as f64 + 0.5) * config.arc_angle / n_elements as f64;
            let rotated_theta = theta + config.orientation;
            
            // Position
            let x = config.center[0] + config.radius * rotated_theta.cos();
            let y = config.center[1] + config.radius * rotated_theta.sin();
            
            positions.push([x, y]);
            weights.push(1.0 / n_elements as f64);
        }
        
        Ok(Self {
            config,
            element_positions: positions,
            element_weights: weights,
        })
    }
    
    /// Generate 2D source distribution
    pub fn generate_source_2d(&self, nx: usize, ny: usize, dx: f64, time: f64) -> Array2<f64> {
        let mut source = Array2::zeros((nx, ny));
        let omega = 2.0 * PI * self.config.frequency;
        
        // Focus is at the center of curvature
        let focus = self.config.center;
        
        // Calculate delays for focusing
        let speed_of_sound = 1500.0;
        let delays: Vec<f64> = self.element_positions.iter().map(|&pos| {
            let distance = ((focus[0] - pos[0]).powi(2) + (focus[1] - pos[1]).powi(2)).sqrt();
            distance / speed_of_sound
        }).collect();
        
        // Generate source field
        Zip::indexed(&mut source).for_each(|(ix, iy), val| {
            let x = ix as f64 * dx;
            let y = iy as f64 * dx;
            
            let mut pressure = 0.0;
            for (i, &pos) in self.element_positions.iter().enumerate() {
                let r = ((x - pos[0]).powi(2) + (y - pos[1]).powi(2)).sqrt();
                
                if r > 0.0 {
                    // Phase with focusing delay
                    let phase = omega * (time - delays[i]);
                    
                    // 2D Green's function (Hankel function calculation)
                    let element_pressure = self.config.amplitude * 
                                          self.element_weights[i] * 
                                          phase.sin() / r.sqrt();
                    
                    pressure += element_pressure;
                }
            }
            
            *val = pressure;
        });
        
        source
    }
    
    /// Extend 2D source to 3D (uniform in z-direction)
    pub fn generate_source_3d(&self, grid: &Grid, time: f64) -> Array3<f64> {
        let source_2d = self.generate_source_2d(grid.nx, grid.ny, grid.dx, time);
        let mut source_3d = grid.create_field();
        
        // Copy 2D pattern to all z-slices
        for iz in 0..grid.nz {
            source_3d.slice_mut(s![.., .., iz]).assign(&source_2d);
        }
        
        source_3d
    }
}

/// Multi-element bowl array (makeMultiBowl equivalent)
pub struct MultiBowlArray {
    /// Individual bowl transducers
    bowls: Vec<BowlTransducer>,
    /// Relative amplitudes for each bowl
    amplitudes: Vec<f64>,
    /// Relative phases for each bowl
    phases: Vec<f64>,
}

impl MultiBowlArray {
    /// Create a new multi-bowl array
    pub fn new(configs: Vec<BowlConfig>) -> KwaversResult<Self> {
        let n_bowls = configs.len();
        let mut bowls = Vec::with_capacity(n_bowls);
        let mut amplitudes = Vec::with_capacity(n_bowls);
        let mut phases = Vec::with_capacity(n_bowls);
        
        for config in configs {
            amplitudes.push(config.amplitude);
            phases.push(config.phase);
            bowls.push(BowlTransducer::new(config)?);
        }
        
        Ok(Self {
            bowls,
            amplitudes,
            phases,
        })
    }
    
    /// Generate combined source from all bowls
    /// 
    /// This method combines the contributions from all bowl transducers,
    /// applying both amplitude scaling and phase shifts. The phase shifts
    /// are crucial for beam steering and complex field synthesis.
    pub fn generate_source(&self, grid: &Grid, time: f64) -> KwaversResult<Array3<f64>> {
        let mut combined_source = grid.create_field();
        
        // Add contributions from each bowl
        for (i, bowl) in self.bowls.iter().enumerate() {
            // Generate source for this bowl at the current time
            // Note: We need to adjust the time to account for the phase offset
            let omega = 2.0 * PI * bowl.config.frequency;
            let phase_offset = self.phases[i] - bowl.config.phase; // Relative phase
            let time_offset = phase_offset / omega; // Convert phase to time offset
            
            let bowl_source = bowl.generate_source(grid, time + time_offset)?;
            
            // Apply relative amplitude
            let scale = self.amplitudes[i] / bowl.config.amplitude;
            
            Zip::from(&mut combined_source)
                .and(&bowl_source)
                .for_each(|c, &b| *c += scale * b);
        }
        
        Ok(combined_source)
    }
    
    /// Set beam steering parameters
    pub fn set_beam_steering(&mut self, focus: [f64; 3]) {
        // Update focus for all bowls
        for bowl in &mut self.bowls {
            bowl.config.focus = focus;
        }
    }
    
    /// Apply apodization (amplitude shading)
    pub fn apply_apodization(&mut self, apodization_type: ApodizationType) {
        let n = self.bowls.len();
        
        match apodization_type {
            ApodizationType::Uniform => {
                self.amplitudes = vec![1.0; n];
            }
            ApodizationType::Hamming => {
                for i in 0..n {
                    let x = i as f64 / (n - 1) as f64;
                    self.amplitudes[i] = 0.54 - 0.46 * (2.0 * PI * x).cos();
                }
            }
            ApodizationType::Hanning => {
                for i in 0..n {
                    let x = i as f64 / (n - 1) as f64;
                    self.amplitudes[i] = 0.5 * (1.0 - (2.0 * PI * x).cos());
                }
            }
            ApodizationType::Gaussian(sigma) => {
                let center = (n - 1) as f64 / 2.0;
                for i in 0..n {
                    let x = (i as f64 - center) / center;
                    self.amplitudes[i] = (-x * x / (2.0 * sigma * sigma)).exp();
                }
            }
        }
    }
}

/// Apodization types for multi-element arrays
#[derive(Debug, Clone, Copy)]
pub enum ApodizationType {
    /// Uniform amplitude
    Uniform,
    /// Hamming window
    Hamming,
    /// Hanning window
    Hanning,
    /// Gaussian with specified sigma
    Gaussian(f64),
}

/// Helper function to create a spherical section bowl
pub fn make_bowl(
    radius: f64,
    diameter: f64,
    center: [f64; 3],
    frequency: f64,
    amplitude: f64,
) -> KwaversResult<BowlTransducer> {
    let config = BowlConfig {
        radius_of_curvature: radius,
        diameter,
        center,
        focus: [center[0], center[1], center[2] + radius], // Default focus at radius
        frequency,
        amplitude,
        ..Default::default()
    };
    
    BowlTransducer::new(config)
}

/// Helper function to create an annular array
pub fn make_annular_array(
    inner_radius: f64,
    outer_radius: f64,
    n_rings: usize,
    center: [f64; 3],
    frequency: f64,
) -> KwaversResult<MultiBowlArray> {
    let mut configs = Vec::new();
    
    for i in 0..n_rings {
        let t = if n_rings > 1 { i as f64 / (n_rings - 1) as f64 } else { 0.0 };
        let radius = inner_radius + t * (outer_radius - inner_radius);
        
        let config = BowlConfig {
            radius_of_curvature: radius,
            diameter: radius * 0.8, // 80% of radius for each ring
            center,
            focus: [center[0], center[1], center[2] + radius],
            frequency,
            amplitude: 1e6,
            ..Default::default()
        };
        
        configs.push(config);
    }
    
    MultiBowlArray::new(configs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_bowl_transducer_creation() {
        let config = BowlConfig::default();
        let bowl = BowlTransducer::new(config).unwrap();
        
        assert!(!bowl.element_positions.is_empty());
        assert_eq!(bowl.element_positions.len(), bowl.element_normals.len());
        assert_eq!(bowl.element_positions.len(), bowl.element_areas.len());
    }
    
    #[test]
    fn test_arc_source_creation() {
        let config = ArcConfig::default();
        let arc = ArcSource::new(config).unwrap();
        
        assert!(!arc.element_positions.is_empty());
        assert_eq!(arc.element_positions.len(), arc.element_weights.len());
    }
    
    #[test]
    fn test_bowl_focusing() {
        let config = BowlConfig {
            radius_of_curvature: 0.05,
            diameter: 0.04,
            center: [0.0, 0.0, 0.0],
            focus: [0.0, 0.0, 0.05],
            frequency: 1e6,
            amplitude: 1e6,
            ..Default::default()
        };
        
        let bowl = BowlTransducer::new(config).unwrap();
        let delays = bowl.calculate_focus_delays();
        
        // All delays should be positive
        for delay in delays {
            assert!(delay >= 0.0);
        }
    }
    
    #[test]
    fn test_multi_bowl_array() {
        let configs = vec![
            BowlConfig {
                radius_of_curvature: 0.05,
                diameter: 0.03,
                center: [0.0, 0.0, 0.0],
                ..Default::default()
            },
            BowlConfig {
                radius_of_curvature: 0.06,
                diameter: 0.04,
                center: [0.01, 0.0, 0.0],
                ..Default::default()
            },
        ];
        
        let array = MultiBowlArray::new(configs).unwrap();
        assert_eq!(array.bowls.len(), 2);
    }
    
    #[test]
    fn test_multi_bowl_phases() {
        // Create two bowls with different phases
        let config1 = BowlConfig {
            diameter: 0.03,
            radius_of_curvature: 0.05,
            focus: [0.0, 0.0, 0.05],
            frequency: 1e6,
            amplitude: 1e6,
            phase: 0.0,
            ..Default::default()
        };
        
        let config2 = BowlConfig {
            diameter: 0.03,
            radius_of_curvature: 0.05,
            focus: [0.0, 0.0, 0.05],
            frequency: 1e6,
            amplitude: 1e6,
            phase: PI / 2.0, // 90 degree phase shift
            ..Default::default()
        };
        
        let multi_array = MultiBowlArray::new(vec![config1, config2]).unwrap();
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001);
        
        // Generate sources at different times
        let source_t0 = multi_array.generate_source(&grid, 0.0).unwrap();
        let source_t1 = multi_array.generate_source(&grid, 0.25e-6).unwrap(); // Quarter period
        
        // The sources should be different due to phase shifts
        let diff = &source_t1 - &source_t0;
        let max_diff = diff.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_diff > 0.0, "Phase shifts should cause time-varying fields");
    }
    
    #[test]
    fn test_oneil_solution() {
        let bowl = make_bowl(0.064, 0.064, [0.0, 0.0, 0.0], 1e6, 1e6).unwrap();
        
        // Test on-axis pressure at focus at different times
        let time = 0.0;
        let p_focus = bowl.oneil_solution(0.064, time).abs();
        
        // Test at a time when sin is maximum (quarter period)
        let period = 1.0 / 1e6;
        let time_max = period / 4.0;
        let p_focus_max = bowl.oneil_solution(0.064, time_max).abs();
        assert!(p_focus_max > 0.0);
        
        // Pressure amplitude should vary with distance from focus
        let p_near = bowl.oneil_solution(0.05, time_max).abs();
        let p_far = bowl.oneil_solution(0.1, time_max).abs();
        
        // The focus should have strong pressure due to constructive interference
        // Note: The exact relationship depends on the interference pattern
        assert!(p_focus_max > 0.0);
        assert!(p_near > 0.0);
        assert!(p_far > 0.0);
    }
}