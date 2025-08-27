//! Bowl transducer implementation

use crate::{
    error::{KwaversError, KwaversResult},
    grid::Grid,
};
use ndarray::{Array2, Array3};
use std::f64::consts::PI;

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
}

impl BowlConfig {
    /// Create a new bowl configuration
    pub fn new(
        radius_of_curvature: f64,
        diameter: f64,
        center: [f64; 3],
        focus: [f64; 3],
        frequency: f64,
        amplitude: f64,
    ) -> Self {
        Self {
            radius_of_curvature,
            diameter,
            center,
            focus,
            frequency,
            amplitude,
            phase: 0.0,
            element_size: None,
        }
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> KwaversResult<()> {
        use crate::error::ValidationError;
        
        if self.radius_of_curvature <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::RangeValidation {
                field: "radius_of_curvature".to_string(),
                value: self.radius_of_curvature.to_string(),
                min: "0".to_string(),
                max: "inf".to_string(),
            }));
        }
        
        if self.diameter <= 0.0 || self.diameter > 2.0 * self.radius_of_curvature {
            return Err(KwaversError::Validation(ValidationError::RangeValidation {
                field: "diameter".to_string(),
                value: self.diameter.to_string(),
                min: "0".to_string(),
                max: (2.0 * self.radius_of_curvature).to_string(),
            }));
        }
        
        if self.frequency <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::RangeValidation {
                field: "frequency".to_string(),
                value: self.frequency.to_string(),
                min: "0".to_string(),
                max: "inf".to_string(),
            }));
        }
        
        Ok(())
    }
    
    /// Calculate the f-number (focal length / diameter)
    pub fn f_number(&self) -> f64 {
        let focal_length = (self.focus[0] - self.center[0]).powi(2)
            + (self.focus[1] - self.center[1]).powi(2)
            + (self.focus[2] - self.center[2]).powi(2);
        focal_length.sqrt() / self.diameter
    }
    
    /// Calculate the opening angle
    pub fn opening_angle(&self) -> f64 {
        2.0 * (self.diameter / (2.0 * self.radius_of_curvature)).asin()
    }
}

/// Bowl transducer implementation
pub struct BowlTransducer {
    config: BowlConfig,
    element_positions: Vec<[f64; 3]>,
    element_normals: Vec<[f64; 3]>,
    element_areas: Vec<f64>,
}

impl BowlTransducer {
    /// Create a new bowl transducer
    pub fn new(config: BowlConfig) -> KwaversResult<Self> {
        config.validate()?;
        
        let mut transducer = Self {
            config,
            element_positions: Vec::new(),
            element_normals: Vec::new(),
            element_areas: Vec::new(),
        };
        
        transducer.discretize_surface()?;
        Ok(transducer)
    }
    
    /// Discretize the bowl surface into elements
    fn discretize_surface(&mut self) -> KwaversResult<()> {
        let element_size = self.config.element_size
            .unwrap_or(self.config.diameter / 100.0);
        
        let n_theta = ((2.0 * PI * self.config.radius_of_curvature) / element_size) as usize;
        let n_phi = ((self.config.opening_angle() * self.config.radius_of_curvature) / element_size) as usize;
        
        for i in 0..n_theta {
            let theta = 2.0 * PI * i as f64 / n_theta as f64;
            
            for j in 0..n_phi {
                let phi = self.config.opening_angle() * j as f64 / n_phi as f64;
                
                // Calculate element position on bowl surface
                let x = self.config.radius_of_curvature * phi.sin() * theta.cos();
                let y = self.config.radius_of_curvature * phi.sin() * theta.sin();
                let z = self.config.radius_of_curvature * (1.0 - phi.cos());
                
                let position = [
                    self.config.center[0] + x,
                    self.config.center[1] + y,
                    self.config.center[2] + z,
                ];
                
                // Calculate normal vector (pointing toward focus)
                let dx = self.config.focus[0] - position[0];
                let dy = self.config.focus[1] - position[1];
                let dz = self.config.focus[2] - position[2];
                let norm = (dx * dx + dy * dy + dz * dz).sqrt();
                
                let normal = [dx / norm, dy / norm, dz / norm];
                
                // Calculate element area
                let area = element_size * element_size;
                
                self.element_positions.push(position);
                self.element_normals.push(normal);
                self.element_areas.push(area);
            }
        }
        
        Ok(())
    }
    
    /// Get the element positions
    pub fn element_positions(&self) -> &[[f64; 3]] {
        &self.element_positions
    }
    
    /// Get the element normals
    pub fn element_normals(&self) -> &[[f64; 3]] {
        &self.element_normals
    }
    
    /// Get the element areas
    pub fn element_areas(&self) -> &[f64] {
        &self.element_areas
    }
    
    /// Calculate the pressure at a given point
    pub fn pressure_at_point(&self, point: [f64; 3], time: f64) -> f64 {
        let mut total_pressure = 0.0;
        let k = 2.0 * PI * self.config.frequency / 1500.0; // Assume water sound speed
        
        for (i, &pos) in self.element_positions.iter().enumerate() {
            let dx = point[0] - pos[0];
            let dy = point[1] - pos[1];
            let dz = point[2] - pos[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            
            if r > 1e-10 {
                let phase = k * r - 2.0 * PI * self.config.frequency * time + self.config.phase;
                let directivity = self.calculate_directivity(i, point);
                total_pressure += self.config.amplitude * self.element_areas[i] * directivity * phase.cos() / r;
            }
        }
        
        total_pressure
    }
    
    /// Calculate directivity for an element
    fn calculate_directivity(&self, element_index: usize, point: [f64; 3]) -> f64 {
        let normal = &self.element_normals[element_index];
        let pos = &self.element_positions[element_index];
        
        let dx = point[0] - pos[0];
        let dy = point[1] - pos[1];
        let dz = point[2] - pos[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        
        if r > 1e-10 {
            let cos_angle = (normal[0] * dx + normal[1] * dy + normal[2] * dz) / r;
            cos_angle.max(0.0)
        } else {
            1.0
        }
    }
}