//! Hemisphere geometry and element placement

use super::constants::*;
use super::element::ElementConfiguration;
use crate::error::{ConfigError, KwaversError, KwaversResult};
use std::f64::consts::PI;

/// Hemisphere geometry definition
#[derive(Debug, Clone)]
pub struct HemisphereGeometry {
    /// Radius of hemisphere (m)
    pub radius: f64,
    /// F-number (focal_length/aperture)
    pub f_number: f64,
    /// Aperture diameter (m)
    pub aperture: f64,
    /// Focal length (m)
    pub focal_length: f64,
}

impl HemisphereGeometry {
    /// Create new hemisphere geometry
    pub fn new(radius: f64) -> KwaversResult<Self> {
        if radius <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "radius".to_string(),
                value: radius.to_string(),
                constraint: "must be positive".to_string(),
            }));
        }

        let f_number = 0.9; // Typical for hemispherical arrays
        let aperture = 2.0 * radius;
        let focal_length = f_number * aperture;

        Ok(Self {
            radius,
            f_number,
            aperture,
            focal_length,
        })
    }

    /// Get geometric focus point
    pub fn focal_point(&self) -> [f64; 3] {
        [0.0, 0.0, self.focal_length]
    }
}

/// Element placement algorithms
pub struct ElementPlacement;

impl ElementPlacement {
    /// Generate element positions on hemisphere
    pub fn generate_elements(
        geometry: &HemisphereGeometry,
        num_elements: usize,
    ) -> KwaversResult<Vec<ElementConfiguration>> {
        let mut elements = Vec::with_capacity(num_elements);

        // Use spiral placement for uniform distribution
        let golden_angle = PI * (3.0 - (5.0_f64).sqrt());

        for i in 0..num_elements {
            let y = 1.0 - (i as f64 / (num_elements - 1) as f64);
            let radius_at_y = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f64;

            let x = theta.cos() * radius_at_y;
            let z = theta.sin() * radius_at_y;

            // Scale to hemisphere radius
            let position = [
                x * geometry.radius,
                y * geometry.radius,
                z * geometry.radius,
            ];

            // Normal points inward (toward focus)
            let normal = [-x, -y, -z];

            // Element radius based on density
            let element_radius = geometry.radius / (num_elements as f64).sqrt() * 0.4;

            elements.push(ElementConfiguration::new(position, normal, element_radius));
        }

        Ok(elements)
    }

    /// Generate sparse element distribution
    pub fn generate_sparse(
        geometry: &HemisphereGeometry,
        density_factor: f64,
    ) -> KwaversResult<Vec<ElementConfiguration>> {
        let base_elements =
            (4.0 * PI * geometry.radius * geometry.radius * MAX_ELEMENT_DENSITY) as usize;
        let num_elements = (base_elements as f64 * density_factor) as usize;
        Self::generate_elements(geometry, num_elements)
    }
}
