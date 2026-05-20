//! Hemisphere geometry and element placement

use super::constants::MAX_ELEMENT_DENSITY;
use super::element::ElementConfiguration;
use crate::core::error::{ConfigError, KwaversError, KwaversResult};
use crate::domain::source::transducers::focused::{SphericalCapConfig, SphericalCapLayout};
use std::f64::consts::PI;

/// Hemisphere geometry definition
#[derive(Debug, Clone)]
pub struct HemisphereGeometry {
    /// Radius of hemisphere (m)
    pub radius: f64,
    /// F-number (`focal_length/aperture`)
    pub f_number: f64,
    /// Aperture diameter (m)
    pub aperture: f64,
    /// Focal length (m)
    pub focal_length: f64,
}

impl HemisphereGeometry {
    /// Create new hemisphere geometry
    /// # Errors
    /// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
    ///
    pub fn new(radius: f64) -> KwaversResult<Self> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "radius".to_owned(),
                value: radius.to_string(),
                constraint: "must be positive and finite".to_owned(),
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn focal_point(&self) -> [f64; 3] {
        [0.0, 0.0, self.focal_length]
    }
}

/// Element placement algorithms
#[derive(Debug)]
pub struct ElementPlacement;

impl ElementPlacement {
    /// Generate element positions on hemisphere
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn generate_elements(
        geometry: &HemisphereGeometry,
        num_elements: usize,
    ) -> KwaversResult<Vec<ElementConfiguration>> {
        let layout = SphericalCapLayout::new(SphericalCapConfig::hemisphere(
            num_elements,
            geometry.radius,
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ))?;
        let element_radius = geometry.radius / (num_elements as f64).sqrt() * 0.4;

        Ok(layout
            .elements()
            .iter()
            .map(|element| {
                ElementConfiguration::new(
                    element.position_m,
                    element.normal_to_focus,
                    element_radius,
                )
            })
            .collect())
    }

    /// Generate sparse element distribution
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_elements_use_source_domain_hemisphere() {
        let geometry = HemisphereGeometry::new(0.15).unwrap();
        let elements = ElementPlacement::generate_elements(&geometry, 64).unwrap();

        assert_eq!(elements.len(), 64);
        for element in elements {
            let position = element.position;
            let radius = position[0].hypot(position[1]).hypot(position[2]);
            assert!((radius - geometry.radius).abs() < 1.0e-12);
            assert!(
                position[1] >= -1.0e-12,
                "hemisphere aperture must remain on +y side"
            );

            let normal = element.normal;
            let normal_norm = normal[0].hypot(normal[1]).hypot(normal[2]);
            assert!((normal_norm - 1.0).abs() < 1.0e-12);
            assert!((position[0] + geometry.radius * normal[0]).abs() < 1.0e-12);
            assert!((position[1] + geometry.radius * normal[1]).abs() < 1.0e-12);
            assert!((position[2] + geometry.radius * normal[2]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn singleton_element_is_finite() {
        let geometry = HemisphereGeometry::new(0.15).unwrap();
        let elements = ElementPlacement::generate_elements(&geometry, 1).unwrap();

        assert_eq!(elements.len(), 1);
        assert!(elements[0].position.iter().all(|value| value.is_finite()));
        assert!(elements[0].normal.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn zero_elements_are_rejected() {
        let geometry = HemisphereGeometry::new(0.15).unwrap();
        let result = ElementPlacement::generate_elements(&geometry, 0);
        assert!(result.is_err());
    }

    #[test]
    fn nonfinite_radius_is_rejected() {
        assert!(HemisphereGeometry::new(f64::NAN).is_err());
    }
}
