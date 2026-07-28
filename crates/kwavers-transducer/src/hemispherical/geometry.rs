//! Hemisphere geometry and element placement

use super::constants::MAX_ELEMENT_DENSITY;
use super::element::ElementConfiguration;
use crate::transducers::focused::{SphericalCapConfig, SphericalCapLayout};
use aequitas::systems::si::quantities::Length;
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

/// Hemisphere geometry definition
#[derive(Debug, Clone)]
pub struct HemisphereGeometry {
    /// Radius of hemisphere in the SI base unit metre.
    pub radius: Length<f64>,
    /// F-number (`focal_length/aperture`)
    pub f_number: f64,
    /// Aperture diameter in the SI base unit metre.
    pub aperture: Length<f64>,
    /// Focal length in the SI base unit metre.
    pub focal_length: Length<f64>,
}

impl HemisphereGeometry {
    /// Create new hemisphere geometry
    /// # Errors
    /// - Returns `KwaversError::Config` if the precondition for a Config-class constraint is violated.
    ///
    pub fn new(radius: Length<f64>) -> KwaversResult<Self> {
        let radius_m = radius.into_base();
        if !radius_m.is_finite() || radius_m <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "radius".to_owned(),
                value: radius_m.to_string(),
                constraint: "must be positive and finite".to_owned(),
            }));
        }

        let f_number = 0.9; // Typical for hemispherical arrays
        let aperture = Length::from_base(2.0 * radius_m);
        let focal_length = Length::from_base(f_number * aperture.into_base());

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
    pub fn focal_point(&self) -> [Length<f64>; 3] {
        [
            Length::from_base(0.0),
            Length::from_base(0.0),
            self.focal_length,
        ]
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
            geometry.radius.into_base(),
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ))?;
        let element_radius =
            Length::from_base(geometry.radius.into_base() / (num_elements as f64).sqrt() * 0.4);

        Ok(layout
            .elements()
            .iter()
            .map(|element| {
                ElementConfiguration::new(
                    element.position_m.map(Length::from_base),
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
        let base_elements = (FOUR_PI
            * geometry.radius.into_base()
            * geometry.radius.into_base()
            * MAX_ELEMENT_DENSITY) as usize;
        let num_elements = (base_elements as f64 * density_factor) as usize;
        Self::generate_elements(geometry, num_elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_elements_use_source_domain_hemisphere() {
        let geometry = HemisphereGeometry::new(Length::from_base(0.15)).unwrap();
        let elements = ElementPlacement::generate_elements(&geometry, 64).unwrap();

        assert_eq!(elements.len(), 64);
        for element in elements {
            let position = element.position;
            let radius = position[0]
                .into_base()
                .hypot(position[1].into_base())
                .hypot(position[2].into_base());
            assert!((radius - geometry.radius.into_base()).abs() < 1.0e-12);
            assert!(
                position[1].into_base() >= -1.0e-12,
                "hemisphere aperture must remain on +y side"
            );

            let normal = element.normal;
            let normal_norm = normal[0].hypot(normal[1]).hypot(normal[2]);
            assert!((normal_norm - 1.0).abs() < 1.0e-12);
            assert!(
                (position[0].into_base() + geometry.radius.into_base() * normal[0]).abs() < 1.0e-12
            );
            assert!(
                (position[1].into_base() + geometry.radius.into_base() * normal[1]).abs() < 1.0e-12
            );
            assert!(
                (position[2].into_base() + geometry.radius.into_base() * normal[2]).abs() < 1.0e-12
            );
        }
    }

    #[test]
    fn singleton_element_is_finite() {
        let geometry = HemisphereGeometry::new(Length::from_base(0.15)).unwrap();
        let elements = ElementPlacement::generate_elements(&geometry, 1).unwrap();

        assert_eq!(elements.len(), 1);
        assert!(elements[0]
            .position
            .iter()
            .all(|value| value.into_base().is_finite()));
        assert!(elements[0].normal.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn zero_elements_are_rejected() {
        let geometry = HemisphereGeometry::new(Length::from_base(0.15)).unwrap();
        let result = ElementPlacement::generate_elements(&geometry, 0);
        assert!(result.is_err());
    }

    #[test]
    fn nonfinite_radius_is_rejected() {
        assert!(HemisphereGeometry::new(Length::from_base(f64::NAN)).is_err());
    }
}
