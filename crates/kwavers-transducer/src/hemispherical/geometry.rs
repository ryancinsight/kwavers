//! Hemisphere geometry and element placement

use super::constants::MAX_ELEMENT_DENSITY;
use super::element::ElementConfiguration;
use crate::transducers::focused::{SphericalCapConfig, SphericalCapLayout};
use aequitas::systems::si::quantities::Length;
use aequitas::systems::si::units::Meter;
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

/// Hemisphere geometry definition
#[derive(Debug, Clone)]
pub struct HemisphereGeometry {
    /// Radius of the hemisphere.
    pub radius: Length<f64>,
    /// F-number (`focal_length / aperture`) — dimensionless.
    pub f_number: f64,
    /// Aperture diameter.
    pub aperture: Length<f64>,
    /// Focal length.
    pub focal_length: Length<f64>,
}

impl HemisphereGeometry {
    /// Create a new hemisphere geometry from a physical radius.
    ///
    /// # Errors
    /// Returns `KwaversError::Config` when `radius` is not positive and finite.
    pub fn new(radius: Length<f64>) -> KwaversResult<Self> {
        let radius_m = radius.in_unit::<Meter>();
        if !radius_m.is_finite() || radius_m <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "radius".to_owned(),
                value: radius_m.to_string(),
                constraint: "must be positive and finite".to_owned(),
            }));
        }

        let f_number = 0.9; // Typical for hemispherical arrays
        let aperture = Length::from_unit::<Meter>(2.0 * radius_m);
        let focal_length = Length::from_unit::<Meter>(f_number * aperture.in_unit::<Meter>());

        Ok(Self {
            radius,
            f_number,
            aperture,
            focal_length,
        })
    }

    /// Return the geometric focus point as mesh coordinates `[x, y, z]` in metres.
    ///
    /// Scalar extraction is valid here: the result is passed directly to the mesh layer.
    #[must_use]
    pub fn focal_point(&self) -> [f64; 3] {
        [0.0, 0.0, self.focal_length.in_unit::<Meter>()]
    }
}

/// Element placement algorithms
#[derive(Debug)]
pub struct ElementPlacement;

impl ElementPlacement {
    /// Generate element positions on hemisphere.
    ///
    /// # Errors
    /// Returns an error if the layout or element count is invalid.
    pub fn generate_elements(
        geometry: &HemisphereGeometry,
        num_elements: usize,
    ) -> KwaversResult<Vec<ElementConfiguration>> {
        // Extract scalar at the mesh/formula boundary: SphericalCapConfig uses raw f64 metres.
        let radius_m = geometry.radius.in_unit::<Meter>();
        let layout = SphericalCapLayout::new(SphericalCapConfig::hemisphere(
            num_elements,
            radius_m,
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ))?;
        let element_radius_m = radius_m / (num_elements as f64).sqrt() * 0.4;
        let element_radius = Length::from_unit::<Meter>(element_radius_m);

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

    /// Generate sparse element distribution.
    ///
    /// # Errors
    /// Returns an error if the layout or element count is invalid.
    pub fn generate_sparse(
        geometry: &HemisphereGeometry,
        density_factor: f64,
    ) -> KwaversResult<Vec<ElementConfiguration>> {
        // Scalar extraction: area formula uses raw metres squared.
        let radius_m = geometry.radius.in_unit::<Meter>();
        let base_elements = (FOUR_PI * radius_m * radius_m * MAX_ELEMENT_DENSITY) as usize;
        let num_elements = (base_elements as f64 * density_factor) as usize;
        Self::generate_elements(geometry, num_elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::Meter;

    fn radius_150mm() -> Length<f64> {
        Length::from_unit::<Meter>(0.15)
    }

    #[test]
    fn generated_elements_use_source_domain_hemisphere() {
        let geometry = HemisphereGeometry::new(radius_150mm()).unwrap();
        let elements = ElementPlacement::generate_elements(&geometry, 64).unwrap();
        let radius_m = geometry.radius.in_unit::<Meter>();

        assert_eq!(elements.len(), 64);
        for element in elements {
            let position = element.position;
            let dist = position[0].hypot(position[1]).hypot(position[2]);
            assert!((dist - radius_m).abs() < 1.0e-12);
            assert!(
                position[1] >= -1.0e-12,
                "hemisphere aperture must remain on +y side"
            );

            let normal = element.normal;
            let normal_norm = normal[0].hypot(normal[1]).hypot(normal[2]);
            assert!((normal_norm - 1.0).abs() < 1.0e-12);
            assert!((position[0] + radius_m * normal[0]).abs() < 1.0e-12);
            assert!((position[1] + radius_m * normal[1]).abs() < 1.0e-12);
            assert!((position[2] + radius_m * normal[2]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn singleton_element_is_finite() {
        let geometry = HemisphereGeometry::new(radius_150mm()).unwrap();
        let elements = ElementPlacement::generate_elements(&geometry, 1).unwrap();

        assert_eq!(elements.len(), 1);
        assert!(elements[0].position.iter().all(|value| value.is_finite()));
        assert!(elements[0].normal.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn zero_elements_are_rejected() {
        let geometry = HemisphereGeometry::new(radius_150mm()).unwrap();
        let result = ElementPlacement::generate_elements(&geometry, 0);
        assert!(result.is_err());
    }

    #[test]
    fn nonfinite_radius_is_rejected() {
        assert!(HemisphereGeometry::new(Length::from_unit::<Meter>(f64::NAN)).is_err());
    }

    #[test]
    fn negative_radius_is_rejected() {
        assert!(HemisphereGeometry::new(Length::from_unit::<Meter>(-0.1)).is_err());
    }

    #[test]
    fn focal_point_at_focal_length() {
        let geometry = HemisphereGeometry::new(radius_150mm()).unwrap();
        let fp = geometry.focal_point();
        assert_eq!(fp[0], 0.0);
        assert_eq!(fp[1], 0.0);
        let expected = geometry.focal_length.in_unit::<Meter>();
        assert!((fp[2] - expected).abs() < 1.0e-15);
    }
}
