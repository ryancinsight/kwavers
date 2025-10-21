//! Transducer Geometry Support for FNM
//!
//! Provides geometry definitions and conversions for various transducer types
//! compatible with the Fast Nearfield Method.
//!
//! ## Supported Geometries
//!
//! - Rectangular pistons (linear and phased arrays)
//! - Circular pistons (focused transducers)
//! - Arbitrary aperture shapes (through discretization)

use ndarray::Array2;
use std::f64::consts::PI;

/// Transducer geometry type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransducerType {
    /// Rectangular piston transducer
    Rectangular,
    /// Circular piston transducer
    Circular,
    /// Arbitrary shaped aperture
    Arbitrary,
}

/// Transducer geometry for FNM calculations
#[derive(Debug, Clone)]
pub struct TransducerGeometry {
    /// Transducer type
    pub transducer_type: TransducerType,
    /// Element positions [N × 3] (x, y, z) in meters
    pub element_positions: Array2<f64>,
    /// Element sizes [N × 2] (width, height) in meters
    pub element_sizes: Array2<f64>,
    /// Element normal vectors [N × 3]
    pub element_normals: Array2<f64>,
    /// Optional apodization weights [N]
    pub apodization: Option<Vec<f64>>,
    /// Optional time delays [N] in seconds
    pub delays: Option<Vec<f64>>,
}

impl TransducerGeometry {
    /// Create a rectangular piston transducer
    ///
    /// # Arguments
    ///
    /// * `width` - Transducer width (m)
    /// * `height` - Transducer height (m)
    /// * `center` - Center position [x, y, z] (m)
    ///
    /// # Example
    ///
    /// ```
    /// use kwavers::physics::transducer::fast_nearfield::geometry::TransducerGeometry;
    ///
    /// let geometry = TransducerGeometry::rectangular(0.01, 0.01, [0.0, 0.0, 0.0]);
    /// ```
    pub fn rectangular(width: f64, height: f64, center: [f64; 3]) -> Self {
        // Single element rectangular piston
        let mut element_positions = Array2::zeros((1, 3));
        element_positions[[0, 0]] = center[0];
        element_positions[[0, 1]] = center[1];
        element_positions[[0, 2]] = center[2];

        let mut element_sizes = Array2::zeros((1, 2));
        element_sizes[[0, 0]] = width;
        element_sizes[[0, 1]] = height;

        let mut element_normals = Array2::zeros((1, 3));
        element_normals[[0, 2]] = 1.0; // Normal along z-axis

        Self {
            transducer_type: TransducerType::Rectangular,
            element_positions,
            element_sizes,
            element_normals,
            apodization: None,
            delays: None,
        }
    }

    /// Create a circular piston transducer
    ///
    /// # Arguments
    ///
    /// * `radius` - Transducer radius (m)
    /// * `center` - Center position [x, y, z] (m)
    ///
    /// # Example
    ///
    /// ```
    /// use kwavers::physics::transducer::fast_nearfield::geometry::TransducerGeometry;
    ///
    /// let geometry = TransducerGeometry::circular(0.005, [0.0, 0.0, 0.0]);
    /// ```
    pub fn circular(radius: f64, center: [f64; 3]) -> Self {
        // Single element circular piston
        let mut element_positions = Array2::zeros((1, 3));
        element_positions[[0, 0]] = center[0];
        element_positions[[0, 1]] = center[1];
        element_positions[[0, 2]] = center[2];

        // Store radius as diameter for consistency
        let mut element_sizes = Array2::zeros((1, 2));
        element_sizes[[0, 0]] = 2.0 * radius;
        element_sizes[[0, 1]] = 2.0 * radius;

        let mut element_normals = Array2::zeros((1, 3));
        element_normals[[0, 2]] = 1.0; // Normal along z-axis

        Self {
            transducer_type: TransducerType::Circular,
            element_positions,
            element_sizes,
            element_normals,
            apodization: None,
            delays: None,
        }
    }

    /// Create a phased array with multiple elements
    ///
    /// # Arguments
    ///
    /// * `num_elements` - Number of array elements
    /// * `element_width` - Width of each element (m)
    /// * `element_height` - Height of each element (m)
    /// * `pitch` - Center-to-center spacing (m)
    /// * `center` - Array center position [x, y, z] (m)
    ///
    /// # Example
    ///
    /// ```
    /// use kwavers::physics::transducer::fast_nearfield::geometry::TransducerGeometry;
    ///
    /// let geometry = TransducerGeometry::phased_array(
    ///     64, 0.0003, 0.005, 0.0004, [0.0, 0.0, 0.0]
    /// );
    /// ```
    pub fn phased_array(
        num_elements: usize,
        element_width: f64,
        element_height: f64,
        pitch: f64,
        center: [f64; 3],
    ) -> Self {
        let mut element_positions = Array2::zeros((num_elements, 3));
        let mut element_sizes = Array2::zeros((num_elements, 2));
        let mut element_normals = Array2::zeros((num_elements, 3));

        // Calculate positions for linear array along x-axis
        let array_width = (num_elements - 1) as f64 * pitch;
        let start_x = center[0] - array_width / 2.0;

        for i in 0..num_elements {
            element_positions[[i, 0]] = start_x + i as f64 * pitch;
            element_positions[[i, 1]] = center[1];
            element_positions[[i, 2]] = center[2];

            element_sizes[[i, 0]] = element_width;
            element_sizes[[i, 1]] = element_height;

            element_normals[[i, 2]] = 1.0; // All normals along z-axis
        }

        Self {
            transducer_type: TransducerType::Rectangular,
            element_positions,
            element_sizes,
            element_normals,
            apodization: None,
            delays: None,
        }
    }

    /// Set apodization weights for the transducer elements
    ///
    /// # Arguments
    ///
    /// * `weights` - Apodization weights (length must match number of elements)
    pub fn with_apodization(mut self, weights: Vec<f64>) -> Self {
        self.apodization = Some(weights);
        self
    }

    /// Set time delays for the transducer elements
    ///
    /// # Arguments
    ///
    /// * `delays` - Time delays in seconds (length must match number of elements)
    pub fn with_delays(mut self, delays: Vec<f64>) -> Self {
        self.delays = Some(delays);
        self
    }

    /// Get number of elements in the transducer
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.element_positions.nrows()
    }

    /// Get total aperture area (m²)
    #[must_use]
    pub fn aperture_area(&self) -> f64 {
        match self.transducer_type {
            TransducerType::Rectangular => {
                self.element_sizes.iter().copied().product::<f64>() * self.num_elements() as f64
            }
            TransducerType::Circular => {
                let radius = self.element_sizes[[0, 0]] / 2.0;
                PI * radius * radius
            }
            TransducerType::Arbitrary => {
                // Sum element areas
                (0..self.num_elements())
                    .map(|i| self.element_sizes[[i, 0]] * self.element_sizes[[i, 1]])
                    .sum()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular_geometry() {
        let geometry = TransducerGeometry::rectangular(0.01, 0.015, [0.0, 0.0, 0.0]);
        assert_eq!(geometry.transducer_type, TransducerType::Rectangular);
        assert_eq!(geometry.num_elements(), 1);
        assert!((geometry.aperture_area() - 0.00015).abs() < 1e-10);
    }

    #[test]
    fn test_circular_geometry() {
        let radius = 0.005;
        let geometry = TransducerGeometry::circular(radius, [0.0, 0.0, 0.0]);
        assert_eq!(geometry.transducer_type, TransducerType::Circular);
        assert_eq!(geometry.num_elements(), 1);
        
        let expected_area = PI * radius * radius;
        assert!((geometry.aperture_area() - expected_area).abs() < 1e-10);
    }

    #[test]
    fn test_phased_array() {
        let geometry = TransducerGeometry::phased_array(
            64, 0.0003, 0.005, 0.0004, [0.0, 0.0, 0.0]
        );
        assert_eq!(geometry.num_elements(), 64);
        assert_eq!(geometry.transducer_type, TransducerType::Rectangular);
        
        // Check element spacing
        let x0 = geometry.element_positions[[0, 0]];
        let x1 = geometry.element_positions[[1, 0]];
        assert!((x1 - x0 - 0.0004).abs() < 1e-10);
    }

    #[test]
    fn test_apodization() {
        let weights = vec![1.0, 0.9, 0.8, 0.7];
        let geometry = TransducerGeometry::phased_array(
            4, 0.001, 0.005, 0.0015, [0.0, 0.0, 0.0]
        ).with_apodization(weights.clone());
        
        assert!(geometry.apodization.is_some());
        assert_eq!(geometry.apodization.unwrap(), weights);
    }

    #[test]
    fn test_delays() {
        let delays = vec![0.0, 1e-7, 2e-7, 3e-7];
        let geometry = TransducerGeometry::phased_array(
            4, 0.001, 0.005, 0.0015, [0.0, 0.0, 0.0]
        ).with_delays(delays.clone());
        
        assert!(geometry.delays.is_some());
        assert_eq!(geometry.delays.unwrap(), delays);
    }
}
