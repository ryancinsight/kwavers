//! Geometry state and deformation tracking for flexible transducer arrays
//!
//! This module handles the geometric representation and deformation state
//! of flexible transducer arrays.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Geometry state of the flexible transducer array
#[derive(Debug, Clone)]
pub struct GeometryState {
    /// Current element positions [x, y, z] for each element
    pub element_positions: Array2<f64>,
    /// Element orientations (normal vectors)
    pub element_normals: Array2<f64>,
    /// Confidence values for position estimates (0-1)
    pub position_confidence: Array1<f64>,
    /// Timestamp of last geometry update
    pub timestamp: f64,
    /// Deformation state
    pub deformation: DeformationState,
}

impl GeometryState {
    /// Create a new geometry state for a flat array
    #[must_use]
    pub fn flat_array(num_elements: usize, spacing: f64) -> Self {
        let mut positions = Array2::zeros((num_elements, 3));
        let mut normals = Array2::zeros((num_elements, 3));

        // Initialize flat array along x-axis
        for i in 0..num_elements {
            let x = (i as f64 - (num_elements - 1) as f64 / 2.0) * spacing;
            positions[[i, 0]] = x;
            positions[[i, 1]] = 0.0;
            positions[[i, 2]] = 0.0;

            // All normals point in +z direction for flat array
            normals[[i, 0]] = 0.0;
            normals[[i, 1]] = 0.0;
            normals[[i, 2]] = 1.0;
        }

        Self {
            element_positions: positions,
            element_normals: normals,
            position_confidence: Array1::ones(num_elements),
            timestamp: 0.0,
            deformation: DeformationState::default(),
        }
    }

    /// Update element positions and normals
    pub fn update_positions(&mut self, positions: Array2<f64>, normals: Array2<f64>) {
        self.element_positions = positions;
        self.element_normals = normals;
    }

    /// Calculate curvature from current positions
    #[must_use]
    pub fn calculate_curvature(&self) -> f64 {
        if self.element_positions.nrows() < 3 {
            return 0.0;
        }

        // Simple curvature estimation using three-point formula
        let mut total_curvature = 0.0;
        let n_elements = self.element_positions.nrows();

        for i in 1..n_elements - 1 {
            let p1 = self.element_positions.row(i - 1);
            let p2 = self.element_positions.row(i);
            let p3 = self.element_positions.row(i + 1);

            // Calculate vectors
            let v1 = &p2 - &p1;
            let v2 = &p3 - &p2;

            // Calculate angle between vectors
            let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
            let mag1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mag2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

            if mag1 > 0.0 && mag2 > 0.0 {
                let cos_angle = (dot_product / (mag1 * mag2)).clamp(-1.0, 1.0);
                let angle = cos_angle.acos();
                total_curvature += angle;
            }
        }

        total_curvature / (n_elements - 2) as f64
    }

    /// Get the centroid of all element positions
    #[must_use]
    pub fn centroid(&self) -> [f64; 3] {
        let n = self.element_positions.nrows() as f64;
        let sum = self.element_positions.sum_axis(ndarray::Axis(0));
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }
}

/// Deformation state tracking
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DeformationState {
    /// Curvature radius (m), None if flat
    pub curvature_radius: Option<f64>,
    /// Strain values for each element
    pub strain: Vec<f64>,
    /// Stress values for each element (Pa)
    pub stress: Vec<f64>,
    /// Total deformation energy (J)
    pub deformation_energy: f64,
    /// Maximum allowable deformation before damage
    pub max_safe_deformation: f64,
}

impl DeformationState {
    /// Check if deformation is within safe limits
    #[must_use]
    pub fn is_safe(&self) -> bool {
        self.strain
            .iter()
            .all(|&s| s.abs() < self.max_safe_deformation)
    }

    /// Calculate deformation energy from strain and stress
    pub fn calculate_energy(&mut self) {
        self.deformation_energy = self
            .strain
            .iter()
            .zip(self.stress.iter())
            .map(|(strain, stress)| 0.5 * strain * stress)
            .sum();
    }
}
