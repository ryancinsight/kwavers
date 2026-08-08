//! Geometry state and deformation tracking for flexible transducer arrays
//!
//! This module handles the geometric representation and deformation state
//! of flexible transducer arrays.

use aequitas::systems::si::quantities::{
    Dimensionless, EnergyPerVolume, Length, Pressure, ReciprocalLength, Time,
};
use aequitas::systems::si::units::{JoulePerCubicMeter, Pascal, PerMeter};
use leto::{Array1, Array2};
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
    /// Timestamp of last geometry update.
    pub timestamp: Time<f64>,
    /// Deformation state
    pub deformation: DeformationState,
}

impl GeometryState {
    /// Create a new geometry state for a flat array
    #[must_use]
    pub fn flat_array(num_elements: usize, spacing: f64) -> Self {
        let mut positions = Array2::zeros([num_elements, 3]);
        let mut normals = Array2::zeros([num_elements, 3]);

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
            position_confidence: Array1::ones([num_elements]),
            timestamp: Time::from_unit::<aequitas::systems::si::units::Second>(0.0),
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
    pub fn calculate_curvature(&self) -> ReciprocalLength<f64> {
        if self.element_positions.shape()[0] < 3 {
            return ReciprocalLength::from_unit::<PerMeter>(0.0);
        }

        // Average the Menger curvature of each consecutive three-point arc.
        // Unlike the turning angle alone, this retains the required 1/m unit.
        let mut total_curvature = 0.0;
        let n_elements = self.element_positions.shape()[0];

        for i in 1..n_elements - 1 {
            let p1 = [
                self.element_positions[[i - 1, 0]],
                self.element_positions[[i - 1, 1]],
                self.element_positions[[i - 1, 2]],
            ];
            let p2 = [
                self.element_positions[[i, 0]],
                self.element_positions[[i, 1]],
                self.element_positions[[i, 2]],
            ];
            let p3 = [
                self.element_positions[[i + 1, 0]],
                self.element_positions[[i + 1, 1]],
                self.element_positions[[i + 1, 2]],
            ];

            // Calculate vectors
            let v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
            let v2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];

            let mag1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mag2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
            let chord: [f64; 3] = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];
            let cross = [
                v1[1].mul_add(v2[2], -(v1[2] * v2[1])),
                v1[2].mul_add(v2[0], -(v1[0] * v2[2])),
                v1[0].mul_add(v2[1], -(v1[1] * v2[0])),
            ];
            let cross_norm = cross.iter().map(|x| x * x).sum::<f64>().sqrt();
            let chord_norm = chord.iter().map(|x| x * x).sum::<f64>().sqrt();

            if mag1 > 0.0 && mag2 > 0.0 && chord_norm > 0.0 {
                total_curvature += 2.0 * cross_norm / (mag1 * mag2 * chord_norm);
            }
        }

        ReciprocalLength::from_unit::<PerMeter>(total_curvature / (n_elements - 2) as f64)
    }

    /// Get the centroid of all element positions
    #[must_use]
    pub fn centroid(&self) -> [f64; 3] {
        let n = self.element_positions.shape()[0] as f64;
        let mut sum = [0.0; 3];
        for i in 0..self.element_positions.shape()[0] {
            sum[0] += self.element_positions[[i, 0]];
            sum[1] += self.element_positions[[i, 1]];
            sum[2] += self.element_positions[[i, 2]];
        }
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }
}

/// Deformation state tracking
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DeformationState {
    /// Curvature radius, `None` for a flat or underdetermined centreline.
    pub curvature_radius: Option<Length<f64>>,
    /// Strain values for each element
    pub strain: Vec<Dimensionless<f64>>,
    /// Stress values for each element (Pa)
    pub stress: Vec<Pressure<f64>>,
    /// Strain-energy density (J/m³) derived from `½ ε σ`.
    pub deformation_energy_density: EnergyPerVolume<f64>,
    /// Maximum allowable deformation before damage
    pub max_safe_deformation: Dimensionless<f64>,
}

impl DeformationState {
    /// Check if deformation is within safe limits
    #[must_use]
    pub fn is_safe(&self) -> bool {
        self.strain
            .iter()
            .all(|s| s.into_base().abs() < self.max_safe_deformation.into_base())
    }

    /// Calculate deformation energy from strain and stress
    pub fn calculate_energy(&mut self) {
        let density = self
            .strain
            .iter()
            .zip(self.stress.iter())
            .map(|(strain, stress)| 0.5 * strain.into_base() * stress.in_unit::<Pascal>())
            .sum();
        self.deformation_energy_density = EnergyPerVolume::from_unit::<JoulePerCubicMeter>(density);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::{Meter, PerMeter};

    #[test]
    fn flat_geometry_has_zero_inverse_length_curvature() {
        let geometry = GeometryState::flat_array(5, 1.0e-3);

        assert_eq!(geometry.calculate_curvature().in_unit::<PerMeter>(), 0.0);
    }

    #[test]
    fn deformation_energy_is_reported_as_energy_density() {
        let mut deformation = DeformationState {
            curvature_radius: Some(Length::from_unit::<Meter>(0.01)),
            strain: vec![Dimensionless::from_base(0.1)],
            stress: vec![Pressure::from_unit::<Pascal>(2.0e6)],
            deformation_energy_density: EnergyPerVolume::from_base(0.0),
            max_safe_deformation: Dimensionless::from_base(0.2),
        };

        deformation.calculate_energy();

        assert_eq!(
            deformation
                .deformation_energy_density
                .in_unit::<JoulePerCubicMeter>(),
            1.0e5
        );
        assert!(deformation.is_safe());
    }
}
