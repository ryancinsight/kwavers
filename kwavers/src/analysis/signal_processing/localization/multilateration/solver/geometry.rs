//! Geometric helper methods for `Multilateration`.
//!
//! Contains point-to-point distance, residual computation, and Jacobian
//! assembly for the TDOA Levenberg-Marquardt iteration.

use super::Multilateration;

impl Multilateration {
    pub(super) fn compute_residuals(
        &self,
        position: &[f64; 3],
        measured_range_diffs: &[f64],
    ) -> Vec<f64> {
        let ref_pos = &self.sensor_positions[0];
        let ref_range = self.distance(position, ref_pos);

        measured_range_diffs
            .iter()
            .enumerate()
            .map(|(i, &measured)| {
                let sensor_pos = &self.sensor_positions[i + 1];
                let range = self.distance(position, sensor_pos);
                measured - (range - ref_range)
            })
            .collect()
    }

    pub(super) fn compute_jacobian(&self, position: &[f64; 3]) -> Vec<[f64; 3]> {
        let ref_pos = &self.sensor_positions[0];
        let ref_range = self.distance(position, ref_pos);

        let ref_dx = (position[0] - ref_pos[0]) / ref_range;
        let ref_dy = (position[1] - ref_pos[1]) / ref_range;
        let ref_dz = (position[2] - ref_pos[2]) / ref_range;

        (1..self.num_sensors)
            .map(|i| {
                let sensor_pos = &self.sensor_positions[i];
                let range = self.distance(position, sensor_pos);
                let dx = (position[0] - sensor_pos[0]) / range;
                let dy = (position[1] - sensor_pos[1]) / range;
                let dz = (position[2] - sensor_pos[2]) / range;
                [-(dx - ref_dx), -(dy - ref_dy), -(dz - ref_dz)]
            })
            .collect()
    }

    pub(super) fn distance(&self, p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
    }
}
