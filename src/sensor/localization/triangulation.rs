// localization/triangulation.rs - Triangulation methods

use super::{Position, SensorArray};
use crate::error::KwaversResult;

/// Triangulator for position estimation
#[derive(Debug)]
pub struct Triangulator {
    #[allow(dead_code)]
    method: TriangulationMethod,
}

/// Triangulation methods
#[derive(Debug, Clone, Copy)]
pub enum TriangulationMethod {
    LeastSquares,
    WeightedLeastSquares,
    MaximumLikelihood,
}

impl Triangulator {
    /// Create new triangulator
    #[must_use]
    pub fn new(method: TriangulationMethod) -> Self {
        Self { method }
    }

    /// Triangulate position from ranges
    pub fn triangulate(&self, ranges: &[f64], array: &SensorArray) -> KwaversResult<Position> {
        if ranges.len() < 3 {
            return Err(crate::error::KwaversError::InvalidInput(
                "Need at least 3 ranges for 3D triangulation".to_string(),
            ));
        }

        // Multilateration using least squares
        // Solves the overdetermined system for position (x, y, z)
        // where ||(x, y, z) - sensor_i|| = range_i
        //
        // Reference: Cheung et al. (2004), "Least squares localization"
        
        let positions = array.get_sensor_positions();
        let n = ranges.len().min(positions.len());
        
        if n < 3 {
            return Ok(array.centroid());
        }
        
        // Use first sensor as reference point
        let ref_pos = positions[0];
        let ref_range = ranges[0];
        
        // Build linear system: A*x = b
        // where each row represents: 2(p_i - p_0)·x = ||p_i||² - ||p_0||² + r_0² - r_i²
        use nalgebra::{DMatrix, DVector};
        
        let mut a_mat = DMatrix::zeros(n - 1, 3);
        let mut b_vec = DVector::zeros(n - 1);
        
        for i in 1..n {
            let pi = positions[i];
            let ri = ranges[i];
            
            // Row: 2(p_i - p_0)
            a_mat[(i - 1, 0)] = 2.0 * (pi.x - ref_pos.x);
            a_mat[(i - 1, 1)] = 2.0 * (pi.y - ref_pos.y);
            a_mat[(i - 1, 2)] = 2.0 * (pi.z - ref_pos.z);
            
            // RHS: ||p_i||² - ||p_0||² + r_0² - r_i²
            let pi_norm_sq = pi.x * pi.x + pi.y * pi.y + pi.z * pi.z;
            let p0_norm_sq = ref_pos.x * ref_pos.x + ref_pos.y * ref_pos.y + ref_pos.z * ref_pos.z;
            b_vec[i - 1] = pi_norm_sq - p0_norm_sq + ref_range * ref_range - ri * ri;
        }
        
        // Solve least squares: x = (A^T A)^(-1) A^T b
        let ata = a_mat.transpose() * &a_mat;
        let atb = a_mat.transpose() * &b_vec;
        
        match ata.try_inverse() {
            Some(ata_inv) => {
                let solution = ata_inv * atb;
                Ok(Position {
                    x: solution[0],
                    y: solution[1],
                    z: solution[2],
                })
            }
            None => {
                // Matrix is singular, fall back to centroid
                Ok(array.centroid())
            }
        }
    }

    /// Triangulate from angles
    pub fn triangulate_angles(
        &self,
        angles: &[(f64, f64)],
        array: &SensorArray,
    ) -> KwaversResult<Position> {
        if angles.len() < 2 {
            return Err(crate::error::KwaversError::InvalidInput(
                "Need at least 2 angle measurements".to_string(),
            ));
        }

        // Angle-based triangulation using direction vectors
        // Each angle pair (azimuth, elevation) defines a ray from sensor position
        // Find point that minimizes distance to all rays
        //
        // Reference: Hartley & Sturm (1997), "Triangulation"
        
        let positions = array.get_sensor_positions();
        let n = angles.len().min(positions.len());
        
        if n < 2 {
            return Ok(array.centroid());
        }
        
        use nalgebra::{DMatrix, DVector, Vector3};
        
        // For each sensor, compute direction vector from angles
        let mut rays: Vec<(Vector3<f64>, Vector3<f64>)> = Vec::new();
        
        for i in 0..n {
            let (azimuth, elevation) = angles[i];
            let pos = positions[i];
            
            // Direction vector from spherical coordinates
            // x = cos(elev) * cos(azim)
            // y = cos(elev) * sin(azim)
            // z = sin(elev)
            let dir = Vector3::new(
                elevation.cos() * azimuth.cos(),
                elevation.cos() * azimuth.sin(),
                elevation.sin(),
            );
            
            let origin = Vector3::new(pos.x, pos.y, pos.z);
            rays.push((origin, dir));
        }
        
        // Find point minimizing sum of squared distances to all rays
        // Using algebraic solution for ray intersection
        let mut a_mat = DMatrix::zeros(3 * (n - 1), 3);
        let mut b_vec = DVector::zeros(3 * (n - 1));
        
        for i in 1..n {
            let (_p0, _d0) = &rays[0];
            let (pi, di) = &rays[i];
            
            // For each pair of rays, add constraints
            // (I - d·d^T)(p - p_i) = 0
            let idx = 3 * (i - 1);
            
            for j in 0..3 {
                for k in 0..3 {
                    let delta = if j == k { 1.0 } else { 0.0 };
                    a_mat[(idx + j, k)] = delta - di[j] * di[k];
                }
                b_vec[idx + j] = di[j] * pi[j];
            }
        }
        
        // Least squares solution
        let ata = a_mat.transpose() * &a_mat;
        let atb = a_mat.transpose() * &b_vec;
        
        match ata.try_inverse() {
            Some(ata_inv) => {
                let solution = ata_inv * atb;
                Ok(Position {
                    x: solution[0],
                    y: solution[1],
                    z: solution[2],
                })
            }
            None => Ok(array.centroid()),
        }
    }
}
