//! Rotation matrices and transformations for anisotropic materials

use crate::constants::elastic::BOND_TRANSFORM_FACTOR;
use ndarray::Array2;

/// 3D rotation matrix for coordinate transformations
#[derive(Debug, Clone)]
pub struct RotationMatrix {
    /// 3x3 rotation matrix
    r: Array2<f64>,
}

impl RotationMatrix {
    /// Create rotation matrix from Euler angles (ZYX convention)
    pub fn from_euler(phi: f64, theta: f64, psi: f64) -> Self {
        let (sp, cp) = phi.sin_cos();
        let (st, ct) = theta.sin_cos();
        let (ss, cs) = psi.sin_cos();

        let mut r = Array2::zeros((3, 3));

        // ZYX rotation matrix
        r[[0, 0]] = ct * cp;
        r[[0, 1]] = sp * st * cp - cs * sp;
        r[[0, 2]] = cs * st * cp + sp * sp;

        r[[1, 0]] = ct * sp;
        r[[1, 1]] = sp * st * sp + cs * cp;
        r[[1, 2]] = cs * st * sp - sp * cp;

        r[[2, 0]] = -st;
        r[[2, 1]] = sp * ct;
        r[[2, 2]] = cs * ct;

        Self { r }
    }

    /// Create rotation around Z axis
    pub fn z_rotation(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        let mut r = Array2::eye(3);

        r[[0, 0]] = c;
        r[[0, 1]] = -s;
        r[[1, 0]] = s;
        r[[1, 1]] = c;

        Self { r }
    }

    /// Create rotation around Y axis
    pub fn y_rotation(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        let mut r = Array2::eye(3);

        r[[0, 0]] = c;
        r[[0, 2]] = s;
        r[[2, 0]] = -s;
        r[[2, 2]] = c;

        Self { r }
    }

    /// Create rotation around X axis
    pub fn x_rotation(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        let mut r = Array2::eye(3);

        r[[1, 1]] = c;
        r[[1, 2]] = -s;
        r[[2, 1]] = s;
        r[[2, 2]] = c;

        Self { r }
    }

    /// Apply rotation to a vector
    pub fn apply_to_vector(&self, v: &[f64; 3]) -> [f64; 3] {
        let mut result = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i] += self.r[[i, j]] * v[j];
            }
        }
        result
    }

    /// Apply rotation to stiffness tensor (Bond transformation)
    pub fn apply_to_stiffness(&self, c: &Array2<f64>) -> Array2<f64> {
        // Bond transformation matrix (6x6)
        let bond = self.bond_matrix();

        // C' = M * C * M^T
        let temp = bond.dot(c);
        temp.dot(&bond.t())
    }

    /// Create Bond transformation matrix for Voigt notation
    fn bond_matrix(&self) -> Array2<f64> {
        let mut m = Array2::zeros((6, 6));
        let r = &self.r;

        // Upper-left 3x3 block
        for i in 0..3 {
            for j in 0..3 {
                m[[i, j]] = r[[i, j]] * r[[i, j]];
            }
        }

        // Upper-right 3x3 block
        m[[0, 3]] = BOND_TRANSFORM_FACTOR * r[[0, 1]] * r[[0, 2]];
        m[[0, 4]] = BOND_TRANSFORM_FACTOR * r[[0, 0]] * r[[0, 2]];
        m[[0, 5]] = BOND_TRANSFORM_FACTOR * r[[0, 0]] * r[[0, 1]];

        m[[1, 3]] = BOND_TRANSFORM_FACTOR * r[[1, 1]] * r[[1, 2]];
        m[[1, 4]] = BOND_TRANSFORM_FACTOR * r[[1, 0]] * r[[1, 2]];
        m[[1, 5]] = BOND_TRANSFORM_FACTOR * r[[1, 0]] * r[[1, 1]];

        m[[2, 3]] = BOND_TRANSFORM_FACTOR * r[[2, 1]] * r[[2, 2]];
        m[[2, 4]] = BOND_TRANSFORM_FACTOR * r[[2, 0]] * r[[2, 2]];
        m[[2, 5]] = BOND_TRANSFORM_FACTOR * r[[2, 0]] * r[[2, 1]];

        // Lower-left 3x3 block
        m[[3, 0]] = r[[1, 0]] * r[[2, 0]];
        m[[3, 1]] = r[[1, 1]] * r[[2, 1]];
        m[[3, 2]] = r[[1, 2]] * r[[2, 2]];

        m[[4, 0]] = r[[0, 0]] * r[[2, 0]];
        m[[4, 1]] = r[[0, 1]] * r[[2, 1]];
        m[[4, 2]] = r[[0, 2]] * r[[2, 2]];

        m[[5, 0]] = r[[0, 0]] * r[[1, 0]];
        m[[5, 1]] = r[[0, 1]] * r[[1, 1]];
        m[[5, 2]] = r[[0, 2]] * r[[1, 2]];

        // Lower-right 3x3 block (complex terms)
        m[[3, 3]] = r[[1, 1]] * r[[2, 2]] + r[[1, 2]] * r[[2, 1]];
        m[[3, 4]] = r[[1, 0]] * r[[2, 2]] + r[[1, 2]] * r[[2, 0]];
        m[[3, 5]] = r[[1, 0]] * r[[2, 1]] + r[[1, 1]] * r[[2, 0]];

        m[[4, 3]] = r[[0, 1]] * r[[2, 2]] + r[[0, 2]] * r[[2, 1]];
        m[[4, 4]] = r[[0, 0]] * r[[2, 2]] + r[[0, 2]] * r[[2, 0]];
        m[[4, 5]] = r[[0, 0]] * r[[2, 1]] + r[[0, 1]] * r[[2, 0]];

        m[[5, 3]] = r[[0, 1]] * r[[1, 2]] + r[[0, 2]] * r[[1, 1]];
        m[[5, 4]] = r[[0, 0]] * r[[1, 2]] + r[[0, 2]] * r[[1, 0]];
        m[[5, 5]] = r[[0, 0]] * r[[1, 1]] + r[[0, 1]] * r[[1, 0]];

        m
    }

    /// Get the rotation matrix
    pub fn matrix(&self) -> &Array2<f64> {
        &self.r
    }

    /// Check if rotation is valid (orthogonal with det = 1)
    pub fn is_valid(&self) -> bool {
        // Check orthogonality: R * R^T = I
        let identity = self.r.dot(&self.r.t());
        let mut is_orthogonal = true;

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (identity[[i, j]] - expected).abs() > 1e-10 {
                    is_orthogonal = false;
                    break;
                }
            }
        }

        // Check determinant = 1 (proper rotation)
        let det = self.determinant();
        is_orthogonal && (det - 1.0).abs() < 1e-10
    }

    /// Calculate determinant
    fn determinant(&self) -> f64 {
        let r = &self.r;
        r[[0, 0]] * (r[[1, 1]] * r[[2, 2]] - r[[1, 2]] * r[[2, 1]])
            - r[[0, 1]] * (r[[1, 0]] * r[[2, 2]] - r[[1, 2]] * r[[2, 0]])
            + r[[0, 2]] * (r[[1, 0]] * r[[2, 1]] - r[[1, 1]] * r[[2, 0]])
    }
}
