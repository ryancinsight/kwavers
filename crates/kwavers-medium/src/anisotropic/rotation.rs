//! Rotation matrices and transformations for anisotropic materials

use kwavers_core::constants::BOND_TRANSFORM_FACTOR;
use ndarray::Array2;

/// 3D rotation matrix for coordinate transformations
#[derive(Debug, Clone)]
pub struct RotationMatrix {
    /// 3x3 rotation matrix
    r: Array2<f64>,
}

impl RotationMatrix {
    /// Create rotation matrix from Euler angles (ZYX intrinsic convention).
    ///
    /// `phi` rotates about Z, `theta` about Y, `psi` about X (intrinsic / body-fixed):
    /// `R = R_z(phi) · R_y(theta) · R_x(psi)`. Reference: Goldstein, *Classical
    /// Mechanics* (3rd ed.), §4.4.
    #[must_use]
    pub fn from_euler(phi: f64, theta: f64, psi: f64) -> Self {
        let (sp, cp) = phi.sin_cos();
        let (st, ct) = theta.sin_cos();
        let (ss, cs) = psi.sin_cos();

        let mut r = Array2::zeros((3, 3));

        // Row 0: [cp·ct,  cp·st·ss − sp·cs,  cp·st·cs + sp·ss]
        r[[0, 0]] = cp * ct;
        r[[0, 1]] = (cp * st).mul_add(ss, -(sp * cs));
        r[[0, 2]] = (cp * st).mul_add(cs, sp * ss);

        // Row 1: [sp·ct,  sp·st·ss + cp·cs,  sp·st·cs − cp·ss]
        r[[1, 0]] = sp * ct;
        r[[1, 1]] = (sp * st).mul_add(ss, cp * cs);
        r[[1, 2]] = (sp * st).mul_add(cs, -(cp * ss));

        // Row 2: [−st,  ct·ss,  ct·cs]
        r[[2, 0]] = -st;
        r[[2, 1]] = ct * ss;
        r[[2, 2]] = ct * cs;

        Self { r }
    }

    /// Create rotation around Z axis
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn apply_to_vector(&self, v: &[f64; 3]) -> [f64; 3] {
        let mut result = [0.0; 3];
        for (i, result_item) in result.iter_mut().enumerate() {
            for (j, &v_item) in v.iter().enumerate().take(3) {
                *result_item += self.r[[i, j]] * v_item;
            }
        }
        result
    }

    /// Apply rotation to stiffness tensor (Bond transformation)
    #[must_use]
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
        m[[3, 3]] = r[[1, 1]].mul_add(r[[2, 2]], r[[1, 2]] * r[[2, 1]]);
        m[[3, 4]] = r[[1, 0]].mul_add(r[[2, 2]], r[[1, 2]] * r[[2, 0]]);
        m[[3, 5]] = r[[1, 0]].mul_add(r[[2, 1]], r[[1, 1]] * r[[2, 0]]);

        m[[4, 3]] = r[[0, 1]].mul_add(r[[2, 2]], r[[0, 2]] * r[[2, 1]]);
        m[[4, 4]] = r[[0, 0]].mul_add(r[[2, 2]], r[[0, 2]] * r[[2, 0]]);
        m[[4, 5]] = r[[0, 0]].mul_add(r[[2, 1]], r[[0, 1]] * r[[2, 0]]);

        m[[5, 3]] = r[[0, 1]].mul_add(r[[1, 2]], r[[0, 2]] * r[[1, 1]]);
        m[[5, 4]] = r[[0, 0]].mul_add(r[[1, 2]], r[[0, 2]] * r[[1, 0]]);
        m[[5, 5]] = r[[0, 0]].mul_add(r[[1, 1]], r[[0, 1]] * r[[1, 0]]);

        m
    }

    /// Get the rotation matrix
    #[must_use]
    pub fn matrix(&self) -> &Array2<f64> {
        &self.r
    }

    /// Check if rotation is valid (orthogonal with det = 1)
    #[must_use]
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
        r[[0, 2]].mul_add(
            r[[1, 0]].mul_add(r[[2, 1]], -(r[[1, 1]] * r[[2, 0]])),
            r[[0, 0]].mul_add(
                r[[1, 1]].mul_add(r[[2, 2]], -(r[[1, 2]] * r[[2, 1]])),
                -(r[[0, 1]] * r[[1, 0]].mul_add(r[[2, 2]], -(r[[1, 2]] * r[[2, 0]]))),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    /// `from_euler(0, 0, 0)` must be the identity matrix.
    #[test]
    fn from_euler_zero_is_identity() {
        let rot = RotationMatrix::from_euler(0.0, 0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rot.r[[i, j]] - expected).abs() < 1e-12,
                    "identity element ({i},{j}): got {got}, expected {expected}",
                    got = rot.r[[i, j]],
                );
            }
        }
        assert!(rot.is_valid());
    }

    /// `from_euler` must depend on every Euler angle. Prior to the 2026-05-21
    /// fix `sin(psi)` was discarded (`let (_ss, cs)`) so rotations about the
    /// body-fixed X axis silently degenerated to the identity in 6 of 9
    /// matrix entries.
    #[test]
    fn from_euler_depends_on_all_three_angles() {
        let psi_zero = RotationMatrix::from_euler(0.0, 0.0, 0.0);
        let psi_quarter = RotationMatrix::from_euler(0.0, 0.0, FRAC_PI_2);
        let max_diff = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (psi_quarter.r[[i, j]] - psi_zero.r[[i, j]]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 0.5,
            "rotation matrix must change when psi rotates by π/2, max delta = {max_diff}"
        );
    }

    /// `from_euler` must produce an orthogonal matrix with determinant +1
    /// for arbitrary angles.
    #[test]
    fn from_euler_is_proper_rotation() {
        for &(phi, theta, psi) in &[
            (0.3_f64, 0.7, -0.4),
            (1.2, -0.9, 2.1),
            (FRAC_PI_2, FRAC_PI_2, FRAC_PI_2),
        ] {
            let rot = RotationMatrix::from_euler(phi, theta, psi);
            assert!(
                rot.is_valid(),
                "rotation ({phi},{theta},{psi}) is not orthogonal with det=1"
            );
        }
    }

    /// `from_euler(phi, 0, 0)` must equal a pure Z-axis rotation by `phi`.
    #[test]
    fn from_euler_phi_only_equals_z_rotation() {
        let phi = 0.6_f64;
        let from_zyx = RotationMatrix::from_euler(phi, 0.0, 0.0);
        let z_only = RotationMatrix::z_rotation(phi);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (from_zyx.r[[i, j]] - z_only.r[[i, j]]).abs() < 1e-12,
                    "({i},{j}): zyx={zyx}, z={z}",
                    zyx = from_zyx.r[[i, j]],
                    z = z_only.r[[i, j]],
                );
            }
        }
    }

    /// Bond transformation of an isotropic stiffness tensor must leave it
    /// invariant under arbitrary rotations (anisotropy is rotation-symmetric).
    #[test]
    fn bond_transform_preserves_isotropic_stiffness() {
        let lambda = 5.0e9_f64;
        let mu = 3.0e9_f64;
        let mut c = Array2::zeros((6, 6));
        for i in 0..3 {
            for j in 0..3 {
                c[[i, j]] = if i == j { lambda + 2.0 * mu } else { lambda };
            }
        }
        for i in 3..6 {
            c[[i, i]] = mu;
        }

        let rot = RotationMatrix::from_euler(0.4, -0.7, 1.1);
        let c_rot = rot.apply_to_stiffness(&c);

        // Reference scale: any nonzero element of the unrotated C.
        let scale = lambda + 2.0 * mu;
        // Floating-point roundoff in a chained 6x6 · 6x6 · 6x6 multiplication
        // accumulates ~6·6 multiplies per element ≈ O(1e-14)·scale.
        let tol = 1.0e-12 * scale;
        for i in 0..6 {
            for j in 0..6 {
                let abs_err = (c_rot[[i, j]] - c[[i, j]]).abs();
                assert!(
                    abs_err < tol,
                    "Bond rotation of isotropic C broke at ({i},{j}): \
                     original={orig}, rotated={rot_v}, abs_err={abs_err}, tol={tol}",
                    orig = c[[i, j]],
                    rot_v = c_rot[[i, j]],
                );
            }
        }
    }
}
