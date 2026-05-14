use crate::core::error::KwaversResult;
use ndarray::Array2;

/// Registration transformation (rigid or affine)
#[derive(Debug, Clone)]
pub struct RegistrationTransform {
    /// Transformation type
    pub transform_type: TransformationType,
    /// 4×4 transformation matrix (homogeneous coordinates)
    pub matrix: Array2<f64>,
    /// Registration error (RMS distance after alignment)
    pub registration_error_mm: f64,
    /// Number of iterations until convergence
    pub iterations: usize,
}

/// Transformation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformationType {
    /// Rigid (6 DOF): translation + rotation
    Rigid,
    /// Affine (12 DOF): rigid + scaling + shear
    Affine,
    /// Non-rigid (many DOF): deformable registration
    NonRigid,
}

impl RegistrationTransform {
    /// Create identity transformation
    #[must_use]
    pub fn identity() -> Self {
        let mut matrix = Array2::zeros((4, 4));
        for i in 0..4 {
            matrix[[i, i]] = 1.0;
        }

        Self {
            transform_type: TransformationType::Rigid,
            matrix,
            registration_error_mm: 0.0,
            iterations: 0,
        }
    }

    /// Apply transformation to 3D point
    #[must_use]
    pub fn apply_to_point(&self, point: (f64, f64, f64)) -> (f64, f64, f64) {
        // Convert to homogeneous coordinates
        let p = [point.0, point.1, point.2, 1.0];

        // Apply transformation
        let mut result = [0.0; 4];
        for (i, item) in result.iter_mut().enumerate() {
            *item = (0..4).map(|j| self.matrix[[i, j]] * p[j]).sum();
        }

        // Convert back to 3D
        (
            result[0] / result[3],
            result[1] / result[3],
            result[2] / result[3],
        )
    }

    /// Invert a homogeneous rigid-body transformation.
    ///
    /// ## Theorem — Inverse of a 4×4 Homogeneous Rigid Transform
    ///
    /// A rigid-body transform in homogeneous coordinates is:
    /// ```text
    ///   T = [ R  t ]    where R ∈ SO(3), t ∈ ℝ³
    ///       [ 0  1 ]
    /// ```
    ///
    /// Its exact inverse is:
    /// ```text
    ///   T⁻¹ = [ Rᵀ   −Rᵀ·t ]
    ///         [ 0       1   ]
    /// ```
    ///
    /// **Proof**: T · T⁻¹ =
    ///   [ R  t ] · [ Rᵀ  −Rᵀt ] = [ R·Rᵀ  −R·Rᵀt + t ] = [ I  0 ] = I₄  ✓
    ///   [ 0  1 ]   [ 0     1  ]   [ 0         1       ]   [ 0  1 ]
    ///
    /// Since R ∈ SO(3), R·Rᵀ = I₃ (orthogonality) and det(R) = 1.
    ///
    /// This formula is exact (no numerical LU decomposition needed) and produces
    /// a result that is guaranteed to lie in SE(3) — the naive LU inverse of a
    /// rotation matrix can drift out of SO(3) due to floating-point rounding.
    ///
    /// ## Reference
    ///
    /// Craig, J.J. (2005). *Introduction to Robotics: Mechanics and Control*.
    /// 3rd ed. Pearson. §2.3, Eq. 2.45.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn invert(&self) -> KwaversResult<Self> {
        // Extract R (3×3 upper-left) and t (column 3, rows 0..3) from the 4×4 matrix.
        // Matrix is stored row-major: self.matrix[[row, col]].
        let m = &self.matrix;

        // Rᵀ (3×3): element [i,j] of Rᵀ = element [j,i] of R
        let mut rt = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rt[i][j] = m[[j, i]];
            }
        }

        // −Rᵀ·t (3×1)
        let t = [m[[0, 3]], m[[1, 3]], m[[2, 3]]];
        let mut neg_rt_t = [0.0_f64; 3];
        for i in 0..3 {
            neg_rt_t[i] = -rt[i][2].mul_add(t[2], rt[i][0].mul_add(t[0], rt[i][1] * t[1]));
        }

        // Assemble 4×4 inverse matrix
        let mut inv = Array2::<f64>::zeros((4, 4));
        for i in 0..3 {
            for j in 0..3 {
                inv[[i, j]] = rt[i][j];
            }
            inv[[i, 3]] = neg_rt_t[i];
        }
        inv[[3, 3]] = 1.0;

        Ok(Self {
            transform_type: self.transform_type,
            matrix: inv,
            registration_error_mm: self.registration_error_mm,
            iterations: self.iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registration_transform_identity() {
        let transform = RegistrationTransform::identity();
        assert_eq!(transform.transform_type, TransformationType::Rigid);

        // Identity transform should preserve coordinates
        let point = (10.0, 20.0, 30.0);
        let result = transform.apply_to_point(point);
        assert!((result.0 - point.0).abs() < 0.01);
        assert!((result.1 - point.1).abs() < 0.01);
        assert!((result.2 - point.2).abs() < 0.01);
    }

    /// Identity transform inverted must still be identity.
    ///
    /// T = I₄  ⟹  T⁻¹ = I₄  (Craig 2005, §2.3)
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_invert_identity() {
        let id = RegistrationTransform::identity();
        let inv = id.invert().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv.matrix[[i, j]] - expected).abs() < 1e-14,
                    "identity.invert()[[{i},{j}]] = {}, expected {expected}",
                    inv.matrix[[i, j]]
                );
            }
        }
    }

    /// A pure translation T(d) inverted must give T(-d).
    ///
    /// T = [ I  d ]  ⟹  T⁻¹ = [ I  -d ]
    ///     [ 0  1 ]             [ 0   1 ]
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_invert_pure_translation() {
        let dx = 5.0_f64;
        let mut t = RegistrationTransform::identity();
        t.matrix[[0, 3]] = dx;

        let inv = t.invert().unwrap();
        assert!(
            (inv.matrix[[0, 3]] - (-dx)).abs() < 1e-14,
            "translation inverse x = {}, expected {}",
            inv.matrix[[0, 3]],
            -dx
        );
        assert!((inv.matrix[[1, 3]]).abs() < 1e-14);
        assert!((inv.matrix[[2, 3]]).abs() < 1e-14);
    }

    /// Round-trip T · T⁻¹ ≈ I₄ for a combined rotation + translation.
    ///
    /// Uses a 90° rotation about the z-axis with a non-trivial translation.
    /// The round-trip error must be below f64 machine epsilon × grid_size (12).
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_invert_round_trip() {
        use std::f64::consts::FRAC_PI_2;

        // 90° rotation about z: R_z(π/2) = [[0,-1,0],[1,0,0],[0,0,1]]
        let mut t = RegistrationTransform::identity();
        t.matrix[[0, 0]] = FRAC_PI_2.cos();
        t.matrix[[0, 1]] = -FRAC_PI_2.sin();
        t.matrix[[1, 0]] = FRAC_PI_2.sin();
        t.matrix[[1, 1]] = FRAC_PI_2.cos();
        // Translation
        t.matrix[[0, 3]] = 3.0;
        t.matrix[[1, 3]] = -7.0;
        t.matrix[[2, 3]] = 1.5;

        let inv = t.invert().unwrap();

        // Compute T · T⁻¹
        let mut product = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    product[[i, j]] += t.matrix[[i, k]] * inv.matrix[[k, j]];
                }
            }
        }

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (product[[i, j]] - expected).abs();
                assert!(
                    err < 1e-12,
                    "T·T⁻¹[[{i},{j}]] = {:.2e} error, expected ~0",
                    err
                );
            }
        }
    }
}
