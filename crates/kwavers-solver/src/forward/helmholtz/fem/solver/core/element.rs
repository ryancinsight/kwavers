use super::FemHelmholtzSolver;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

type Vec3 = [f64; 3];
type Mat3 = [[f64; 3]; 3];

/// Per-element FEM assembly arrays: stiffness `K_e`, consistent mass `M_e`, and
/// RHS `f_e`, one entry per mesh element.
type ElementMatrices = (
    Vec<Array2<Complex64>>,
    Vec<Array2<Complex64>>,
    Vec<Array1<Complex64>>,
);

impl FemHelmholtzSolver {
    /// Compute per-element stiffness K_e, consistent mass M_e, and RHS f_e.
    ///
    /// P1 reference-element gradients:
    /// ```text
    /// ∇ξ₀ = (−1,−1,−1), ∇ξ₁ = (1,0,0), ∇ξ₂ = (0,1,0), ∇ξ₃ = (0,0,1)
    /// ∇φᵢ = J^{−T} ∇ξᵢ
    /// ```
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn compute_element_matrices(&self) -> KwaversResult<ElementMatrices> {
        let mut element_stiffness = Vec::with_capacity(self.mesh.elements.len());
        let mut element_mass = Vec::with_capacity(self.mesh.elements.len());
        let mut element_rhs = Vec::with_capacity(self.mesh.elements.len());

        for element in &self.mesh.elements {
            let p0 = self.mesh.nodes[element.nodes[0]].coordinates;
            let p1 = self.mesh.nodes[element.nodes[1]].coordinates;
            let p2 = self.mesh.nodes[element.nodes[2]].coordinates;
            let p3 = self.mesh.nodes[element.nodes[3]].coordinates;

            // Jacobian J = [p₁−p₀ | p₂−p₀ | p₃−p₀]
            let j_mat = mat3_from_cols(v_sub(p1, p0), v_sub(p2, p0), v_sub(p3, p0));
            let det_j = mat3_determinant(j_mat);
            let volume = det_j.abs() / 6.0;

            if volume < 1e-14 {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "element_assembly".to_owned(),
                    condition_number: 0.0,
                }));
            }

            let j_inv_t = mat3_transpose(mat3_inverse(j_mat).ok_or(
                KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "jacobian_inverse".to_owned(),
                    condition_number: 0.0,
                }),
            )?);

            let grad_phi_ref = [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ];

            let mut grad_phi_phys = [[0.0; 3]; 4];
            for k in 0..4 {
                grad_phi_phys[k] = mat3_mul_vec(j_inv_t, grad_phi_ref[k]);
            }

            // Stiffness: K_ij = V (∇φᵢ · ∇φⱼ)
            let mut k_elem = Array2::<Complex64>::zeros((4, 4));
            for r in 0..4 {
                for c in 0..4 {
                    k_elem[[r, c]] =
                        Complex64::from(v_dot(grad_phi_phys[r], grad_phi_phys[c]) * volume);
                }
            }

            // Consistent mass: M_ii = V/10, M_ij = V/20 (i≠j)
            let mut m_elem = Array2::<Complex64>::zeros((4, 4));
            let v_over_20 = Complex64::from(volume / 20.0);
            for r in 0..4 {
                for c in 0..4 {
                    m_elem[[r, c]] = if r == c { v_over_20 * 2.0 } else { v_over_20 };
                }
            }

            element_stiffness.push(k_elem);
            element_mass.push(m_elem);
            element_rhs.push(Array1::<Complex64>::zeros(4));
        }

        Ok((element_stiffness, element_mass, element_rhs))
    }
}

#[inline]
fn v_sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn v_dot(a: Vec3, b: Vec3) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

#[inline]
fn mat3_from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3 {
    [[c0[0], c1[0], c2[0]], [c0[1], c1[1], c2[1]], [c0[2], c1[2], c2[2]]]
}

#[inline]
fn mat3_mul_vec(m: Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0].mul_add(v[0], m[0][1].mul_add(v[1], m[0][2] * v[2])),
        m[1][0].mul_add(v[0], m[1][1].mul_add(v[1], m[1][2] * v[2])),
        m[2][0].mul_add(v[0], m[2][1].mul_add(v[1], m[2][2] * v[2])),
    ]
}

#[inline]
fn mat3_transpose(m: Mat3) -> Mat3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[inline]
fn mat3_determinant(m: Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[inline]
fn mat3_inverse(m: Mat3) -> Option<Mat3> {
    let det = mat3_determinant(m);
    if det.abs() < 1e-14 {
        return None;
    }

    let inv_det = 1.0 / det;
    let cofactor = [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            -(m[1][0] * m[2][2] - m[1][2] * m[2][0]),
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
        ],
        [
            -(m[0][1] * m[2][2] - m[0][2] * m[2][1]),
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            -(m[0][0] * m[2][1] - m[0][1] * m[2][0]),
        ],
        [
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
            -(m[0][0] * m[1][2] - m[0][2] * m[1][0]),
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ];

    Some([
        [cofactor[0][0] * inv_det, cofactor[1][0] * inv_det, cofactor[2][0] * inv_det],
        [cofactor[0][1] * inv_det, cofactor[1][1] * inv_det, cofactor[2][1] * inv_det],
        [cofactor[0][2] * inv_det, cofactor[1][2] * inv_det, cofactor[2][2] * inv_det],
    ])
}
