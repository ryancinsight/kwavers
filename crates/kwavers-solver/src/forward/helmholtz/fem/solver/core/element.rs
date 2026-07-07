use super::FemHelmholtzSolver;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::geometry::Vector3;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Per-element FEM assembly arrays: stiffness `K_e`, consistent mass `M_e`, and
/// RHS `f_e`, one entry per mesh element.
type ElementMatrices = (
    Vec<Array2<Complex64>>,
    Vec<Array2<Complex64>>,
    Vec<Array1<Complex64>>,
);

fn inverse_rows_from_columns(
    col0: Vector3<f64>,
    col1: Vector3<f64>,
    col2: Vector3<f64>,
) -> Option<[Vector3<f64>; 3]> {
    let det = col0.dot(col1.cross(col2));
    if det.abs() < 1e-14 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        col1.cross(col2) * inv_det,
        col2.cross(col0) * inv_det,
        col0.cross(col1) * inv_det,
    ])
}

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

            let v0 = Vector3::new(p0[0], p0[1], p0[2]);
            let v1 = Vector3::new(p1[0], p1[1], p1[2]);
            let v2 = Vector3::new(p2[0], p2[1], p2[2]);
            let v3 = Vector3::new(p3[0], p3[1], p3[2]);

            // Jacobian J = [p₁−p₀ | p₂−p₀ | p₃−p₀]
            let col0 = v1 - v0;
            let col1 = v2 - v0;
            let col2 = v3 - v0;
            let det_j = col0.dot(col1.cross(col2));
            let volume = det_j.abs() / 6.0;

            if volume < 1e-14 {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "element_assembly".to_owned(),
                    condition_number: 0.0,
                }));
            }

            let inv_rows = inverse_rows_from_columns(col0, col1, col2).ok_or(
                KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "jacobian_inverse".to_owned(),
                    condition_number: 0.0,
                }),
            )?;

            let grad_phi_ref = [
                Vector3::new(-1.0, -1.0, -1.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ];

            let mut grad_phi_phys = [Vector3::zeros(); 4];
            for k in 0..4 {
                grad_phi_phys[k] = inv_rows[0] * grad_phi_ref[k].x
                    + inv_rows[1] * grad_phi_ref[k].y
                    + inv_rows[2] * grad_phi_ref[k].z;
            }

            // Stiffness: K_ij = V (∇φᵢ · ∇φⱼ)
            let mut k_elem = Array2::<Complex64>::zeros((4, 4));
            for r in 0..4 {
                for c in 0..4 {
                    k_elem[[r, c]] =
                        Complex64::from(grad_phi_phys[r].dot(grad_phi_phys[c]) * volume);
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
