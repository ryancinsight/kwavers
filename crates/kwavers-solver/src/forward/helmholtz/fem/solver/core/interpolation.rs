use super::FemHelmholtzSolver;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::{Array1, ArrayView2};
use num_complex::Complex64;

type Vec3 = [f64; 3];
type Mat3 = [[f64; 3]; 3];

impl FemHelmholtzSolver {
    /// Interpolate the nodal solution at arbitrary query points via barycentric coordinates.
    ///
    /// Returns zero for query points outside the mesh domain.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn interpolate_solution(
        &self,
        query_points: ArrayView2<f64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let num_points = query_points.nrows();
        let mut results = Array1::zeros(num_points);

        for (i, row) in query_points.outer_iter().enumerate() {
            let point = [row[0], row[1], row[2]];
            let elements = self.mesh.locate_point(point);

            if let Some(&elem_idx) = elements.first() {
                let element = &self.mesh.elements[elem_idx];
                let nodes = element.nodes;

                let p0 = self.mesh.nodes[nodes[0]].coordinates;
                let p1 = self.mesh.nodes[nodes[1]].coordinates;
                let p2 = self.mesh.nodes[nodes[2]].coordinates;
                let p3 = self.mesh.nodes[nodes[3]].coordinates;

                let (u, v, w, t) = self.compute_shape_functions(point, p0, p1, p2, p3)?;

                results[i] = self.solution[nodes[0]] * Complex64::from(t)
                    + self.solution[nodes[1]] * Complex64::from(u)
                    + self.solution[nodes[2]] * Complex64::from(v)
                    + self.solution[nodes[3]] * Complex64::from(w);
            }
            // Point outside mesh → result[i] stays 0.0
        }

        Ok(results)
    }

    /// Compute barycentric coordinates (u, v, w, t) for `point` inside tetrahedron {p0..p3}.
    ///
    /// Maps physical coordinates to reference coordinates via J^{-1}:
    /// ```text
    /// [u, v, w]ᵀ = J^{-1} (point − p₀),  t = 1 − u − v − w
    /// ```
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn compute_shape_functions(
        &self,
        point: [f64; 3],
        p0: [f64; 3],
        p1: [f64; 3],
        p2: [f64; 3],
        p3: [f64; 3],
    ) -> KwaversResult<(f64, f64, f64, f64)> {
        let m = mat3_from_cols(v_sub(p1, p0), v_sub(p2, p0), v_sub(p3, p0));
        let inv = mat3_inverse(m).ok_or_else(|| {
            KwaversError::Numerical(NumericalError::SingularMatrix {
                operation: "element_interpolation".to_owned(),
                condition_number: 0.0,
            })
        })?;

        let uvw = mat3_mul_vec(inv, v_sub(point, p0));
        let u = uvw[0];
        let v = uvw[1];
        let w = uvw[2];
        let t = 1.0 - u - v - w;

        Ok((u, v, w, t))
    }
}

#[inline]
fn v_sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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

    // inverse = adjugate / det = transpose(cofactor) / det
    Some([
        [cofactor[0][0] * inv_det, cofactor[1][0] * inv_det, cofactor[2][0] * inv_det],
        [cofactor[0][1] * inv_det, cofactor[1][1] * inv_det, cofactor[2][1] * inv_det],
        [cofactor[0][2] * inv_det, cofactor[1][2] * inv_det, cofactor[2][2] * inv_det],
    ])
}
