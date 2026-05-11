use super::FemHelmholtzSolver;
use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use nalgebra::{Matrix3, Vector3};
use ndarray::{Array1, ArrayView2};
use num_complex::Complex64;

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
        let a = Vector3::new(p0[0], p0[1], p0[2]);
        let b = Vector3::new(p1[0], p1[1], p1[2]);
        let c = Vector3::new(p2[0], p2[1], p2[2]);
        let d = Vector3::new(p3[0], p3[1], p3[2]);
        let p = Vector3::new(point[0], point[1], point[2]);

        let m = Matrix3::from_columns(&[b - a, c - a, d - a]);
        let inv = m.try_inverse().ok_or_else(|| {
            KwaversError::Numerical(NumericalError::SingularMatrix {
                operation: "element_interpolation".to_owned(),
                condition_number: 0.0,
            })
        })?;

        let uvw = inv * (p - a);
        let u = uvw[0];
        let v = uvw[1];
        let w = uvw[2];
        let t = 1.0 - u - v - w;

        Ok((u, v, w, t))
    }
}
