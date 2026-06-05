use ndarray::Array2;
use num_complex::Complex64;

use super::config::BurtonMillerConfig;
use kwavers_core::error::KwaversResult;

/// Burton-Miller System Matrix Builder.
///
/// Constructs combined CBIE + α·HBIE system matrix.
#[derive(Debug)]
pub struct BurtonMillerAssembler {
    pub(super) config: BurtonMillerConfig,
}

impl BurtonMillerAssembler {
    #[must_use]
    pub fn new(config: BurtonMillerConfig) -> Self {
        Self { config }
    }

    /// Compute area-weighted vertex normals for a triangular mesh (Gouraud shading).
    fn compute_vertex_normals(&self, nodes: &[[f64; 3]], elements: &[[usize; 3]]) -> Vec<[f64; 3]> {
        let mut normals: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; nodes.len()];

        for &elem in elements {
            let (n1, n2, n3) = (elem[0], elem[1], elem[2]);
            let tri_normal = self.triangle_normal(nodes[n1], nodes[n2], nodes[n3]);
            let area = self.triangle_area(nodes[n1], nodes[n2], nodes[n3]);
            for &v in &[n1, n2, n3] {
                normals[v][0] += area * tri_normal[0];
                normals[v][1] += area * tri_normal[1];
                normals[v][2] += area * tri_normal[2];
            }
        }

        for n in &mut normals {
            let len = n[2].mul_add(n[2], n[0].mul_add(n[0], n[1] * n[1])).sqrt();
            if len > 1e-14 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            } else {
                *n = [0.0, 0.0, 1.0];
            }
        }

        normals
    }

    /// Assemble Burton-Miller H matrix (CBIE + α·HBIE).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn assemble_h_matrix(
        &self,
        boundary_nodes: &[[f64; 3]],
        elements: &[[usize; 3]],
        num_collocation_points: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        let n = num_collocation_points;
        let mut h_matrix = Array2::zeros((n, n));
        let alpha = self.config.coupling_alpha;
        let vertex_normals = self.compute_vertex_normals(boundary_nodes, elements);

        for i in 0..n {
            let collocation_point = boundary_nodes[i];
            let collocation_normal = vertex_normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]);

            for (elem_idx, &elem) in elements.iter().enumerate() {
                let node1 = boundary_nodes[elem[0]];
                let node2 = boundary_nodes[elem[1]];
                let node3 = boundary_nodes[elem[2]];

                let (h_cbie, h_hbie) = self.element_contribution(
                    &collocation_point,
                    &collocation_normal,
                    node1,
                    node2,
                    node3,
                )?;

                for &global_node_idx in &elements[elem_idx] {
                    h_matrix[[i, global_node_idx]] += h_cbie;
                    h_matrix[[i, global_node_idx]] += alpha * h_hbie;
                }
            }

            h_matrix[[i, i]] += Complex64::new(0.5, 0.0) + alpha * Complex64::new(0.5, 0.0);
        }

        Ok(h_matrix)
    }

    /// Assemble Burton-Miller G matrix (Neumann data RHS).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn assemble_g_matrix(
        &self,
        boundary_nodes: &[[f64; 3]],
        elements: &[[usize; 3]],
        num_collocation_points: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        let n = num_collocation_points;
        let mut g_matrix = Array2::zeros((n, n));
        let alpha = self.config.coupling_alpha;
        let vertex_normals = self.compute_vertex_normals(boundary_nodes, elements);

        for i in 0..n {
            let collocation_point = boundary_nodes[i];
            let collocation_normal = vertex_normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]);

            for (elem_idx, &elem) in elements.iter().enumerate() {
                let node1 = boundary_nodes[elem[0]];
                let node2 = boundary_nodes[elem[1]];
                let node3 = boundary_nodes[elem[2]];

                let (g_cbie, g_hbie) = self.element_contribution_g(
                    &collocation_point,
                    &collocation_normal,
                    node1,
                    node2,
                    node3,
                )?;

                for &global_node_idx in &elements[elem_idx] {
                    g_matrix[[i, global_node_idx]] += g_cbie + alpha * g_hbie;
                }
            }
        }

        Ok(g_matrix)
    }

    /// Compute element contribution to H matrix (CBIE + HBIE).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn element_contribution(
        &self,
        collocation: &[f64; 3],
        collocation_normal: &[f64; 3],
        node1: [f64; 3],
        node2: [f64; 3],
        node3: [f64; 3],
    ) -> KwaversResult<(Complex64, Complex64)> {
        let k = self.config.wavenumber;
        let mut h_cbie = Complex64::new(0.0, 0.0);
        let mut h_hbie = Complex64::new(0.0, 0.0);

        let gauss_points: [(f64, f64); 3] = [(1.0 / 3.0, 1.0 / 3.0), (0.6, 0.2), (0.2, 0.6)];
        let gauss_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        let normal_y = self.triangle_normal(node1, node2, node3);

        for (gp_idx, &(xi, eta)) in gauss_points.iter().enumerate() {
            let zeta = 1.0 - xi - eta;
            let point_on_element = [
                eta.mul_add(node3[0], zeta * node1[0] + xi * node2[0]),
                eta.mul_add(node3[1], zeta * node1[1] + xi * node2[1]),
                eta.mul_add(node3[2], zeta * node1[2] + xi * node2[2]),
            ];

            let r = self.distance(collocation, &point_on_element);
            if r < self.config.singular_regularization {
                continue;
            }

            let dg_dn = self.greens_function_normal_derivative_full(
                k,
                r,
                collocation,
                &point_on_element,
                &normal_y,
            );
            h_cbie += gauss_weights[gp_idx] * dg_dn;

            let d2g_dndn = self.greens_function_double_normal_derivative(
                k,
                r,
                collocation,
                &point_on_element,
                &normal_y,
                collocation_normal,
            );
            h_hbie += gauss_weights[gp_idx] * d2g_dndn;
        }

        let element_area = self.triangle_area(node1, node2, node3);
        h_cbie *= element_area;
        h_hbie *= element_area;

        Ok((h_cbie, h_hbie))
    }

    /// Compute element contribution to G matrix (CBIE + HBIE).
    ///
    /// CBIE G-kernel: `G(x,y)` — the Helmholtz free-space Green's function.
    /// HBIE G-kernel: `∂G/∂n_x(x,y) = G·(ik−1/R)·(x−y)·n_x/R` — the normal
    /// derivative of G with respect to the collocation point x.
    ///
    /// To reuse `greens_function_normal_derivative_full` (which computes
    /// `∂G/∂n_y = G·(ik−1/R)·(y−x)·n_y/R`), the source and observer arguments
    /// are **swapped**: passing `r_src = point_on_element` and `r_obs = collocation`
    /// yields `rhat = (collocation−point)/R`, giving `∂G/∂n_x` correctly.
    fn element_contribution_g(
        &self,
        collocation: &[f64; 3],
        collocation_normal: &[f64; 3],
        node1: [f64; 3],
        node2: [f64; 3],
        node3: [f64; 3],
    ) -> KwaversResult<(Complex64, Complex64)> {
        let k = self.config.wavenumber;
        let mut g_cbie = Complex64::new(0.0, 0.0);
        let mut g_hbie = Complex64::new(0.0, 0.0);

        let gauss_points: [(f64, f64); 3] = [(1.0 / 3.0, 1.0 / 3.0), (0.6, 0.2), (0.2, 0.6)];
        let gauss_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        for (gp_idx, &(xi, eta)) in gauss_points.iter().enumerate() {
            let zeta = 1.0 - xi - eta;
            let point_on_element = [
                eta.mul_add(node3[0], zeta * node1[0] + xi * node2[0]),
                eta.mul_add(node3[1], zeta * node1[1] + xi * node2[1]),
                eta.mul_add(node3[2], zeta * node1[2] + xi * node2[2]),
            ];

            let r = self.distance(collocation, &point_on_element);
            if r < self.config.singular_regularization {
                continue;
            }

            let g = self.greens_function_helmholtz(k, r);
            g_cbie += gauss_weights[gp_idx] * g;

            // ∂G/∂n_x: swap src/obs so rhat = (collocation − point)/R
            let dg_dnx = self.greens_function_normal_derivative_full(
                k,
                r,
                &point_on_element,
                collocation,
                collocation_normal,
            );
            g_hbie += gauss_weights[gp_idx] * dg_dnx;
        }

        let element_area = self.triangle_area(node1, node2, node3);
        g_cbie *= element_area;
        g_hbie *= element_area;

        Ok((g_cbie, g_hbie))
    }
}
