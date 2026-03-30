//! Burton-Miller Formulation for Boundary Element Method
//!
//! The Burton-Miller formulation combines the direct and indirect boundary integral
//! equations to eliminate spurious resonances (fictitious eigenfrequencies) that occur
//! in standard BEM at the natural frequencies of the interior domain.
//!
//! **Problem**: Standard CBIE/BCIE have non-uniqueness at interior resonances
//! **Solution**: Linear combination of both formulations gives unique solution everywhere
//!
//! **Mathematical Foundation**:
//!
//! Standard Combined Integral Equation (CBIE):
//! ```text
//! c(r)u(r) + ∫_Γ [∂G(r,r')/∂n(r')] u(r') dΓ = ∫_Γ G(r,r') t(r') dΓ
//! ```
//!
//! where:
//! - c(r) = 1 for r inside domain, 1/2 for r on boundary, 0 for r outside
//! - G = exp(ik|r-r'|)/(4π|r-r'|) (free space Green's function for Helmholtz)
//! - t = ∂u/∂n (normal derivative)
//!
//! Hypersingular Integral Equation (HBIE):
//! ```text
//! d(r)t(r) - ∫_Γ [∂²G(r,r')/∂n(r)∂n(r')] u(r') dΓ = ∫_Γ [∂G(r,r')/∂n(r)] t(r') dΓ
//! ```
//!
//! where:
//! - d(r) = 1/2 for r on smooth boundary
//! - Second term requires careful regularization (strongly singular integral)
//!
//! **Burton-Miller Formulation** (Burton & Miller 1971):
//! ```text
//! [CBIE] + α·[HBIE] = combined equation
//! ```
//!
//! The parameter α (coupling parameter) typically:
//! - α = 0: pure CBIE (non-unique at interior resonances)
//! - α = 1/ik: optimal choice at most frequencies (Chen & Hong 1999)
//! - α = i/(ω·ρ·c): alternative with physical meaning
//!
//! **Advantages**:
//! 1. **Unique Solution**: Valid everywhere, no spurious resonances
//! 2. **Robustness**: Works across frequency range without special treatment
//! 3. **Convergence**: Better condition number than pure CBIE
//!
//! **Implementation Considerations**:
//! 1. Hypersingular integral ∂²G/∂n∂n requires regularization
//! 2. Integration more expensive (second derivatives of Green's function)
//! 3. Better suited for exterior acoustic problems
//!
//! **References**:
//! - Burton, A. J., & Miller, G. F. (1971). "The application of integral equation methods
//!   to the numerical solution of some exterior boundary-value problems."
//!   *Proceedings of the Royal Society A*, 323(1553), 201-210.
//! - Chen, J. T., & Hong, H. K. (1999). "Review of dual boundary element methods with emphasis
//!   on hypersingular integrals and divergent series." *Applied Mechanics Reviews*, 52(1), 17-33.
//! - Marburg, S., & Schneider, S. (Eds.). (2015). *Computational Acoustics of Noise Propagation
//!   in Fluids-Finite and Boundary Element Methods*. Springer.

use crate::core::error::KwaversResult;
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Configuration for Burton-Miller BEM formulation
#[derive(Debug, Clone, Copy)]
pub struct BurtonMillerConfig {
    /// Wavenumber k = 2πf/c
    pub wavenumber: f64,
    /// Coupling parameter α (default: 1/ik = -i/k)
    /// For wavenumber k, α = Complex64::new(0.0, -1.0) / k
    pub coupling_alpha: Complex64,
    /// Frequency for reference (Hz)
    pub frequency: f64,
    /// Medium sound speed (m/s)
    pub sound_speed: f64,
    /// Small value for regularization of singular integrals
    pub singular_regularization: f64,
    /// Tolerance for CBIE and HBIE assembly
    pub assembly_tolerance: f64,
}

impl BurtonMillerConfig {
    /// Create new Burton-Miller configuration
    pub fn new(frequency: f64, sound_speed: f64) -> Self {
        let wavenumber = 2.0 * PI * frequency / sound_speed;
        // Optimal coupling: α = 1/(ik) = -i/k
        let coupling_alpha = Complex64::new(0.0, -1.0 / wavenumber);

        Self {
            wavenumber,
            coupling_alpha,
            frequency,
            sound_speed,
            singular_regularization: 1e-10,
            assembly_tolerance: 1e-12,
        }
    }

    /// Set custom coupling parameter
    pub fn with_coupling_alpha(mut self, alpha: Complex64) -> Self {
        self.coupling_alpha = alpha;
        self
    }
}

/// Burton-Miller System Matrix Builder
///
/// Constructs combined CBIE + α·HBIE system matrix
#[derive(Debug)]
pub struct BurtonMillerAssembler {
    config: BurtonMillerConfig,
}

impl BurtonMillerAssembler {
    /// Create new assembler
    pub fn new(config: BurtonMillerConfig) -> Self {
        Self { config }
    }

    /// Compute area-weighted vertex normals for a triangular mesh.
    ///
    /// ## Algorithm (Gouraud shading / FEM normal averaging)
    ///
    /// For each vertex v, the vertex normal is the area-weighted sum of the outward
    /// normals of all triangles incident on v, normalized to unit length:
    /// ```text
    /// n_v = (Σ_{T: v ∈ T} A_T · n_T) / |Σ_{T: v ∈ T} A_T · n_T|
    /// ```
    /// where A_T is the area of triangle T and n_T is its outward unit normal.
    ///
    /// ## Reference
    /// - Gouraud, H. (1971). Continuous shading of curved surfaces. IEEE Trans. Comput. C-20(6).
    fn compute_vertex_normals(
        &self,
        nodes: &[[f64; 3]],
        elements: &[[usize; 3]],
    ) -> Vec<[f64; 3]> {
        let mut normals: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; nodes.len()];

        for &elem in elements {
            let (n1, n2, n3) = (elem[0], elem[1], elem[2]);
            let tri_normal = self.triangle_normal(nodes[n1], nodes[n2], nodes[n3]);
            let area = self.triangle_area(nodes[n1], nodes[n2], nodes[n3]);
            // Accumulate area-weighted normal for each vertex
            for &v in &[n1, n2, n3] {
                normals[v][0] += area * tri_normal[0];
                normals[v][1] += area * tri_normal[1];
                normals[v][2] += area * tri_normal[2];
            }
        }

        // Normalise each vertex normal
        for n in &mut normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-14 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            } else {
                *n = [0.0, 0.0, 1.0]; // Fallback: z-pointing normal
            }
        }

        normals
    }

    /// Assemble Burton-Miller system matrix
    ///
    /// Returns matrix H_combined = [CBIE contribution] + α·[HBIE contribution]
    ///
    /// # Arguments
    /// * `boundary_nodes` - Boundary element nodes (x, y, z)
    /// * `elements` - Triangle elements (3 node indices each)
    /// * `num_collocation_points` - Number of collocation points (usually equals number of nodes)
    pub fn assemble_h_matrix(
        &self,
        boundary_nodes: &[[f64; 3]],
        elements: &[[usize; 3]],
        num_collocation_points: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        let n = num_collocation_points;
        let mut h_matrix = Array2::zeros((n, n));

        let _k = self.config.wavenumber;
        let alpha = self.config.coupling_alpha;

        // Precompute area-weighted vertex normals for the hypersingular (HBIE) kernel
        let vertex_normals = self.compute_vertex_normals(boundary_nodes, elements);

        // Assemble H matrix row by row (collocation points)
        for i in 0..n {
            let collocation_point = boundary_nodes[i];
            // Outward normal at the collocation node (area-weighted vertex normal)
            let collocation_normal = vertex_normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]);

            // Loop over boundary elements
            for (elem_idx, &elem) in elements.iter().enumerate() {
                let node1 = boundary_nodes[elem[0]];
                let node2 = boundary_nodes[elem[1]];
                let node3 = boundary_nodes[elem[2]];

                // Integrate over element
                let (h_cbie, h_hbie) = self.element_contribution(
                    &collocation_point,
                    &collocation_normal,
                    node1,
                    node2,
                    node3,
                )?;

                // Add contributions to matrix
                for &global_node_idx in &elements[elem_idx] {
                    // CBIE contribution (coefficient in direct integral equation)
                    h_matrix[[i, global_node_idx]] += h_cbie;

                    // Add HBIE contribution with coupling parameter
                    h_matrix[[i, global_node_idx]] += alpha * h_hbie;
                }
            }

            // Diagonal term: self-interaction on smooth boundary is c(r) = 1/2 for CBIE
            // For Burton-Miller: diagonal contains 1/2 + α·(1/2) from HBIE contribution
            h_matrix[[i, i]] += Complex64::new(0.5, 0.0) + alpha * Complex64::new(0.5, 0.0);
        }

        Ok(h_matrix)
    }

    /// Assemble Burton-Miller system matrix G (RHS coefficients)
    ///
    /// Returns matrix G for the Neumann data (∂u/∂n)
    pub fn assemble_g_matrix(
        &self,
        boundary_nodes: &[[f64; 3]],
        elements: &[[usize; 3]],
        num_collocation_points: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        let n = num_collocation_points;
        let mut g_matrix = Array2::zeros((n, n));

        let _k = self.config.wavenumber;
        let alpha = self.config.coupling_alpha;

        // Assemble G matrix (similar to H but with different kernel)
        for i in 0..n {
            let collocation_point = boundary_nodes[i];

            for (elem_idx, &elem) in elements.iter().enumerate() {
                let node1 = boundary_nodes[elem[0]];
                let node2 = boundary_nodes[elem[1]];
                let node3 = boundary_nodes[elem[2]];

                let (g_cbie, g_hbie) =
                    self.element_contribution_g(&collocation_point, node1, node2, node3)?;

                for &global_node_idx in &elements[elem_idx] {
                    g_matrix[[i, global_node_idx]] += g_cbie + alpha * g_hbie;
                }
            }
        }

        Ok(g_matrix)
    }

    /// Compute element contribution to H matrix
    /// Returns (CBIE_contribution, HBIE_contribution)
    fn element_contribution(
        &self,
        collocation: &[f64; 3],
        collocation_normal: &[f64; 3],
        node1: [f64; 3],
        node2: [f64; 3],
        node3: [f64; 3],
    ) -> KwaversResult<(Complex64, Complex64)> {
        let k = self.config.wavenumber;

        // Numerical integration over triangular element
        // Using 3-point Gaussian quadrature for linear triangles
        let mut h_cbie = Complex64::new(0.0, 0.0);
        let mut h_hbie = Complex64::new(0.0, 0.0);

        // Gauss points for triangle (barycentric coordinates)
        let gauss_points = [(1.0 / 3.0, 1.0 / 3.0), (0.6, 0.2), (0.2, 0.6)];
        let gauss_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        // Source-element outward normal (n_y): constant over flat triangle
        let normal_y = self.triangle_normal(node1, node2, node3);

        for (gp_idx, &(xi, eta)) in gauss_points.iter().enumerate() {
            // Map barycentric to Cartesian
            let zeta = 1.0 - xi - eta;
            let point_on_element = [
                zeta * node1[0] + xi * node2[0] + eta * node3[0],
                zeta * node1[1] + xi * node2[1] + eta * node3[1],
                zeta * node1[2] + xi * node2[2] + eta * node3[2],
            ];

            // Distance
            let r = self.distance(collocation, &point_on_element);
            if r < self.config.singular_regularization {
                continue; // Skip singular points
            }

            // CBIE kernel: ∂G/∂n_y (normal derivative of G at source point)
            let dg_dn = self.greens_function_normal_derivative_full(
                k, r, collocation, &point_on_element, &normal_y,
            );
            h_cbie += gauss_weights[gp_idx] * dg_dn;

            // HBIE kernel: ∂²G/(∂n_x ∂n_y) (hypersingular kernel)
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

        // Account for element Jacobian
        let element_area = self.triangle_area(node1, node2, node3);
        h_cbie *= element_area;
        h_hbie *= element_area;

        Ok((h_cbie, h_hbie))
    }

    /// Compute element contribution to G matrix
    fn element_contribution_g(
        &self,
        collocation: &[f64; 3],
        node1: [f64; 3],
        node2: [f64; 3],
        node3: [f64; 3],
    ) -> KwaversResult<(Complex64, Complex64)> {
        let k = self.config.wavenumber;

        let mut g_cbie = Complex64::new(0.0, 0.0);
        let mut g_hbie = Complex64::new(0.0, 0.0);

        let gauss_points = [(1.0 / 3.0, 1.0 / 3.0), (0.6, 0.2), (0.2, 0.6)];
        let gauss_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        for (gp_idx, &(xi, eta)) in gauss_points.iter().enumerate() {
            let zeta = 1.0 - xi - eta;
            let point_on_element = [
                zeta * node1[0] + xi * node2[0] + eta * node3[0],
                zeta * node1[1] + xi * node2[1] + eta * node3[1],
                zeta * node1[2] + xi * node2[2] + eta * node3[2],
            ];

            let r = self.distance(collocation, &point_on_element);
            if r < self.config.singular_regularization {
                continue;
            }

            let g = self.greens_function_helmholtz(k, r);

            // CBIE G kernel: G
            g_cbie += gauss_weights[gp_idx] * g;

            // HBIE G kernel: ∂G/∂n (normal derivative of Green's function at source)
            let _normal = self.triangle_normal(node1, node2, node3);
            let dg_dn =
                self.greens_function_normal_derivative(k, r, collocation, &point_on_element);
            g_hbie += gauss_weights[gp_idx] * dg_dn;
        }

        let element_area = self.triangle_area(node1, node2, node3);
        g_cbie *= element_area;
        g_hbie *= element_area;

        Ok((g_cbie, g_hbie))
    }

    // ==================== Helper Functions ====================

    /// 3D Euclidean distance
    fn distance(&self, p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
    }

    /// Free space Green's function for Helmholtz equation
    /// G(r, r') = exp(ikr) / (4πr)
    fn greens_function_helmholtz(&self, k: f64, r: f64) -> Complex64 {
        let phase = Complex64::new(0.0, k * r);
        phase.exp() / (4.0 * PI * r)
    }

    /// Normal derivative of Green's function: ∂G/∂n_y (at field/source point y).
    ///
    /// ## Formula
    ///
    /// For G = exp(ikr)/(4πr) with r = |y−x|, r̂ = (y−x)/r:
    /// ```text
    /// ∂G/∂n_y = G · (ik − 1/r) · (r̂ · n_y)
    /// ```
    ///
    /// The 1-component approximation in the legacy `greens_function_normal_derivative` used
    /// only `r̂_x`, which is only correct when n_y is aligned with the x-axis.
    /// This function computes the full 3D dot product.
    ///
    /// ## Reference
    /// - Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic Scattering
    ///   Theory* (3rd ed.). Springer. §3.1, Eq. 3.41.
    fn greens_function_normal_derivative_full(
        &self,
        k: f64,
        r: f64,
        collocation: &[f64; 3],
        point: &[f64; 3],
        normal_y: &[f64; 3],
    ) -> Complex64 {
        if r < 1e-12 {
            return Complex64::new(0.0, 0.0);
        }
        // r̂ = (y − x)/r = (point − collocation)/r
        let rhat = [
            (point[0] - collocation[0]) / r,
            (point[1] - collocation[1]) / r,
            (point[2] - collocation[2]) / r,
        ];
        let cos_ny = rhat[0] * normal_y[0] + rhat[1] * normal_y[1] + rhat[2] * normal_y[2];
        let g = self.greens_function_helmholtz(k, r);
        let alpha = Complex64::new(0.0, k) - Complex64::new(1.0 / r, 0.0);
        g * alpha * cos_ny
    }

    /// Legacy 1-D approximation of ∂G/∂n (preserved for callers that used it directly).
    ///
    /// **Warning**: Only correct when the outward normal at `point` is aligned with the
    /// x-axis (i.e. `n_y = [1, 0, 0]`). For general meshes use
    /// `greens_function_normal_derivative_full`.
    fn greens_function_normal_derivative(
        &self,
        k: f64,
        r: f64,
        collocation: &[f64; 3],
        point: &[f64; 3],
    ) -> Complex64 {
        if r < 1e-12 {
            return Complex64::new(0.0, 0.0);
        }
        let dr_dn = (collocation[0] - point[0]) / r;
        let phase = Complex64::new(0.0, k * r);
        let exp_ikr = phase.exp();
        let dg_dr = (Complex64::new(0.0, k) - Complex64::new(1.0 / r, 0.0)) * exp_ikr
            / (4.0 * PI * r * r);
        dg_dr * dr_dn
    }

    /// Full hypersingular kernel ∂²G/(∂n_x ∂n_y) (Colton & Kress 2013, §3.3).
    ///
    /// ## Derivation (Colton & Kress 2013, Theorem 3.3)
    ///
    /// For G = exp(ikr)/(4πr), r = |y−x|, r̂ = (y−x)/r,
    /// outward normals n_x (at receiver x) and n_y (at source y):
    ///
    /// ```text
    /// ∂G/∂n_y = G · α(r) · (r̂·n_y)    where α(r) = ik − 1/r
    ///
    /// ∂²G/(∂n_x ∂n_y) = G · [
    ///   (k² + 3ik/r − 3/r²) · (r̂·n_x) · (r̂·n_y)
    ///   + (1/r² − ik/r) · (n_x · n_y)
    /// ]
    /// ```
    ///
    /// **Derivation sketch** (chain rule applied to ∂/∂n_x[G·α·cos_ny]):
    /// - ∂G/∂x_i = −G·α·r̂_i  (since ∂r/∂x_i = −r̂_i)
    /// - ∂α/∂x_i = −r̂_i/r²   (since α = ik−1/r, ∂(1/r)/∂r·(−r̂_i) = r̂_i/r²)
    /// - ∂cos_ny/∂x_i = r̂_i·cos_ny/r − n_y_i/r
    /// Assembling and contracting with n_x gives the formula above.
    ///
    /// **Static limit** (k→0): reduces to standard 1/r³ dipole kernel as expected:
    /// `(−3/r²)·cos_nx·cos_ny·G + (1/r²)·nxny·G = G(nxny − 3coscos)/r² = (nxny−3coscos)/(4πr³)`.
    ///
    /// **Near-singularity**: for r < 1e-10 return 0 (caller skips by regularisation threshold).
    /// Note: the hypersingular r → 0 limit requires dedicated regularisation (Guiggiani 1992);
    /// this function is only called for non-singular Gauss points.
    ///
    /// ## References
    /// - Burton, A.J. & Miller, G.F. (1971). Proc. Roy. Soc. A 323(1553), 201–210.
    /// - Colton, D. & Kress, R. (2013). *Inverse Acoustic and Electromagnetic Scattering
    ///   Theory* (3rd ed.). Springer. §3.3, Theorem 3.3.
    /// - Guiggiani, M. (1992). Formulation and numerical treatment of boundary integral
    ///   equations with hypersingular kernels. Comp. Meth. Appl. Mech. Engng. 93(2), 283–301.
    fn greens_function_double_normal_derivative(
        &self,
        k: f64,
        r: f64,
        collocation: &[f64; 3],
        point: &[f64; 3],
        normal_y: &[f64; 3],
        normal_x: &[f64; 3],
    ) -> Complex64 {
        if r < 1e-10 {
            // Near-singular limit — caller uses regularisation to skip these points
            return Complex64::new(0.0, 0.0);
        }

        // r̂ = (y − x)/r (from receiver x to source y)
        let rhat = [
            (point[0] - collocation[0]) / r,
            (point[1] - collocation[1]) / r,
            (point[2] - collocation[2]) / r,
        ];

        // Geometry scalars
        let cos_nx = rhat[0] * normal_x[0] + rhat[1] * normal_x[1] + rhat[2] * normal_x[2];
        let cos_ny = rhat[0] * normal_y[0] + rhat[1] * normal_y[1] + rhat[2] * normal_y[2];
        let nxny = normal_x[0] * normal_y[0] + normal_x[1] * normal_y[1] + normal_x[2] * normal_y[2];

        let g = self.greens_function_helmholtz(k, r);

        // Coefficient for the (r̂·n_x)(r̂·n_y) term: k² + 3ik/r − 3/r²
        let coeff_cos = Complex64::new(k * k - 3.0 / (r * r), 3.0 * k / r);
        // Coefficient for the (n_x·n_y) term: 1/r² − ik/r
        let coeff_nx = Complex64::new(1.0 / (r * r), -k / r);

        g * (coeff_cos * (cos_nx * cos_ny) + coeff_nx * nxny)
    }

    /// Triangle normal vector (outward)
    fn triangle_normal(&self, p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> [f64; 3] {
        // Edge vectors
        let e1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let e2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];

        // Cross product
        let normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Normalize
        let norm = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
        if norm > 1e-12 {
            [normal[0] / norm, normal[1] / norm, normal[2] / norm]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    /// Triangle area
    fn triangle_area(&self, p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
        let e1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let e2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];

        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_burton_miller_config_creation() {
        let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
        assert!(cfg.wavenumber > 0.0);
        assert!(cfg.coupling_alpha.norm() > 0.0);
    }

    #[test]
    fn test_burton_miller_config_custom_alpha() {
        let cfg = BurtonMillerConfig::new(100_000.0, 1500.0)
            .with_coupling_alpha(Complex64::new(0.0, 1.0));
        assert_eq!(cfg.coupling_alpha, Complex64::new(0.0, 1.0));
    }

    #[test]
    fn test_assembler_creation() {
        let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
        let _assembler = BurtonMillerAssembler::new(cfg);
    }

    #[test]
    fn test_greens_function_helmholtz() {
        let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
        let assembler = BurtonMillerAssembler::new(cfg);

        let g = assembler.greens_function_helmholtz(cfg.wavenumber, 0.1);
        assert!(!g.re.is_nan() && !g.im.is_nan());
        assert!(g.norm() > 0.0);
    }

    #[test]
    fn test_triangle_normal() {
        let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
        let assembler = BurtonMillerAssembler::new(cfg);

        let n1 = [0.0, 0.0, 0.0];
        let n2 = [1.0, 0.0, 0.0];
        let n3 = [0.0, 1.0, 0.0];

        let normal = assembler.triangle_normal(n1, n2, n3);
        assert!((normal[2] - 1.0).abs() < 1e-10); // Should point in +z
    }

    #[test]
    fn test_triangle_area() {
        let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
        let assembler = BurtonMillerAssembler::new(cfg);

        let n1 = [0.0, 0.0, 0.0];
        let n2 = [1.0, 0.0, 0.0];
        let n3 = [0.0, 1.0, 0.0];

        let area = assembler.triangle_area(n1, n2, n3);
        assert!((area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_distance() {
        let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
        let assembler = BurtonMillerAssembler::new(cfg);

        let p1 = [0.0, 0.0, 0.0];
        let p2 = [3.0, 4.0, 0.0];

        let dist = assembler.distance(&p1, &p2);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Tests for greens_function_double_normal_derivative (hypersingular kernel)
    // -----------------------------------------------------------------------

    /// Parallel normals n_x = n_y = ẑ, field point y above collocation x.
    ///
    /// ## Analytical value
    ///
    /// With r̂ = ẑ, n_x = ẑ, n_y = ẑ:
    ///   cos_nx = 1, cos_ny = 1, nxny = 1
    ///
    /// ∂²G/(∂n_x ∂n_y) = G * [(k² + 3ik/r − 3/r²)·1 + (1/r² − ik/r)·1]
    ///                  = G * [k² + 3ik/r − 3/r² + 1/r² − ik/r]
    ///                  = G * [k² + 2ik/r − 2/r²]
    #[test]
    fn test_hypersingular_parallel_normals_matches_formula() {
        let k = 2.0_f64;
        let r = 1.0_f64;
        let cfg = BurtonMillerConfig::new(1000.0, 1500.0);
        let assembler = BurtonMillerAssembler::new(cfg);

        let collocation = [0.0_f64, 0.0, 0.0];
        let point = [0.0_f64, 0.0, r]; // y = x + r·ẑ
        let nx = [0.0_f64, 0.0, 1.0]; // ẑ
        let ny = [0.0_f64, 0.0, 1.0]; // ẑ

        let result = assembler.greens_function_double_normal_derivative(
            k, r, &collocation, &point, &ny, &nx,
        );

        let g = assembler.greens_function_helmholtz(k, r);
        let expected = g * Complex64::new(k * k - 2.0 / (r * r), 2.0 * k / r);

        let diff = (result - expected).norm();
        assert!(
            diff < 1e-12,
            "parallel normals: |result − expected| = {:.3e}",
            diff
        );
    }

    /// Static limit k=0: ∂²G/(∂n_x ∂n_y) = (nxny − 3coscos)/(4πr³).
    ///
    /// Proof: for k=0, G = 1/(4πr) and the formula reduces to
    ///   G * (−3/r²)*cos_nx*cos_ny + G*(1/r²)*nxny = (nxny − 3coscos)/(4πr³).
    #[test]
    fn test_hypersingular_static_limit_matches_dipole_kernel() {
        let k = 0.0_f64; // Static (Laplace) case
        let r = 2.0_f64;
        let cfg = BurtonMillerConfig::new(1.0, 1.0); // dummy
        let assembler = BurtonMillerAssembler::new(cfg);

        let collocation = [0.0_f64, 0.0, 0.0];
        let point = [r, 0.0, 0.0]; // r̂ = x̂
        let nx = [0.0_f64, 0.0, 1.0]; // ẑ (perpendicular to r̂)
        let ny = [0.0_f64, 0.0, 1.0]; // ẑ

        // cos_nx = r̂·n_x = 0 (ẑ⊥x̂), cos_ny = 0, nxny = 1
        // ∂²G/(∂n_x ∂n_y) = G * (1/r² - 0) * 1 = G/r²
        let result = assembler.greens_function_double_normal_derivative(
            k, r, &collocation, &point, &ny, &nx,
        );

        let g_static = 1.0 / (4.0 * std::f64::consts::PI * r); // G for k=0
        let expected = Complex64::new(g_static / (r * r), 0.0);
        let diff = (result - expected).norm();
        assert!(
            diff < 1e-14,
            "static limit nxny=1, cos=0: |result − expected| = {:.3e}",
            diff
        );
    }

    /// Perpendicular normals n_x ⊥ n_y ⊥ r̂: nxny = 0, both cos terms = 0 → result ≈ 0.
    ///
    /// Specifically: r̂ = x̂, n_x = ŷ, n_y = ẑ.
    ///   cos_nx = r̂·ŷ = 0, cos_ny = r̂·ẑ = 0, nxny = ŷ·ẑ = 0 → 0.
    #[test]
    fn test_hypersingular_all_perpendicular_is_zero() {
        let k = 5.0_f64;
        let r = 1.5_f64;
        let cfg = BurtonMillerConfig::new(1000.0, 1500.0);
        let assembler = BurtonMillerAssembler::new(cfg);

        let collocation = [0.0_f64, 0.0, 0.0];
        let point = [r, 0.0, 0.0]; // r̂ = x̂
        let nx = [0.0_f64, 1.0, 0.0]; // ŷ
        let ny = [0.0_f64, 0.0, 1.0]; // ẑ

        let result = assembler.greens_function_double_normal_derivative(
            k, r, &collocation, &point, &ny, &nx,
        );

        assert!(
            result.norm() < 1e-14,
            "all-perpendicular: expected 0, got {:.3e}",
            result.norm()
        );
    }

    /// Near-singularity guard: r < 1e-10 returns exactly zero.
    #[test]
    fn test_hypersingular_near_singular_returns_zero() {
        let cfg = BurtonMillerConfig::new(1000.0, 1500.0);
        let assembler = BurtonMillerAssembler::new(cfg);

        let collocation = [0.0_f64, 0.0, 0.0];
        let point = [1e-11_f64, 0.0, 0.0];
        let r = 1e-11_f64;
        let n = [0.0_f64, 0.0, 1.0];

        let result = assembler.greens_function_double_normal_derivative(
            2.0, r, &collocation, &point, &n, &n,
        );
        assert_eq!(result, Complex64::new(0.0, 0.0));
    }
}
