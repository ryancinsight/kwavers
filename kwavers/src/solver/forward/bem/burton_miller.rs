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
//! ```
//! c(r)u(r) + ∫_Γ [∂G(r,r')/∂n(r')] u(r') dΓ = ∫_Γ G(r,r') t(r') dΓ
//! ```
//!
//! where:
//! - c(r) = 1 for r inside domain, 1/2 for r on boundary, 0 for r outside
//! - G = exp(ik|r-r'|)/(4π|r-r'|) (free space Green's function for Helmholtz)
//! - t = ∂u/∂n (normal derivative)
//!
//! Hypersingular Integral Equation (HBIE):
//! ```
//! d(r)t(r) - ∫_Γ [∂²G(r,r')/∂n(r)∂n(r')] u(r') dΓ = ∫_Γ [∂G(r,r')/∂n(r)] t(r') dΓ
//! ```
//!
//! where:
//! - d(r) = 1/2 for r on smooth boundary
//! - Second term requires careful regularization (strongly singular integral)
//!
//! **Burton-Miller Formulation** (Burton & Miller 1971):
//! ```
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

        // Assemble H matrix row by row (collocation points)
        for i in 0..n {
            let collocation_point = boundary_nodes[i];

            // Loop over boundary elements
            for (elem_idx, &elem) in elements.iter().enumerate() {
                let node1 = boundary_nodes[elem[0]];
                let node2 = boundary_nodes[elem[1]];
                let node3 = boundary_nodes[elem[2]];

                // Integrate over element
                // For each collocation point i, compute contributions from element j
                let (h_cbie, h_hbie) =
                    self.element_contribution(&collocation_point, node1, node2, node3)?;

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

        for (gp_idx, &(xi, eta)) in gauss_points.iter().enumerate() {
            // Map barycentric to Cartesian
            let zeta = 1.0 - xi - eta;
            let point_on_element = [
                zeta * node1[0] + xi * node2[0] + eta * node3[0],
                zeta * node1[1] + xi * node2[1] + eta * node3[1],
                zeta * node1[2] + xi * node2[2] + eta * node3[2],
            ];

            // Distance and Green's function
            let r = self.distance(collocation, &point_on_element);
            if r < self.config.singular_regularization {
                continue; // Skip singular points
            }

            let _g = self.greens_function_helmholtz(k, r);
            let dg_dn =
                self.greens_function_normal_derivative(k, r, collocation, &point_on_element);

            // Element normal (computed from cross product of two edges)
            let normal = self.triangle_normal(node1, node2, node3);

            // CBIE kernel: ∂G/∂n(r')
            h_cbie += gauss_weights[gp_idx] * dg_dn * normal[0];

            // HBIE kernel: ∂²G/(∂n·∂n') - requires second derivatives
            let d2g_dndn = self.greens_function_double_normal_derivative(
                k,
                r,
                collocation,
                &point_on_element,
                &normal,
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

    /// Normal derivative of Green's function
    /// ∂G/∂n = ∂G/∂r · (normal · grad_r)
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

        let dr_dn = (collocation[0] - point[0]) / r; // Unit vector dot normal

        let phase = Complex64::new(0.0, k * r);
        let exp_ikr = phase.exp();

        // ∂G/∂r = (ik - 1/r) * G / r
        let dg_dr =
            (Complex64::new(0.0, k) - Complex64::new(1.0 / r, 0.0)) * exp_ikr / (4.0 * PI * r * r);

        dg_dr * dr_dn
    }

    /// Double normal derivative (strongly singular) - simplified approximation
    fn greens_function_double_normal_derivative(
        &self,
        k: f64,
        r: f64,
        _collocation: &[f64; 3],
        _point: &[f64; 3],
        _normal: &[f64; 3],
    ) -> Complex64 {
        if r < 1e-10 {
            return Complex64::new(0.0, 0.0);
        }

        // Simplified: ∂²G/(∂n·∂n') ≈ -k² * G / r
        // Full computation requires careful handling of singular behavior
        let g = self.greens_function_helmholtz(k, r);
        -Complex64::new(k * k, 0.0) * g / r
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
}
