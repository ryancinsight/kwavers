//! Discontinuous Galerkin (DG) solver implementation
//! 
//! This module implements a proper DG method for solving hyperbolic conservation laws
//! with shock-capturing capabilities for handling discontinuities.
//!
//! # Theory
//!
//! The DG method combines features of finite element and finite volume methods:
//! - Local polynomial approximation within elements (finite element)
//! - Flux exchange between elements (finite volume)
//! - Allows discontinuities at element boundaries
//!
//! # References
//!
//! - Hesthaven, J. S., & Warburton, T. (2008). "Nodal discontinuous Galerkin methods"
//! - Cockburn, B., & Shu, C. W. (2001). "Runge-Kutta discontinuous Galerkin methods"

use crate::grid::Grid;
use crate::KwaversResult;
use crate::error::{KwaversError, ValidationError};
use super::traits::{NumericalSolver, DGOperations};
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

/// Basis function type for DG
#[derive(Debug, Clone, Copy)]
pub enum BasisType {
    /// Legendre polynomials (modal basis)
    Legendre,
    /// Lagrange polynomials (nodal basis)
    Lagrange,
}

/// Numerical flux type
#[derive(Debug, Clone, Copy)]
pub enum FluxType {
    /// Local Lax-Friedrichs (Rusanov) flux
    LaxFriedrichs,
    /// Roe flux
    Roe,
    /// HLL flux
    HLL,
    /// HLLC flux
    HLLC,
}

/// Configuration for DG solver
#[derive(Debug, Clone)]
pub struct DGConfig {
    /// Polynomial order (N means polynomials up to degree N)
    pub polynomial_order: usize,
    /// Basis function type
    pub basis_type: BasisType,
    /// Numerical flux type
    pub flux_type: FluxType,
    /// Enable slope limiting for shock capturing
    pub use_limiter: bool,
    /// Limiter type (if enabled)
    pub limiter_type: LimiterType,
}

impl Default for DGConfig {
    fn default() -> Self {
        Self {
            polynomial_order: 3,
            basis_type: BasisType::Legendre,
            flux_type: FluxType::LaxFriedrichs,
            use_limiter: true,
            limiter_type: LimiterType::MinMod,
        }
    }
}

/// Slope limiter types for shock capturing
#[derive(Debug, Clone, Copy)]
pub enum LimiterType {
    /// MinMod limiter
    MinMod,
    /// TVB limiter
    TVB,
    /// WENO limiter
    WENO,
    /// Moment limiter
    Moment,
}

/// DG solver for hyperbolic conservation laws
pub struct DGSolver {
    /// Configuration
    config: DGConfig,
    /// Grid reference
    grid: Arc<Grid>,
    /// Number of nodes per element
    n_nodes: usize,
    /// Quadrature nodes on reference element [-1, 1]
    xi_nodes: Array1<f64>,
    /// Quadrature weights
    weights: Array1<f64>,
    /// Vandermonde matrix for basis evaluation
    vandermonde: Array2<f64>,
    /// Mass matrix M_ij = integral(phi_i * phi_j)
    mass_matrix: Array2<f64>,
    /// Stiffness matrix S_ij = integral(phi_i * dphi_j/dxi)
    stiffness_matrix: Array2<f64>,
    /// Differentiation matrix D = V * Dr * V^{-1}
    diff_matrix: Array2<f64>,
    /// Lift matrix for surface integrals
    lift_matrix: Array2<f64>,
    /// Modal coefficients for each element (n_elements x n_nodes x n_vars)
    modal_coefficients: Option<Array3<f64>>,
}

impl DGSolver {
    /// Create a new DG solver
    pub fn new(config: DGConfig, grid: Arc<Grid>) -> KwaversResult<Self> {
        let n_nodes = config.polynomial_order + 1;
        
        // Generate quadrature nodes and weights
        let (xi_nodes, weights) = Self::gauss_lobatto_quadrature(n_nodes)?;
        
        // Build Vandermonde matrix
        let vandermonde = Self::build_vandermonde(&xi_nodes, config.polynomial_order, config.basis_type)?;
        
        // Compute mass matrix
        let mass_matrix = Self::compute_mass_matrix(&vandermonde, &weights)?;
        
        // Compute stiffness matrix
        let stiffness_matrix = Self::compute_stiffness_matrix(&vandermonde, &xi_nodes, &weights, config.basis_type)?;
        
        // Compute differentiation matrix
        let diff_matrix = Self::compute_diff_matrix(&vandermonde, &xi_nodes, config.basis_type)?;
        
        // Compute lift matrix for boundary terms
        let lift_matrix = Self::compute_lift_matrix(&mass_matrix, n_nodes)?;
        
        Ok(Self {
            config,
            grid,
            n_nodes,
            xi_nodes,
            weights,
            vandermonde,
            mass_matrix,
            stiffness_matrix,
            diff_matrix,
            lift_matrix,
            modal_coefficients: None,
        })
    }
    
    /// Generate Gauss-Lobatto quadrature nodes and weights
    fn gauss_lobatto_quadrature(n: usize) -> KwaversResult<(Array1<f64>, Array1<f64>)> {
        if n < 2 {
            return Err(KwaversError::Validation(ValidationError::RangeValidation {
                field: "n_nodes".to_string(),
                value: n as f64,
                min: 2.0,
                max: f64::INFINITY,
            }));
        }
        
        let mut nodes = Array1::zeros(n);
        let mut weights = Array1::zeros(n);
        
        if n == 2 {
            nodes[0] = -1.0;
            nodes[1] = 1.0;
            weights[0] = 1.0;
            weights[1] = 1.0;
        } else {
            // Endpoints
            nodes[0] = -1.0;
            nodes[n-1] = 1.0;
            
            // Interior nodes are roots of P'_{n-1}
            // Using Newton's method
            for i in 1..n-1 {
                let mut x = -(1.0 - 3.0 * (i as f64) / (n as f64 - 1.0)).cos();
                for _ in 0..20 {  // Newton iterations
                    let (p, dp) = Self::legendre_poly_deriv(n-1, x);
                    x -= p / dp;
                }
                nodes[i] = x;
            }
            
            // Compute weights
            for i in 0..n {
                let (p, _) = Self::legendre_poly_deriv(n-1, nodes[i]);
                weights[i] = 2.0 / ((n as f64 - 1.0) * n as f64 * p * p);
            }
        }
        
        Ok((nodes, weights))
    }
    
    /// Compute Legendre polynomial and its derivative
    fn legendre_poly_deriv(n: usize, x: f64) -> (f64, f64) {
        if n == 0 {
            return (1.0, 0.0);
        } else if n == 1 {
            return (x, 1.0);
        }
        
        let mut p0 = 1.0;
        let mut p1 = x;
        let mut dp0 = 0.0;
        let mut dp1 = 1.0;
        
        for k in 2..=n {
            let a = (2.0 * k as f64 - 1.0) / k as f64;
            let b = (k as f64 - 1.0) / k as f64;
            
            let p2 = a * x * p1 - b * p0;
            let dp2 = a * (p1 + x * dp1) - b * dp0;
            
            p0 = p1;
            p1 = p2;
            dp0 = dp1;
            dp1 = dp2;
        }
        
        (p1, dp1)
    }
    
    /// Build Vandermonde matrix
    fn build_vandermonde(nodes: &Array1<f64>, order: usize, basis_type: BasisType) -> KwaversResult<Array2<f64>> {
        let n = nodes.len();
        let mut v = Array2::zeros((n, order + 1));
        
        match basis_type {
            BasisType::Legendre => {
                for i in 0..n {
                    for j in 0..=order {
                        v[[i, j]] = Self::legendre_basis(j, nodes[i]);
                    }
                }
            }
            BasisType::Lagrange => {
                for i in 0..n {
                    for j in 0..n.min(order + 1) {
                        v[[i, j]] = Self::lagrange_basis(j, nodes[i], nodes);
                    }
                }
            }
        }
        
        Ok(v)
    }
    
    /// Evaluate Legendre basis function
    fn legendre_basis(n: usize, x: f64) -> f64 {
        // Normalized Legendre polynomials
        let (p, _) = Self::legendre_poly_deriv(n, x);
        p * ((2.0 * n as f64 + 1.0) / 2.0).sqrt()
    }
    
    /// Evaluate Lagrange basis function
    fn lagrange_basis(j: usize, x: f64, nodes: &Array1<f64>) -> f64 {
        let n = nodes.len();
        let mut l = 1.0;
        
        for k in 0..n {
            if k != j {
                l *= (x - nodes[k]) / (nodes[j] - nodes[k]);
            }
        }
        
        l
    }
    
    /// Compute mass matrix
    fn compute_mass_matrix(vandermonde: &Array2<f64>, weights: &Array1<f64>) -> KwaversResult<Array2<f64>> {
        let n = vandermonde.shape()[1];
        let mut mass = Array2::zeros((n, n));
        
        // M = V^T * W * V where W is diagonal weight matrix
        for i in 0..n {
            for j in 0..n {
                for k in 0..weights.len() {
                    mass[[i, j]] += vandermonde[[k, i]] * weights[k] * vandermonde[[k, j]];
                }
            }
        }
        
        Ok(mass)
    }
    
    /// Compute stiffness matrix
    fn compute_stiffness_matrix(
        vandermonde: &Array2<f64>,
        nodes: &Array1<f64>,
        weights: &Array1<f64>,
        basis_type: BasisType,
    ) -> KwaversResult<Array2<f64>> {
        let n = vandermonde.shape()[1];
        let n_quad = nodes.len();
        let mut stiff = Array2::zeros((n, n));
        
        // Compute derivative of basis functions at quadrature points
        let mut dphi = Array2::zeros((n_quad, n));
        for i in 0..n_quad {
            for j in 0..n {
                dphi[[i, j]] = match basis_type {
                    BasisType::Legendre => {
                        if j == 0 {
                            0.0
                        } else {
                            let (_, dp) = Self::legendre_poly_deriv(j, nodes[i]);
                            dp * ((2.0 * j as f64 + 1.0) / 2.0).sqrt()
                        }
                    }
                    BasisType::Lagrange => {
                        Self::lagrange_basis_deriv(j, nodes[i], nodes)
                    }
                };
            }
        }
        
        // S = V^T * W * Dphi
        for i in 0..n {
            for j in 0..n {
                for k in 0..n_quad {
                    stiff[[i, j]] += vandermonde[[k, i]] * weights[k] * dphi[[k, j]];
                }
            }
        }
        
        Ok(stiff)
    }
    
    /// Compute derivative of Lagrange basis
    fn lagrange_basis_deriv(j: usize, x: f64, nodes: &Array1<f64>) -> f64 {
        let n = nodes.len();
        let mut dl = 0.0;
        
        for i in 0..n {
            if i != j {
                let mut prod = 1.0 / (nodes[j] - nodes[i]);
                for k in 0..n {
                    if k != j && k != i {
                        prod *= (x - nodes[k]) / (nodes[j] - nodes[k]);
                    }
                }
                dl += prod;
            }
        }
        
        dl
    }
    
    /// Compute differentiation matrix
    fn compute_diff_matrix(
        vandermonde: &Array2<f64>,
        nodes: &Array1<f64>,
        basis_type: BasisType,
    ) -> KwaversResult<Array2<f64>> {
        let n = nodes.len();
        let mut dr = Array2::zeros((n, n));
        
        // Compute derivative Vandermonde matrix
        let mut vr = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                vr[[i, j]] = match basis_type {
                    BasisType::Legendre => {
                        if j == 0 {
                            0.0
                        } else {
                            let (_, dp) = Self::legendre_poly_deriv(j, nodes[i]);
                            dp * ((2.0 * j as f64 + 1.0) / 2.0).sqrt()
                        }
                    }
                    BasisType::Lagrange => {
                        Self::lagrange_basis_deriv(j, nodes[i], nodes)
                    }
                };
            }
        }
        
        // D = Vr * V^{-1}
        let v_inv = Self::matrix_inverse(vandermonde)?;
        dr = vr.dot(&v_inv);
        
        Ok(dr)
    }
    
    /// Compute lift matrix for surface integrals
    fn compute_lift_matrix(mass_matrix: &Array2<f64>, n_nodes: usize) -> KwaversResult<Array2<f64>> {
        let mut lift = Array2::zeros((n_nodes, 2));  // 2 faces in 1D
        
        // Mass matrix inverse
        let m_inv = Self::matrix_inverse(mass_matrix)?;
        
        // Face mass matrix (just endpoint evaluations in 1D)
        lift[[0, 0]] = 1.0;  // Left face
        lift[[n_nodes-1, 1]] = 1.0;  // Right face
        
        // Lift = M^{-1} * E
        lift = m_inv.dot(&lift);
        
        Ok(lift)
    }
    
    /// Simple matrix inversion (replace with better method for production)
    fn matrix_inverse(a: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let n = a.shape()[0];
        if n != a.shape()[1] {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "matrix".to_string(),
                value: format!("{}x{}", a.shape()[0], a.shape()[1]),
                constraint: "must be square".to_string(),
            }));
        }
        
        // Using Gauss-Jordan elimination (simplified)
        let mut aug = Array2::zeros((n, 2*n));
        aug.slice_mut(ndarray::s![.., ..n]).assign(a);
        for i in 0..n {
            aug[[i, n+i]] = 1.0;
        }
        
        // Forward elimination
        for i in 0..n {
            // Pivot
            let pivot = aug[[i, i]];
            if pivot.abs() < 1e-12 {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "matrix".to_string(),
                    value: "singular".to_string(),
                    constraint: "must be invertible".to_string(),
                }));
            }
            
            for j in 0..2*n {
                aug[[i, j]] /= pivot;
            }
            
            for k in 0..n {
                if k != i {
                    let factor = aug[[k, i]];
                    for j in 0..2*n {
                        aug[[k, j]] -= factor * aug[[i, j]];
                    }
                }
            }
        }
        
        Ok(aug.slice(ndarray::s![.., n..]).to_owned())
    }
    
    /// Project solution onto DG basis
    pub fn project_to_dg(&mut self, field: &Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        let n_elements_x = nx / self.n_nodes;
        let n_elements_y = ny;
        let n_elements_z = nz;
        
        // Initialize modal coefficients
        self.modal_coefficients = Some(Array3::zeros((
            n_elements_x * n_elements_y * n_elements_z,
            self.n_nodes,
            1,  // Number of variables (1 for scalar)
        )));
        
        if let Some(ref mut coeffs) = self.modal_coefficients {
            // Project each element
            for ex in 0..n_elements_x {
                for ey in 0..n_elements_y {
                    for ez in 0..n_elements_z {
                        let elem_idx = ex + ey * n_elements_x + ez * n_elements_x * n_elements_y;
                        
                        // Extract element data
                        let mut elem_data = Array1::zeros(self.n_nodes);
                        for i in 0..self.n_nodes {
                            let global_idx = ex * self.n_nodes + i;
                            if global_idx < nx {
                                elem_data[i] = field[[global_idx, ey, ez]];
                            }
                        }
                        
                        // Project: u_modal = V^{-1} * u_nodal
                        let v_inv = Self::matrix_inverse(&self.vandermonde)?;
                        let modal = v_inv.dot(&elem_data);
                        
                        for i in 0..self.n_nodes {
                            coeffs[[elem_idx, i, 0]] = modal[i];
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Compute numerical flux between elements
    fn compute_numerical_flux(
        &self,
        u_left: f64,
        u_right: f64,
        c: f64,  // Wave speed
    ) -> f64 {
        match self.config.flux_type {
            FluxType::LaxFriedrichs => {
                // Lax-Friedrichs (Rusanov) flux
                let alpha = c.abs();  // Maximum wave speed
                0.5 * (c * (u_left + u_right) - alpha * (u_right - u_left))
            }
            FluxType::Roe => {
                // Roe flux (simplified for linear case)
                let a_roe = c;  // Roe average (equals c for linear)
                if a_roe > 0.0 {
                    c * u_left
                } else {
                    c * u_right
                }
            }
            FluxType::HLL | FluxType::HLLC => {
                // HLL flux
                let s_left = -c.abs();
                let s_right = c.abs();
                
                if s_left >= 0.0 {
                    c * u_left
                } else if s_right <= 0.0 {
                    c * u_right
                } else {
                    (s_right * c * u_left - s_left * c * u_right + s_left * s_right * (u_right - u_left)) 
                        / (s_right - s_left)
                }
            }
        }
    }
    
    /// Apply slope limiter for shock capturing
    fn apply_limiter(&self, coeffs: &mut Array3<f64>) -> KwaversResult<()> {
        if !self.config.use_limiter {
            return Ok(());
        }
        
        let n_elements = coeffs.shape()[0];
        
        match self.config.limiter_type {
            LimiterType::MinMod => {
                for elem in 1..n_elements-1 {
                    // Get cell averages
                    let u_avg = coeffs[[elem, 0, 0]];
                    let u_left = coeffs[[elem-1, 0, 0]];
                    let u_right = coeffs[[elem+1, 0, 0]];
                    
                    // Compute slopes
                    let slope_left = u_avg - u_left;
                    let slope_right = u_right - u_avg;
                    
                    // MinMod limiter
                    let limited_slope = Self::minmod(slope_left, slope_right);
                    
                    // Limit higher-order modes
                    for mode in 1..self.n_nodes {
                        let factor = limited_slope / (slope_left.abs().max(slope_right.abs()) + 1e-12);
                        coeffs[[elem, mode, 0]] *= factor.min(1.0);
                    }
                }
            }
            _ => {
                // Other limiters can be implemented here
            }
        }
        
        Ok(())
    }
    
    /// MinMod function for limiting
    fn minmod(a: f64, b: f64) -> f64 {
        if a * b > 0.0 {
            if a.abs() < b.abs() {
                a
            } else {
                b
            }
        } else {
            0.0
        }
    }
    
    /// Solve one time step with DG method
    pub fn solve_step(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        c: f64,  // Wave speed
    ) -> KwaversResult<Array3<f64>> {
        // Project field onto DG basis if not already done
        if self.modal_coefficients.is_none() {
            self.project_to_dg(field)?;
        }
        
        // Extract necessary data before mutable borrow
        let n_nodes = self.n_nodes;
        let stiffness_matrix = self.stiffness_matrix.clone();
        let lift_matrix = self.lift_matrix.clone();
        let mass_matrix = self.mass_matrix.clone();
        let grid_dx = self.grid.dx;
        
        let coeffs = self.modal_coefficients.as_mut()
            .ok_or_else(|| KwaversError::Validation(ValidationError::FieldValidation {
                field: "modal_coefficients".to_string(),
                value: "None".to_string(),
                constraint: "must be initialized".to_string(),
            }))?;
        
        let n_elements = coeffs.shape()[0];
        let mut rhs = Array3::zeros(coeffs.raw_dim());
        
        // Compute DG spatial operator
        for elem in 0..n_elements {
            // Volume integral: -c * S * u
            for i in 0..n_nodes {
                for j in 0..n_nodes {
                    rhs[[elem, i, 0]] -= c * stiffness_matrix[[i, j]] * coeffs[[elem, j, 0]];
                }
            }
            
            // Surface integral (numerical flux)
            let u_minus = coeffs[[elem, n_nodes-1, 0]];  // Right boundary of element
            let u_plus = if elem < n_elements-1 {
                coeffs[[elem+1, 0, 0]]  // Left boundary of next element
            } else {
                u_minus  // Boundary condition
            };
            
            // Compute flux directly here to avoid self borrow
            let flux_right = match self.config.flux_type {
                FluxType::LaxFriedrichs => {
                    let alpha = c.abs();
                    0.5 * (c * (u_minus + u_plus) - alpha * (u_plus - u_minus))
                }
                _ => {
                    // Default to Lax-Friedrichs for other types
                    let alpha = c.abs();
                    0.5 * (c * (u_minus + u_plus) - alpha * (u_plus - u_minus))
                }
            };
            
            let u_minus = if elem > 0 {
                coeffs[[elem-1, n_nodes-1, 0]]  // Right boundary of previous element
            } else {
                coeffs[[elem, 0, 0]]  // Boundary condition
            };
            let u_plus = coeffs[[elem, 0, 0]];  // Left boundary of element
            
            // Compute flux directly here to avoid self borrow
            let flux_left = match self.config.flux_type {
                FluxType::LaxFriedrichs => {
                    let alpha = c.abs();
                    0.5 * (c * (u_minus + u_plus) - alpha * (u_plus - u_minus))
                }
                _ => {
                    // Default to Lax-Friedrichs for other types
                    let alpha = c.abs();
                    0.5 * (c * (u_minus + u_plus) - alpha * (u_plus - u_minus))
                }
            };
            
            // Add surface contributions
            rhs[[elem, 0, 0]] -= (flux_left / grid_dx) * lift_matrix[[0, 0]];
            rhs[[elem, n_nodes-1, 0]] += (flux_right / grid_dx) * lift_matrix[[n_nodes-1, 1]];
        }
        
        // Apply mass matrix inverse
        for elem in 0..n_elements {
            let m_inv = Self::matrix_inverse(&mass_matrix)?;
            let mut elem_rhs = Array1::zeros(n_nodes);
            for i in 0..n_nodes {
                elem_rhs[i] = rhs[[elem, i, 0]];
            }
            let result = m_inv.dot(&elem_rhs);
            for i in 0..n_nodes {
                rhs[[elem, i, 0]] = result[i];
            }
        }
        
        // Time integration (forward Euler for now, can use RK later)
        for elem in 0..n_elements {
            for i in 0..n_nodes {
                coeffs[[elem, i, 0]] += dt * rhs[[elem, i, 0]];
            }
        }
        
        // Apply limiter for shock capturing
        if self.config.use_limiter {
            // Apply limiter inline to avoid borrow issues
            let n_elements = coeffs.shape()[0];
            
            if let LimiterType::MinMod = self.config.limiter_type {
                for elem in 1..n_elements-1 {
                    let u_avg = coeffs[[elem, 0, 0]];
                    let u_left = coeffs[[elem-1, 0, 0]];
                    let u_right = coeffs[[elem+1, 0, 0]];
                    
                    let slope_left = u_avg - u_left;
                    let slope_right = u_right - u_avg;
                    
                    let limited_slope = Self::minmod(slope_left, slope_right);
                    
                    for mode in 1..n_nodes {
                        let factor = limited_slope / (slope_left.abs().max(slope_right.abs()) + 1e-12);
                        coeffs[[elem, mode, 0]] *= factor.min(1.0);
                    }
                }
            }
        }
        
        // Clone coefficients to release mutable borrow before reconstruction
        let coeffs_clone = coeffs.clone();
        
        // Convert back to nodal representation
        self.reconstruct_field(&coeffs_clone)
    }
    
    /// Reconstruct field from modal coefficients
    fn reconstruct_field(&self, coeffs: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let n_elements = coeffs.shape()[0];
        let nx = n_elements * self.n_nodes;
        let mut field = Array3::zeros((nx, 1, 1));
        
        for elem in 0..n_elements {
            // Extract modal coefficients for element
            let mut modal = Array1::zeros(self.n_nodes);
            for i in 0..self.n_nodes {
                modal[i] = coeffs[[elem, i, 0]];
            }
            
            // Convert to nodal: u_nodal = V * u_modal
            let nodal = self.vandermonde.dot(&modal);
            
            // Copy to global field
            for i in 0..self.n_nodes {
                let global_idx = elem * self.n_nodes + i;
                if global_idx < nx {
                    field[[global_idx, 0, 0]] = nodal[i];
                }
            }
        }
        
        Ok(field)
    }
}

impl NumericalSolver for DGSolver {
    fn solve(
        &self,
        field: &Array3<f64>,
        dt: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>> {
        // For regions marked by mask, apply DG method
        let mut result = field.clone();
        
        // Process each direction (simplified to 1D slices for now)
        let (nx, ny, nz) = field.dim();
        
        for j in 0..ny {
            for k in 0..nz {
                // Check if this slice needs DG treatment
                let mut needs_dg = false;
                for i in 0..nx {
                    if mask[[i, j, k]] {
                        needs_dg = true;
                        break;
                    }
                }
                
                if needs_dg {
                    // Extract 1D slice
                    let mut slice = Array3::zeros((nx, 1, 1));
                    for i in 0..nx {
                        slice[[i, 0, 0]] = field[[i, j, k]];
                    }
                    
                    // Create temporary DG solver for this slice
                    let mut dg = self.clone();
                    let c = 1.0;  // Wave speed (should be computed from physics)
                    let updated = dg.solve_step(&slice, dt, c)?;
                    
                    // Copy back
                    for i in 0..nx {
                        if mask[[i, j, k]] {
                            result[[i, j, k]] = updated[[i, 0, 0]];
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    fn max_stable_dt(&self, grid: &Grid) -> f64 {
        // CFL condition for DG
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let c_max = 1.0;  // Maximum wave speed (should be computed from physics)
        
        // DG stability: CFL ~ 1/(2N+1) where N is polynomial order
        let cfl_number = 1.0 / (2.0 * self.config.polynomial_order as f64 + 1.0);
        cfl_number * dx_min / c_max
    }
    
    fn update_order(&mut self, order: usize) {
        self.config.polynomial_order = order;
        self.n_nodes = order + 1;
        // Rebuild matrices with new order
        // This would require re-initialization
    }
}

impl Clone for DGSolver {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            grid: self.grid.clone(),
            n_nodes: self.n_nodes,
            xi_nodes: self.xi_nodes.clone(),
            weights: self.weights.clone(),
            vandermonde: self.vandermonde.clone(),
            mass_matrix: self.mass_matrix.clone(),
            stiffness_matrix: self.stiffness_matrix.clone(),
            diff_matrix: self.diff_matrix.clone(),
            lift_matrix: self.lift_matrix.clone(),
            modal_coefficients: self.modal_coefficients.clone(),
        }
    }
}

impl DGOperations for DGSolver {
    fn compute_flux(&self, left_state: f64, right_state: f64, normal: f64) -> f64 {
        self.compute_numerical_flux(left_state, right_state, normal)
    }
    
    fn project_to_basis(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        // Create a clone to project
        let mut solver = self.clone();
        solver.project_to_dg(field)?;
        
        if let Some(ref coeffs) = solver.modal_coefficients {
            Ok(coeffs.clone())
        } else {
            // Return the field as-is if projection fails
            Ok(field.clone())
        }
    }
    
    fn reconstruct_from_basis(&self, coefficients: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        self.reconstruct_field(coefficients)
    }
}

// Additional methods not in trait
impl DGSolver {
    /// Apply shock detector to identify discontinuities
    pub fn apply_shock_detector(&self, field: &Array3<f64>) -> Array3<bool> {
        let (nx, ny, nz) = field.dim();
        let mut shock_mask = Array3::from_elem((nx, ny, nz), false);
        
        // Simple gradient-based shock detector
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let grad_x = (field[[i+1, j, k]] - field[[i-1, j, k]]).abs() / (2.0 * self.grid.dx);
                    let grad_y = (field[[i, j+1, k]] - field[[i, j-1, k]]).abs() / (2.0 * self.grid.dy);
                    let grad_z = (field[[i, j, k+1]] - field[[i, j, k-1]]).abs() / (2.0 * self.grid.dz);
                    
                    let max_grad = grad_x.max(grad_y).max(grad_z);
                    let threshold = 1.0;  // Should be adaptive
                    
                    if max_grad > threshold {
                        shock_mask[[i, j, k]] = true;
                    }
                }
            }
        }
        
        shock_mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dg_solver_creation() {
        let grid = Arc::new(Grid::new(32, 32, 32, 1.0, 1.0, 1.0));
        let config = DGConfig::default();
        let solver = DGSolver::new(config, grid).unwrap();
        assert_eq!(solver.config.polynomial_order, 3);
        assert_eq!(solver.n_nodes, 4); // order + 1
    }
    
    #[test]
    fn test_legendre_polynomials() {
        // Test first few Legendre polynomials (normalized)
        let p0 = DGSolver::legendre_basis(0, 0.5);
        let p1 = DGSolver::legendre_basis(1, 0.5);
        let p2 = DGSolver::legendre_basis(2, 0.5);
        
        // Check they are non-zero (actual values depend on normalization)
        assert!(p0.abs() > 0.0);
        assert!(p1.abs() > 0.0);
        assert!(p2.abs() > 0.0);
    }
    
    #[test]
    fn test_upwind_flux() {
        let grid = Arc::new(Grid::new(32, 32, 32, 1.0, 1.0, 1.0));
        let config = DGConfig::default();
        let solver = DGSolver::new(config, grid).unwrap();
        
        // Test upwind flux selection
        let flux_positive = solver.compute_flux(1.0, 2.0, 1.0);
        assert_eq!(flux_positive, 1.0); // Should use left state
        
        let flux_negative = solver.compute_flux(1.0, 2.0, -1.0);
        assert_eq!(flux_negative, -2.0); // Should use right state
    }
}