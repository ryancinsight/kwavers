//! BEM Solver Implementation
//!
//! **STATUS: STUB / INCOMPLETE**
//!
//! This is a placeholder BEM implementation with simplified matrices.
//! Actual boundary integral assembly is not yet implemented.
//! TODO_AUDIT: P1 - Complete BEM Solver Implementation - Implement full boundary element method with boundary integral assembly and Green's function evaluation
//! DEPENDS ON: math/green_function.rs, domain/boundary/bem_boundary.rs, math/numerics/integration/surface_quadrature.rs
//! MISSING: Boundary integral assembly for H and G matrices
//! MISSING: Green's function evaluation for Helmholtz equation
//! MISSING: Surface quadrature rules for curved elements
//! MISSING: Efficient matrix-vector products for large systems
//! MISSING: Fast multipole method acceleration
//! SEVERITY: HIGH (essential for radiation and scattering problems)
//! THEOREM: Boundary integral equation: c(r)u(r) + ∫_Γ ∂G/∂n u dΓ = ∫_Γ G ∂u/∂n dΓ for Helmholtz equation
//! THEOREM: Green's function: G(r,r') = exp(ik|r-r'|)/(4π|r-r'|) for 3D free space Helmholtz
//! REFERENCES: Wu (2000) Preconditioned GMRES for BEM; Colton & Kress (1998) Inverse Acoustic Problems
//!
//! Core implementation of the Boundary Element Method for acoustic problems.
//! This solver handles the boundary integral formulation and integrates
//! with the domain boundary condition system.

use crate::core::error::KwaversResult;
use crate::domain::boundary::BemBoundaryManager;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use crate::math::linear_algebra::sparse::solver::{IterativeSolver, Preconditioner, SolverConfig};
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Configuration for BEM solver
#[derive(Debug, Clone)]
pub struct BemConfig {
    /// Wavenumber for Helmholtz equation
    pub wavenumber: f64,
    /// Tolerance for iterative solvers
    pub tolerance: f64,
    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,
    /// Use direct solver (dense matrix) instead of iterative
    pub use_direct_solver: bool,
}

impl Default for BemConfig {
    fn default() -> Self {
        Self {
            wavenumber: 1.0,
            tolerance: 1e-8,
            max_iterations: 1000,
            use_direct_solver: false,
        }
    }
}

/// BEM solver for acoustic boundary element problems
#[derive(Debug)]
pub struct BemSolver {
    /// Solver configuration
    #[allow(dead_code)]
    config: BemConfig,
    /// Boundary mesh nodes (compact indexing)
    nodes: Vec<[f64; 3]>,
    /// Boundary elements (triangles, using compact node indices)
    elements: Vec<[usize; 3]>,
    /// Map from global mesh node index to local BEM node index
    #[allow(dead_code)]
    global_to_local_node: HashMap<usize, usize>,
    /// Boundary condition manager
    boundary_manager: BemBoundaryManager,
    /// BEM system matrices (would be computed from boundary integrals)
    h_matrix: Option<CompressedSparseRowMatrix<Complex64>>,
    g_matrix: Option<CompressedSparseRowMatrix<Complex64>>,
}

impl BemSolver {
    /// Create new BEM solver
    ///
    /// # Arguments
    /// * `config` - Solver configuration
    /// * `mesh` - Tetrahedral mesh from which to extract boundary
    #[must_use]
    pub fn new(config: BemConfig, mesh: &TetrahedralMesh) -> KwaversResult<Self> {
        // Extract boundary faces
        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let mut global_to_local_node = HashMap::new();

        // mesh.boundary_faces maps sorted_face_nodes -> (element_idx, face_idx)
        for (sorted_nodes, &(elem_idx, _face_idx)) in &mesh.boundary_faces {
            // Retrieve element nodes to determine orientation
            // Note: We need the element to determine which side is "out"
            let element = &mesh.elements[elem_idx];
            let elem_nodes = element.nodes;

            // The sorted_nodes are just the keys, we need the actual face winding
            // that corresponds to the outward normal.
            let mut face_nodes = Vec::new();
            let mut interior_node = 0; // placeholder

            for &n_idx in &elem_nodes {
                if sorted_nodes.contains(&n_idx) {
                    face_nodes.push(n_idx);
                } else {
                    interior_node = n_idx;
                }
            }

            // Ensure we have 3 face nodes
            if face_nodes.len() != 3 {
                continue;
            }

            // Calculate normal of the face (p1-p0) x (p2-p0)
            let p0 = mesh.nodes[face_nodes[0]].coordinates;
            let p1 = mesh.nodes[face_nodes[1]].coordinates;
            let p2 = mesh.nodes[face_nodes[2]].coordinates;
            let p_in = mesh.nodes[interior_node].coordinates;

            let v1 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
            let v2 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];

            // Normal
            let nx = v1[1]*v2[2] - v1[2]*v2[1];
            let ny = v1[2]*v2[0] - v1[0]*v2[2];
            let nz = v1[0]*v2[1] - v1[1]*v2[0];

            // Vector from p0 to interior node
            let v_in = [p_in[0]-p0[0], p_in[1]-p0[1], p_in[2]-p0[2]];

            // Dot product (Normal . v_in) should be negative for outward normal
            let dot = nx*v_in[0] + ny*v_in[1] + nz*v_in[2];

            let final_face_nodes = if dot > 0.0 {
                // Normal points inward. Flip winding: 0, 2, 1
                [face_nodes[0], face_nodes[2], face_nodes[1]]
            } else {
                // Normal points outward. Keep winding: 0, 1, 2
                [face_nodes[0], face_nodes[1], face_nodes[2]]
            };

            // Map to local indices
            let mut local_face = [0; 3];
            for (i, &global_idx) in final_face_nodes.iter().enumerate() {
                if let Some(&local_idx) = global_to_local_node.get(&global_idx) {
                    local_face[i] = local_idx;
                } else {
                    let new_idx = nodes.len();
                    nodes.push(mesh.nodes[global_idx].coordinates);
                    global_to_local_node.insert(global_idx, new_idx);
                    local_face[i] = new_idx;
                }
            }
            elements.push(local_face);
        }

        Ok(Self {
            config,
            nodes,
            elements,
            global_to_local_node,
            boundary_manager: BemBoundaryManager::new(),
            h_matrix: None,
            g_matrix: None,
        })
    }

    /// Get mutable reference to boundary condition manager
    #[must_use]
    pub fn boundary_manager(&mut self) -> &mut BemBoundaryManager {
        &mut self.boundary_manager
    }

    /// Get reference to boundary condition manager
    #[must_use]
    pub fn boundary_manager_ref(&self) -> &BemBoundaryManager {
        &self.boundary_manager
    }

    /// Get local BEM node index for a global mesh node index
    ///
    /// Returns `None` if the global node is not part of the BEM boundary.
    #[must_use]
    pub fn local_index(&self, global_idx: usize) -> Option<usize> {
        self.global_to_local_node.get(&global_idx).copied()
    }

    /// Assemble BEM system matrices
    ///
    /// Computes the boundary integrals to assemble the H and G matrices.
    /// Uses standard Gaussian quadrature for non-singular elements and
    /// Duffy transformation / singularity handling for singular elements.
    pub fn assemble_system(&mut self) -> KwaversResult<()> {
        let n = self.nodes.len();
        if n == 0 {
            return Ok(());
        }

        // Initialize dense matrix structures (flattened row-major)
        let mut h_values = vec![Complex64::new(0.0, 0.0); n * n];
        let mut g_values = vec![Complex64::new(0.0, 0.0); n * n];

        // Helper to index dense array
        let idx = |row: usize, col: usize| row * n + col;

        let k = self.config.wavenumber;

        // Loop over collocation points (source nodes) i
        for i in 0..n {
            let r_i = self.nodes[i];

            // Loop over boundary elements
            for element in &self.elements {
                let node_indices = element; // [n1, n2, n3] local indices
                let p1 = self.nodes[node_indices[0]];
                let p2 = self.nodes[node_indices[1]];
                let p3 = self.nodes[node_indices[2]];

                // Check for singularity (source node is one of element nodes)
                let singular_idx = node_indices.iter().position(|&idx| idx == i);

                let (h_contrib, g_contrib) = if let Some(vertex_idx) = singular_idx {
                    compute_singular_integrals(k, r_i, [p1, p2, p3], vertex_idx)
                } else {
                    compute_nonsingular_integrals(k, r_i, [p1, p2, p3])
                };

                // Add contributions to matrices
                for m in 0..3 {
                    let col = node_indices[m];
                    h_values[idx(i, col)] += h_contrib[m];
                    g_values[idx(i, col)] += g_contrib[m];
                }
            }

            // Add diagonal term c(r) * I to H matrix
            // For smooth boundary, c(r) = 0.5
            h_values[idx(i, i)] += Complex64::new(0.5, 0.0);
        }

        // Construct CompressedSparseRowMatrix from dense data
        // col_indices: 0, 1, ..., N-1 repeated N times
        let mut col_indices = Vec::with_capacity(n * n);
        for _ in 0..n {
            for c in 0..n {
                col_indices.push(c);
            }
        }

        // row_pointers: 0, N, 2N, ..., N*N
        let mut row_pointers = Vec::with_capacity(n + 1);
        for i in 0..=n {
            row_pointers.push(i * n);
        }

        self.h_matrix = Some(CompressedSparseRowMatrix {
            rows: n,
            cols: n,
            values: h_values,
            col_indices: col_indices.clone(),
            row_pointers: row_pointers.clone(),
            nnz: n * n,
        });

        self.g_matrix = Some(CompressedSparseRowMatrix {
            rows: n,
            cols: n,
            values: g_values,
            col_indices,
            row_pointers,
            nnz: n * n,
        });

        Ok(())
    }

    /// Solve the BEM system
    ///
    /// Applies boundary conditions and solves for the unknown boundary values.
    ///
    /// # Arguments
    /// * `wavenumber` - Acoustic wavenumber (2πf/c)
    /// * `source_terms` - Optional source terms on boundary
    pub fn solve(
        &mut self,
        wavenumber: f64,
        source_terms: Option<&Array1<Complex64>>,
    ) -> KwaversResult<BemSolution> {
        // Ensure system is assembled
        if self.h_matrix.is_none() || self.g_matrix.is_none() {
            // Update wavenumber in config if it differs?
            // Ideally config.wavenumber should match solve wavenumber
            self.config.wavenumber = wavenumber;
            self.assemble_system()?;
        }

        let h_matrix = self.h_matrix.as_ref().unwrap();
        let g_matrix = self.g_matrix.as_ref().unwrap();

        // Assemble system using boundary manager (non-destructive)
        let (a_matrix, mut b_vector) = self
            .boundary_manager
            .assemble_bem_system(h_matrix, g_matrix, wavenumber)?;

        // Apply source terms if provided (additive to RHS)
        if let Some(sources) = source_terms {
            b_vector += sources;
        }

        // Solve the BEM system
        let x = self.solve_bem_system(&a_matrix, &b_vector)?;

        // Reconstruct full solution
        let (boundary_pressure, boundary_velocity) =
            self.boundary_manager.reconstruct_solution(&x, wavenumber);

        Ok(BemSolution {
            boundary_pressure,
            boundary_velocity,
            wavenumber,
        })
    }

    /// Solve the assembled BEM system
    fn solve_bem_system(
        &self,
        a_matrix: &CompressedSparseRowMatrix<Complex64>,
        b_vector: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let solver_config = SolverConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner: Preconditioner::None,
            verbose: false,
        };
        let solver = IterativeSolver::create(solver_config);

        solver.bicgstab_complex(a_matrix, b_vector.view(), None)
    }

    /// Compute scattered field at evaluation points
    ///
    /// Once boundary values are known, compute the field anywhere in space
    /// using the BEM representation formula.
    pub fn compute_scattered_field(
        &self,
        evaluation_points: &Array1<[f64; 3]>,
        solution: &BemSolution,
    ) -> KwaversResult<Array1<Complex64>> {
        let k = solution.wavenumber;

        // Use parallel iterator if possible
        // We attempt to get a slice; if not contiguous, we can use to_vec (allocation) or standard iter
        // Given Array1 is usually contiguous, this should be efficient.
        let points_slice = evaluation_points.as_slice().ok_or_else(|| {
            crate::core::error::KwaversError::InvalidInput(
                "Evaluation points array must be contiguous for parallel processing".to_string(),
            )
        })?;

        let results: Vec<Complex64> = points_slice
            .par_iter()
            .map(|&r_eval| {
                let mut total_field = Complex64::new(0.0, 0.0);

                // Loop over all boundary elements
                for element_indices in &self.elements {
                    let n1 = element_indices[0];
                    let n2 = element_indices[1];
                    let n3 = element_indices[2];

                    let p1 = self.nodes[n1];
                    let p2 = self.nodes[n2];
                    let p3 = self.nodes[n3];

                    // Use nonsingular integrals as evaluation points are assumed to be in the domain
                    // TODO: Check for near-field singularities if point is close to element
                    let (h_res, g_res) = compute_nonsingular_integrals(k, r_eval, [p1, p2, p3]);

                    // Accumulate contribution from this element
                    // Representation formula: u(x) = ∫ (G * q - ∂G/∂n * u) dΓ
                    // h_res computes ∫ ∂G/∂n * shape_fn
                    // g_res computes ∫ G * shape_fn

                    let u_vals = [
                        solution.boundary_pressure[n1],
                        solution.boundary_pressure[n2],
                        solution.boundary_pressure[n3],
                    ];

                    let q_vals = [
                        solution.boundary_velocity[n1],
                        solution.boundary_velocity[n2],
                        solution.boundary_velocity[n3],
                    ];

                    for m in 0..3 {
                        total_field += g_res[m] * q_vals[m] - h_res[m] * u_vals[m];
                    }
                }

                total_field
            })
            .collect();

        Ok(Array1::from_vec(results))
    }
}

/// Compute integrals for non-singular element
/// Returns (h_contrib, g_contrib) vectors of size 3 (one per node)
fn compute_nonsingular_integrals(
    k: f64,
    r_i: [f64; 3],
    element_nodes: [[f64; 3]; 3],
) -> ([Complex64; 3], [Complex64; 3]) {
    let p1 = element_nodes[0];
    let p2 = element_nodes[1];
    let p3 = element_nodes[2];

    // Compute element area and normal
    let v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let v2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];

    let nx = v1[1]*v2[2] - v1[2]*v2[1];
    let ny = v1[2]*v2[0] - v1[0]*v2[2];
    let nz = v1[0]*v2[1] - v1[1]*v2[0];
    let cross_norm = (nx*nx + ny*ny + nz*nz).sqrt();
    let area = 0.5 * cross_norm;
    let normal = [nx/cross_norm, ny/cross_norm, nz/cross_norm];

    // 3-point Gaussian quadrature for triangle
    // (u, v, w) barycentric coordinates, weight = 1/3 * area
    // Points: (2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3) (permutations)
    // Or standard: (1/6, 1/6), (2/3, 1/6), (1/6, 2/3). Weights 1/3.
    let q_points = [
        ([1.0/6.0, 1.0/6.0], 1.0/3.0),
        ([2.0/3.0, 1.0/6.0], 1.0/3.0),
        ([1.0/6.0, 2.0/3.0], 1.0/3.0),
    ];

    let mut h_res = [Complex64::new(0.0, 0.0); 3];
    let mut g_res = [Complex64::new(0.0, 0.0); 3];

    for (uv, w) in &q_points {
        let u = uv[0];
        let v = uv[1];
        let shape_fn = [1.0 - u - v, u, v]; // N1, N2, N3

        // Global coordinate r
        let rx = shape_fn[0]*p1[0] + shape_fn[1]*p2[0] + shape_fn[2]*p3[0];
        let ry = shape_fn[0]*p1[1] + shape_fn[1]*p2[1] + shape_fn[2]*p3[1];
        let rz = shape_fn[0]*p1[2] + shape_fn[1]*p2[2] + shape_fn[2]*p3[2];
        let r = [rx, ry, rz];

        // Green's function G(r_i, r) and Gradient
        let (g_val, grad_g) = green_function(k, r_i, r);

        // ∂G/∂n = ∇G . n
        let d_g_dn = grad_g[0]*normal[0] + grad_g[1]*normal[1] + grad_g[2]*normal[2];

        // Accumulate
        let weight = w * area;
        for m in 0..3 {
            h_res[m] += d_g_dn * shape_fn[m] * weight;
            g_res[m] += g_val * shape_fn[m] * weight;
        }
    }

    (h_res, g_res)
}

/// Compute integrals for singular element (r_i is one of the nodes)
fn compute_singular_integrals(
    k: f64,
    _r_i: [f64; 3],
    element_nodes: [[f64; 3]; 3],
    vertex_idx: usize, // Index of the singular node in element_nodes (0, 1, or 2)
) -> ([Complex64; 3], [Complex64; 3]) {
    // For flat triangular elements, if the source point is on the element,
    // (r - r_i) is perpendicular to the normal n.
    // Thus (r - r_i) . n = 0, so ∂G/∂n = 0.
    // Therefore, H contribution is zero for all shape functions.
    // (The diagonal term c(r) is added separately in the main loop).
    let h_res = [Complex64::new(0.0, 0.0); 3];

    // For G integrals, we have a weak singularity 1/R.
    // We use Duffy transformation or polar coordinates to integrate.
    // Here we split integrand: G = (e^ikR - 1)/(4πR) + 1/(4πR)
    // 1st part is non-singular. 2nd part is 1/R static singularity.
    // However, simplest robust way is to use Duffy coordinates on the triangle.

    // Reorder nodes so singularity is at p0
    let (p0, p1, p2) = match vertex_idx {
        0 => (element_nodes[0], element_nodes[1], element_nodes[2]),
        1 => (element_nodes[1], element_nodes[2], element_nodes[0]),
        2 => (element_nodes[2], element_nodes[0], element_nodes[1]),
        _ => unreachable!(),
    };

    // Duffy transform: Square (u, v) in [0, 1]^2 -> Triangle
    // r = p0 + u * (p1 - p0) + u * v * (p2 - p1)
    // Jacobian J = 2 * Area * u
    // R = |r - p0| = u * |(p1 - p0) + v * (p2 - p1)|
    // G = e^ikR / (4πR)
    // Integral = ∫∫ G * shape_fn * J du dv
    // J cancels 1/R singularity (R has factor u).
    // G * J = e^ikR / (4π) * (2 Area / |...|)

    let v10 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
    let v21 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]];

    let cross_x = v10[1]*v21[2] - v10[2]*v21[1];
    let cross_y = v10[2]*v21[0] - v10[0]*v21[2];
    let cross_z = v10[0]*v21[1] - v10[1]*v21[0];
    let area = 0.5 * (cross_x*cross_x + cross_y*cross_y + cross_z*cross_z).sqrt();

    // Use a tensor product quadrature on square [0,1]^2.
    // 3x3 Gauss points.
    let gauss_1d = [
        (0.1127016653792583, 0.2777777777777778),
        (0.5000000000000000, 0.4444444444444444),
        (0.8872983346207417, 0.2777777777777778),
    ];

    let mut g_res_reordered = [Complex64::new(0.0, 0.0); 3]; // [sing_node, next, next]

    for (u, wu) in &gauss_1d {
        for (v, wv) in &gauss_1d {
            // Mapping
            let dir_x = v10[0] + v * v21[0];
            let dir_y = v10[1] + v * v21[1];
            let dir_z = v10[2] + v * v21[2];

            let _rx = p0[0] + u * dir_x;
            let _ry = p0[1] + u * dir_y;
            let _rz = p0[2] + u * dir_z;

            let r_dist = u * (dir_x*dir_x + dir_y*dir_y + dir_z*dir_z).sqrt();

            // Jacobian factor for geometry (differential area element in Duffy)
            // dA = 2 * Area * u du dv ?
            // Standard Duffy maps to right triangle, then to actual triangle.
            // Let's verify Jacobian.
            // Map (u,v) -> right triangle (xi, eta): xi = u, eta = u*v ? No.
            // Standard map: xi = u(1-v)? No.
            // Let's use the property that dA = 2*Area * u du dv.
            let jac = 2.0 * area * u;

            let g_val = if r_dist < 1e-12 {
                // Limit u->0. e^ikR -> 1. 4pi R.
                // Term is e^ikR / (4pi R) * 2 Area u.
                // = 1 / (4pi u |dir|) * 2 Area u = 2 Area / (4pi |dir|).
                let dir_norm = (dir_x*dir_x + dir_y*dir_y + dir_z*dir_z).sqrt();
                Complex64::new(2.0 * area / (4.0 * PI * dir_norm), 0.0)
            } else {
                Complex64::new(0.0, k * r_dist).exp() / (4.0 * PI * r_dist) * jac
            };

            // Shape functions in local Duffy/Triangle coords
            // Triangle barycentric (L1, L2, L3). P0 is L1=1. P1 is L2=1. P2 is L3=1.
            // r = L1 P0 + L2 P1 + L3 P2
            // r = (1-u) P0 + u(1-v) P1 + u v P2  <-- This is standard "collapsed" map?
            // Let's check:
            // u=0 => P0. Correct.
            // u=1, v=0 => P1. Correct.
            // u=1, v=1 => P2. Correct.
            // Yes, this is the map I used above: p0 + u(p1-p0 + v(p2-p1)) = p0 + u(p1-p0) + uv(p2-p1).
            // = (1-u)p0 + u(1-v)p1 + uv p2.
            // So shape functions are:
            let l0 = 1.0 - u;          // At P0
            let l1 = u * (1.0 - v);    // At P1
            let l2 = u * v;            // At P2

            let weight = wu * wv;

            g_res_reordered[0] += g_val * l0 * weight;
            g_res_reordered[1] += g_val * l1 * weight;
            g_res_reordered[2] += g_val * l2 * weight;
        }
    }

    // Map back to original node order
    let mut g_res_final = [Complex64::new(0.0, 0.0); 3];
    match vertex_idx {
        0 => { // P0 is node 0
            g_res_final[0] = g_res_reordered[0];
            g_res_final[1] = g_res_reordered[1];
            g_res_final[2] = g_res_reordered[2];
        },
        1 => { // P0 is node 1. P1 is node 2. P2 is node 0.
            g_res_final[1] = g_res_reordered[0];
            g_res_final[2] = g_res_reordered[1];
            g_res_final[0] = g_res_reordered[2];
        },
        2 => { // P0 is node 2. P1 is node 0. P2 is node 1.
            g_res_final[2] = g_res_reordered[0];
            g_res_final[0] = g_res_reordered[1];
            g_res_final[1] = g_res_reordered[2];
        },
        _ => unreachable!(),
    }

    (h_res, g_res_final)
}

/// Green's function G(r, r') = e^(ikR) / (4πR)
/// Returns (G, ∇G)
fn green_function(k: f64, r_src: [f64; 3], r_obs: [f64; 3]) -> (Complex64, [Complex64; 3]) {
    let dx = r_obs[0] - r_src[0];
    let dy = r_obs[1] - r_src[1];
    let dz = r_obs[2] - r_src[2];
    let r_dist = (dx*dx + dy*dy + dz*dz).sqrt();

    if r_dist < 1e-12 {
        // Singularity handling should be done by caller.
        // Return 0 or huge value?
        return (Complex64::new(0.0, 0.0), [Complex64::new(0.0, 0.0); 3]);
    }

    let ik_r = Complex64::new(0.0, k * r_dist);
    let exp_ik_r = ik_r.exp();
    let term = exp_ik_r / (4.0 * PI * r_dist);

    // ∇G = (ik - 1/R) * G * (r - r')/R
    // Note: r is the integration point (r_obs), r_i is source (r_src).
    // Gradient is with respect to integration point r (usually)?
    // Wait. H integral is ∂G/∂n(r). So we need gradient w.r.t r.
    // ∇_r (e^ik|r-r'| / |r-r'|)
    // = e^ikR * (ik - 1/R) / R * ∇R / (4pi)
    // ∇R = (r - r') / R = [dx, dy, dz] / R.

    let factor = (Complex64::new(0.0, k) - 1.0/r_dist) * term / r_dist;
    let grad = [
        factor * dx,
        factor * dy,
        factor * dz,
    ];

    (term, grad)
}

/// Solution of BEM system
#[derive(Debug, Clone)]
pub struct BemSolution {
    /// Pressure on boundary nodes
    pub boundary_pressure: Array1<Complex64>,
    /// Normal velocity on boundary nodes
    pub boundary_velocity: Array1<Complex64>,
    /// Wavenumber used in solution
    pub wavenumber: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::tetrahedral::{TetrahedralMesh, BoundaryType};

    fn create_test_mesh() -> TetrahedralMesh {
        let mut mesh = TetrahedralMesh::new();
        // Regular tetrahedron
        let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
        let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
        let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
        let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
        mesh.add_element([n0, n1, n2, n3], 0).unwrap();
        mesh
    }

    #[test]
    fn test_bem_solver_creation() {
        let config = BemConfig::default();
        let mesh = create_test_mesh();
        let solver = BemSolver::new(config, &mesh).unwrap();

        assert_eq!(solver.nodes.len(), 4);
        assert!(solver.boundary_manager_ref().is_empty());
        assert!(solver.h_matrix.is_none());
        assert!(solver.g_matrix.is_none());
    }

    #[test]
    fn test_bem_system_assembly() {
        let config = BemConfig::default();
        let mesh = create_test_mesh();
        let mut solver = BemSolver::new(config, &mesh).unwrap();

        solver.assemble_system().unwrap();

        assert!(solver.h_matrix.is_some());
        assert!(solver.g_matrix.is_some());

        let h = solver.h_matrix.unwrap();
        let g = solver.g_matrix.unwrap();

        // Check diagonal dominance of H (soft check)
        // Diagonal should be 0.5
        for i in 0..4 {
            let diag = h.get_diagonal(i);
            assert!((diag.re - 0.5).abs() < 1e-6, "Diagonal H should be 0.5");
        }

        // G diagonal should be non-zero (singular integral)
        for i in 0..4 {
            let diag = g.get_diagonal(i);
            assert!(diag.norm() > 1e-6, "Diagonal G should be non-zero");
        }
    }

    #[test]
    fn test_bem_boundary_conditions() {
        let config = BemConfig::default();
        let mesh = create_test_mesh();
        let mut solver = BemSolver::new(config, &mesh).unwrap();

        // Configure boundary conditions
        {
            let bc_manager = solver.boundary_manager();
            bc_manager.add_dirichlet(vec![(0, Complex64::new(1.0, 0.0))]);
            bc_manager.add_radiation(vec![1, 2, 3]);
        }

        assert_eq!(solver.boundary_manager_ref().len(), 2);

        // Assemble and solve
        solver.assemble_system().unwrap();
        let solution = solver.solve(1.0, None).unwrap();

        assert_eq!(solution.boundary_pressure.len(), 4);
        assert_eq!(solution.boundary_velocity.len(), 4);
        assert_eq!(solution.wavenumber, 1.0);
    }

    #[test]
    fn test_compute_scattered_field() {
        let config = BemConfig::default();
        let mesh = create_test_mesh();
        let solver = BemSolver::new(config, &mesh).unwrap();

        // Create a fake solution
        // Assume simple constant pressure/velocity
        let n = solver.nodes.len();
        let boundary_pressure = Array1::from_elem(n, Complex64::new(1.0, 0.0));
        let boundary_velocity = Array1::from_elem(n, Complex64::new(0.0, 0.0));

        let solution = BemSolution {
            boundary_pressure,
            boundary_velocity,
            wavenumber: 1.0,
        };

        // Evaluation point outside the tetrahedron
        // Tetra nodes: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        // Point (2,2,2) is outside
        let points = Array1::from_vec(vec![[2.0, 2.0, 2.0]]);

        let field = solver
            .compute_scattered_field(&points, &solution)
            .unwrap();

        assert_eq!(field.len(), 1);
        // Field should be non-zero because boundary pressure is non-zero
        // Integral of dG/dn * u should not be zero
        assert!(field[0].norm() > 1e-10, "Field should be non-zero");
    }
}
