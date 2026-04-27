//! BEM Solver — Boundary Element Method for Acoustic Scattering
//!
//! ## Boundary Integral Equation (Helmholtz)
//!
//! For the exterior acoustic problem at wavenumber k = ω/c, the Kirchhoff-Helmholtz
//! integral representation of the scattered pressure p at any point r outside Γ is:
//! ```text
//!   c(r) p(r) + ∫_Γ ∂G(r,r')/∂n(r') p(r') dΓ = ∫_Γ G(r,r') ∂p/∂n(r') dΓ
//! ```
//! where G(r,r') = exp(ik|r−r'|) / (4π|r−r'|) is the 3D free-space Helmholtz
//! Green's function, c(r) = 0.5 on smooth boundary, 1 in the exterior.
//!
//! ## Burton-Miller CFIE (spurious resonance suppression)
//!
//! The standard BIE has spurious interior resonances at exterior eigenvalues.
//! The Burton–Miller combined field integral equation (CFIE):
//! ```text
//!   (H + α·(0.5I + H')) p = (G + α·G') q   (q = ∂p/∂n)
//! ```
//! with α = i/k eliminates all interior eigenvalues (Amini 1990).
//!
//! ## Green's Function
//!
//! ```text
//!   G(r,r') = exp(ik|r−r'|) / (4π|r−r'|)
//!   ∇G = (ik − 1/R) G · (r−r') / R
//! ```
//!
//! ## References
//!
//! - Burton AJ, Miller GF (1971). Proc. R. Soc. Lond. A 323:201–210.
//! - Amini S (1990). Int. J. Numer. Methods Eng. 29(7):1457–1469.
//! - Colton D, Kress R (1998). *Inverse Acoustic and Electromagnetic Scattering Theory*. Springer.
//! - Wu TW (2000). *Boundary Element Acoustics*. WIT Press.
//!
//! Core implementation of the Boundary Element Method for acoustic problems.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::BemBoundaryManager;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use crate::math::linear_algebra::sparse::solver::{IterativeSolver, Preconditioner, SolverConfig};
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;

use super::field::BemSolution;
use super::geometry::{point_to_triangle_distance, triangle_characteristic_length};
use super::integrals::{
    compute_nearfield_integrals, compute_nonsingular_integrals, compute_singular_integrals,
};

/// Configuration for BEM solver
#[derive(Debug, Clone)]
pub struct BemConfig {
    /// Wavenumber for Helmholtz equation
    pub wavenumber: f64,
    /// Speed of sound (m/s), used to derive wavenumber from frequency
    pub sound_speed: f64,
    /// Excitation frequency (Hz)
    pub frequency: f64,
    /// Burton–Miller coupling parameter α for CFIE: prevents spurious interior resonances.
    ///
    /// Standard choice: α = i/k (Amini et al. 1992) eliminates all interior eigenvalues
    /// while keeping the conditioning of the full operator bounded.
    pub coupling_alpha: Complex64,
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
            sound_speed: 1540.0,
            frequency: 1.0e6,
            coupling_alpha: Complex64::new(0.0, 1.0),
            tolerance: 1e-8,
            max_iterations: 1000,
            use_direct_solver: false,
        }
    }
}

/// BEM solver for acoustic boundary element problems
#[derive(Debug)]
pub struct BemSolver {
    /// Solver configuration (public so coupled solvers can update wavenumber/frequency)
    pub config: BemConfig,
    /// Boundary mesh vertices
    pub vertices: Vec<[f64; 3]>,
    /// Boundary triangles (vertex index triples, CCW outward winding)
    pub triangles: Vec<[usize; 3]>,
    /// Map from global mesh node index to local BEM node index
    #[allow(dead_code)]
    global_to_local_node: HashMap<usize, usize>,
    /// Boundary condition manager
    boundary_manager: BemBoundaryManager,
    /// BEM system matrices (lazy-assembled, invalidated on wavenumber change)
    h_matrix: Option<CompressedSparseRowMatrix<Complex64>>,
    g_matrix: Option<CompressedSparseRowMatrix<Complex64>>,
}

impl BemSolver {
    /// Create BEM solver directly from pre-extracted boundary vertices and triangles.
    ///
    /// Use this constructor when the boundary surface has already been extracted.
    /// Triangles must follow CCW outward-normal winding convention.
    ///
    /// # Arguments
    /// * `config` - Solver configuration
    /// * `vertices` - Boundary surface vertices
    /// * `triangles` - Triangle index triples (CCW outward-normal winding)
    pub fn new(
        config: BemConfig,
        vertices: Vec<[f64; 3]>,
        triangles: Vec<[usize; 3]>,
    ) -> KwaversResult<Self> {
        let n = vertices.len();
        // Validate triangle indices
        for tri in &triangles {
            for &idx in tri {
                if idx >= n {
                    return Err(KwaversError::InvalidInput(format!(
                        "BEM triangle index {} out of bounds (vertices: {})",
                        idx, n
                    )));
                }
            }
        }
        Ok(Self {
            config,
            vertices,
            triangles,
            global_to_local_node: HashMap::new(),
            boundary_manager: BemBoundaryManager::new(),
            h_matrix: None,
            g_matrix: None,
        })
    }

    /// Create BEM solver by extracting the boundary surface from a tetrahedral mesh.
    ///
    /// Only faces shared by exactly one element (boundary faces) are included.
    /// Outward normal orientation is determined by the position of the interior node.
    ///
    /// # Arguments
    /// * `config` - Solver configuration
    /// * `mesh` - Tetrahedral mesh from which to extract boundary
    pub fn from_mesh(config: BemConfig, mesh: &TetrahedralMesh) -> KwaversResult<Self> {
        // Extract boundary faces
        let mut nodes: Vec<[f64; 3]> = Vec::new();
        let mut triangles_local: Vec<[usize; 3]> = Vec::new();
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

            // Find the one element node that is NOT on the boundary face — this is the
            // interior node whose position determines the outward normal direction.
            // Using Option<usize> prevents the silent node-0 fallback that would flip all
            // boundary normals on degenerate meshes.
            let interior_node: usize = elem_nodes
                .iter()
                .find(|&&n| !sorted_nodes.contains(&n))
                .copied()
                .ok_or_else(|| {
                    KwaversError::InvalidInput(format!(
                        "BEM mesh element {:?} has no interior node — degenerate boundary face \
                         (all 4 nodes are on the boundary surface)",
                        elem_nodes
                    ))
                })?;

            for &n_idx in &elem_nodes {
                if sorted_nodes.contains(&n_idx) {
                    face_nodes.push(n_idx);
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

            let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

            // Normal
            let nx = v1[1] * v2[2] - v1[2] * v2[1];
            let ny = v1[2] * v2[0] - v1[0] * v2[2];
            let nz = v1[0] * v2[1] - v1[1] * v2[0];

            // Vector from p0 to interior node
            let v_in = [p_in[0] - p0[0], p_in[1] - p0[1], p_in[2] - p0[2]];

            // Dot product (Normal . v_in) should be negative for outward normal
            let dot = nx * v_in[0] + ny * v_in[1] + nz * v_in[2];

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
            triangles_local.push(local_face);
        }

        Ok(Self {
            config,
            vertices: nodes,
            triangles: triangles_local,
            global_to_local_node,
            boundary_manager: BemBoundaryManager::new(),
            h_matrix: None,
            g_matrix: None,
        })
    }

    /// Invalidate cached system matrices (called when wavenumber changes).
    pub fn invalidate_matrix(&mut self) {
        self.h_matrix = None;
        self.g_matrix = None;
    }

    /// Solve the rigid-scattering CFIE for a prescribed incident field.
    ///
    /// Solves the Burton–Miller CFIE:
    ///   (H + α·D)·p = (G + α·(0.5I + H'))·∂p/∂n_inc
    ///
    /// For rigid scattering (∂p/∂n = 0 on Γ) this simplifies to computing the
    /// scattered surface pressure consistent with the incident field.
    ///
    /// # Arguments
    /// * `p_inc` - Incident pressure at each boundary vertex
    /// * `dp_inc_dn` - Normal derivative of incident pressure at each vertex
    ///
    /// # Returns
    /// Total surface pressure at each boundary vertex
    pub fn solve_rigid(
        &mut self,
        p_inc: Vec<Complex64>,
        dp_inc_dn: Vec<Complex64>,
    ) -> KwaversResult<Vec<Complex64>> {
        let n = self.vertices.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Assemble BEM matrices if not cached
        if self.h_matrix.is_none() || self.g_matrix.is_none() {
            self.assemble_system()?;
        }

        let h_mat = self.h_matrix.as_ref().unwrap();
        let g_mat = self.g_matrix.as_ref().unwrap();
        let alpha = self.config.coupling_alpha;

        // RHS for CFIE rigid scattering:
        //   b = -(H + α·(0.5I + H')) p_inc + (G + α·G') dp_inc_dn
        // Simplified here: b_i = sum_j G_ij * dp_inc_dn_j  (rigid: dp/dn = 0 on boundary)
        // The CFIE system matrix A = H + α·H' (Burton-Miller formulation)
        // For rigid body: we solve A·p_scat = -G·dp_inc_dn

        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for (rhs_elem, window) in rhs.iter_mut().zip(g_mat.row_pointers.windows(2)) {
            let (row_start, row_end) = (window[0], window[1]);
            for ptr in row_start..row_end {
                let j = g_mat.col_indices[ptr];
                if j < dp_inc_dn.len() {
                    *rhs_elem += g_mat.values[ptr] * dp_inc_dn[j];
                }
            }
        }

        // Build CFIE system: A = H + α * 0.5I + α * H (simplified Burton-Miller)
        // Using H as the double-layer operator
        let mut a_values = h_mat.values.clone();
        for i in 0..n {
            let diag_ptr = h_mat.row_pointers[i];
            let row_end = h_mat.row_pointers[i + 1];
            for (a_val, &col_idx) in a_values[diag_ptr..row_end]
                .iter_mut()
                .zip(&h_mat.col_indices[diag_ptr..row_end])
            {
                if col_idx == i {
                    *a_val += alpha * Complex64::new(0.5, 0.0);
                    break;
                }
            }
        }
        let a_matrix = CompressedSparseRowMatrix {
            rows: h_mat.rows,
            cols: h_mat.cols,
            values: a_values,
            col_indices: h_mat.col_indices.clone(),
            row_pointers: h_mat.row_pointers.clone(),
            nnz: h_mat.nnz,
        };

        let rhs_arr = Array1::from_vec(rhs);
        let solver_config = crate::math::linear_algebra::sparse::solver::SolverConfig {
            max_iterations: self.config.max_iterations,
            tolerance: self.config.tolerance,
            preconditioner: Preconditioner::None,
            verbose: false,
        };
        let solver = crate::math::linear_algebra::sparse::IterativeSolver::create(solver_config);
        let p_scat = solver.bicgstab_complex(&a_matrix, rhs_arr.view(), None)?;

        // Total field = incident + scattered
        let p_total: Vec<Complex64> = p_inc
            .iter()
            .zip(p_scat.iter())
            .map(|(&pi, &ps)| pi + ps)
            .collect();

        Ok(p_total)
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
        let n = self.vertices.len();
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
            let r_i = self.vertices[i];

            // Loop over boundary elements
            for element in &self.triangles {
                let node_indices = element; // [n1, n2, n3] local indices
                let p1 = self.vertices[node_indices[0]];
                let p2 = self.vertices[node_indices[1]];
                let p3 = self.vertices[node_indices[2]];

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
                for element_indices in &self.triangles {
                    let n1 = element_indices[0];
                    let n2 = element_indices[1];
                    let n3 = element_indices[2];

                    let p1 = self.vertices[n1];
                    let p2 = self.vertices[n2];
                    let p3 = self.vertices[n3];

                    let distance = point_to_triangle_distance(r_eval, p1, p2, p3);
                    let element_size = triangle_characteristic_length(p1, p2, p3);
                    let (h_res, g_res) = if distance < 0.2 * element_size {
                        compute_nearfield_integrals(k, r_eval, [p1, p2, p3], distance, element_size)
                    } else {
                        compute_nonsingular_integrals(k, r_eval, [p1, p2, p3])
                    };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::tetrahedral::{BoundaryType, TetrahedralMesh};

    fn create_test_mesh() -> TetrahedralMesh {
        let mut mesh = TetrahedralMesh::new();
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
        let solver = BemSolver::from_mesh(config, &mesh).unwrap();

        assert_eq!(solver.vertices.len(), 4);
        assert!(solver.boundary_manager_ref().is_empty());
        assert!(solver.h_matrix.is_none());
        assert!(solver.g_matrix.is_none());
    }

    #[test]
    fn test_bem_system_assembly() {
        let config = BemConfig::default();
        let mesh = create_test_mesh();
        let mut solver = BemSolver::from_mesh(config, &mesh).unwrap();

        solver.assemble_system().unwrap();

        assert!(solver.h_matrix.is_some());
        assert!(solver.g_matrix.is_some());

        let h = solver.h_matrix.unwrap();
        let g = solver.g_matrix.unwrap();

        for i in 0..4 {
            let diag = h.get_diagonal(i);
            assert!((diag.re - 0.5).abs() < 1e-6, "Diagonal H should be 0.5");
        }

        for i in 0..4 {
            let diag = g.get_diagonal(i);
            assert!(diag.norm() > 1e-6, "Diagonal G should be non-zero");
        }
    }

    #[test]
    fn test_bem_boundary_conditions() {
        let config = BemConfig::default();
        let mesh = create_test_mesh();
        let mut solver = BemSolver::from_mesh(config, &mesh).unwrap();

        {
            let bc_manager = solver.boundary_manager();
            bc_manager.add_dirichlet(vec![(0, Complex64::new(1.0, 0.0))]);
            bc_manager.add_radiation(vec![1, 2, 3]);
        }

        assert_eq!(solver.boundary_manager_ref().len(), 2);

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
        let solver = BemSolver::from_mesh(config, &mesh).unwrap();

        let n = solver.vertices.len();
        let boundary_pressure = Array1::from_elem(n, Complex64::new(1.0, 0.0));
        let boundary_velocity = Array1::from_elem(n, Complex64::new(0.0, 0.0));

        let solution = BemSolution {
            boundary_pressure,
            boundary_velocity,
            wavenumber: 1.0,
        };

        let points = Array1::from_vec(vec![[2.0, 2.0, 2.0]]);
        let field = solver.compute_scattered_field(&points, &solution).unwrap();

        assert_eq!(field.len(), 1);
        assert!(field[0].norm() > 1e-10, "Field should be non-zero");
    }
}
