use nalgebra::{Matrix3, Vector3};
use ndarray::Array1;
use num_complex::{Complex64, ComplexFloat};

use crate::core::error::KwaversResult;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use crate::math::linear_algebra::sparse::solver::Preconditioner;
use crate::math::linear_algebra::sparse::{
    CompressedSparseRowMatrix, CoordinateMatrix, IterativeSolver, SolverConfig,
};
use crate::math::numerics::operators::TrilinearInterpolator;
use crate::solver::forward::bem::solver::{BemConfig, BemSolver};

use super::super::{BemFemCouplingConfig, BemFemInterface};

/// BEM-FEM Coupling Solver
#[derive(Debug)]
pub struct BemFemCoupler {
    config: BemFemCouplingConfig,
    pub(crate) interface: BemFemInterface,
    _fem_interpolator: TrilinearInterpolator,
    convergence_history: Vec<f64>,
    iteration_count: usize,
    bem_solver: BemSolver,
}

impl BemFemCoupler {
    /// Create new BEM-FEM coupler.
    pub fn new(
        config: BemFemCouplingConfig,
        fem_mesh: &TetrahedralMesh,
        bem_boundary: &[usize],
    ) -> KwaversResult<Self> {
        let interface = BemFemInterface::new(fem_mesh, bem_boundary)?;

        let bb = &fem_mesh.bounding_box;
        let lx = (bb.max[0] - bb.min[0]).max(1e-12);
        let ly = (bb.max[1] - bb.min[1]).max(1e-12);
        let lz = (bb.max[2] - bb.min[2]).max(1e-12);
        let l_max = lx.max(ly).max(lz);
        let n_e = fem_mesh.elements.len().max(1) as f64;
        let n1d = n_e.cbrt();
        let dx = lx / (n1d * lx / l_max).max(1.0);
        let dy = ly / (n1d * ly / l_max).max(1.0);
        let dz = lz / (n1d * lz / l_max).max(1.0);
        let fem_interpolator = TrilinearInterpolator::new(dx, dy, dz);

        let bem_config = BemConfig::default();
        let boundary_verts: Vec<[f64; 3]> = fem_mesh.nodes.iter().map(|n| n.coordinates).collect();
        let boundary_tris: Vec<[usize; 3]> = fem_mesh
            .boundary_faces
            .keys()
            .filter_map(|face| {
                if face.len() == 3 {
                    Some([face[0], face[1], face[2]])
                } else {
                    None
                }
            })
            .collect();
        let bem_solver = BemSolver::new(bem_config, boundary_verts, boundary_tris)?;

        Ok(Self {
            config,
            interface,
            _fem_interpolator: fem_interpolator,
            convergence_history: Vec::new(),
            iteration_count: 0,
            bem_solver,
        })
    }

    /// Perform coupled BEM-FEM solution.
    pub fn solve_coupled(
        &mut self,
        fem_field: &mut Vec<Complex64>,
        bem_boundary_values: &mut Vec<Complex64>,
        fem_mesh: &TetrahedralMesh,
        wavenumber: f64,
    ) -> KwaversResult<f64> {
        self.convergence_history.clear();
        let mut residual = f64::INFINITY;

        let fem_matrix = self.assemble_system_matrix(fem_mesh, wavenumber)?;

        for iteration in 0..self.config.max_iterations {
            let fem_interface_values = self.extract_fem_interface(fem_field.as_slice())?;
            self.apply_to_bem_boundary(&fem_interface_values, bem_boundary_values.as_mut_slice())?;
            self.solve_bem_system(bem_boundary_values.as_mut_slice(), wavenumber)?;
            let bem_interface_values =
                self.extract_bem_interface(bem_boundary_values.as_slice())?;
            residual = self.apply_to_fem_boundary(
                &bem_interface_values,
                fem_field.as_mut_slice(),
                fem_mesh,
            )?;
            self.solve_linear_system(&fem_matrix, fem_field.as_mut_slice())?;

            self.convergence_history.push(residual);
            self.iteration_count = iteration + 1;

            if residual < self.config.convergence_tolerance {
                break;
            }
        }

        Ok(residual)
    }

    /// Extract FEM field values at interface.
    fn extract_fem_interface(&self, fem_field: &[Complex64]) -> KwaversResult<Vec<Complex64>> {
        let mut interface_values = Vec::new();
        for &node_idx in &self.interface.fem_interface_nodes {
            let value = fem_field.get(node_idx).ok_or_else(|| {
                crate::core::error::KwaversError::InvalidInput(format!(
                    "FEM interface node index {} is out of bounds (fem_field len {})",
                    node_idx,
                    fem_field.len()
                ))
            })?;
            interface_values.push(*value);
        }
        Ok(interface_values)
    }

    /// Apply FEM interface values to BEM boundary.
    fn apply_to_bem_boundary(
        &self,
        fem_values: &[Complex64],
        bem_boundary_values: &mut [Complex64],
    ) -> KwaversResult<()> {
        for (i, &fem_value) in fem_values.iter().enumerate() {
            if i < self.interface.fem_interface_nodes.len() {
                let fem_node_idx = self.interface.fem_interface_nodes[i];
                if let Some(&bem_element_idx) =
                    self.interface.node_element_mapping.get(&fem_node_idx)
                {
                    if bem_element_idx < bem_boundary_values.len() {
                        let current_value = bem_boundary_values[bem_element_idx];
                        bem_boundary_values[bem_element_idx] = self.config.relaxation_factor
                            * fem_value
                            + (1.0 - self.config.relaxation_factor) * current_value;
                    }
                }
            }
        }
        Ok(())
    }

    /// Extract BEM solution at interface.
    fn extract_bem_interface(
        &self,
        bem_boundary_values: &[Complex64],
    ) -> KwaversResult<Vec<Complex64>> {
        let mut interface_values = Vec::new();
        for &bem_element_idx in &self.interface.bem_interface_elements {
            let value = bem_boundary_values.get(bem_element_idx).ok_or_else(|| {
                crate::core::error::KwaversError::InvalidInput(format!(
                    "BEM interface element index {} is out of bounds (bem_boundary_values len {})",
                    bem_element_idx,
                    bem_boundary_values.len()
                ))
            })?;
            interface_values.push(*value);
        }
        Ok(interface_values)
    }

    /// Apply BEM interface values to FEM boundary.
    fn apply_to_fem_boundary(
        &self,
        bem_values: &[Complex64],
        fem_field: &mut [Complex64],
        _fem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<f64> {
        let mut max_residual: f64 = 0.0;
        for (i, &bem_value) in bem_values.iter().enumerate() {
            if i < self.interface.fem_interface_nodes.len() {
                let fem_node_idx = self.interface.fem_interface_nodes[i];
                if fem_node_idx < fem_field.len() {
                    let current_value = fem_field[fem_node_idx];
                    let new_value = self.config.relaxation_factor * bem_value
                        + (1.0 - self.config.relaxation_factor) * current_value;
                    let residual = (new_value - current_value).abs();
                    max_residual = max_residual.max(residual);
                    fem_field[fem_node_idx] = new_value;
                }
            }
        }
        Ok(max_residual)
    }

    /// Solve BEM system via rigid-scattering CFIE.
    fn solve_bem_system(
        &mut self,
        bem_boundary_values: &mut [Complex64],
        wavenumber: f64,
    ) -> KwaversResult<()> {
        use crate::solver::forward::bem::field::{compute_vertex_normals, plane_wave_incident};

        let nv = self.bem_solver.vertices.len();
        if nv == 0 {
            return Ok(());
        }

        let c = self.bem_solver.config.sound_speed;
        let f = wavenumber * c / (2.0 * std::f64::consts::PI);
        self.bem_solver.config.frequency = f;
        self.bem_solver.config.wavenumber = wavenumber;
        self.bem_solver.config.coupling_alpha = num_complex::Complex64::new(0.0, 1.0 / wavenumber);
        self.bem_solver.invalidate_matrix();

        let normals = compute_vertex_normals(&self.bem_solver.vertices, &self.bem_solver.triangles);
        let (p_inc, dp_inc_dn) = plane_wave_incident(
            &self.bem_solver.vertices,
            &normals,
            [1.0, 0.0, 0.0],
            wavenumber,
            num_complex::Complex64::new(1.0, 0.0),
        );

        let p_surface = self.bem_solver.solve_rigid(p_inc, dp_inc_dn)?;

        for (local_idx, &global_idx) in self.interface.bem_interface_elements.iter().enumerate() {
            if global_idx < bem_boundary_values.len() && local_idx < p_surface.len() {
                bem_boundary_values[global_idx] = p_surface[local_idx];
            }
        }

        Ok(())
    }

    /// Assemble FEM system matrix.
    pub(super) fn assemble_system_matrix(
        &self,
        fem_mesh: &TetrahedralMesh,
        wavenumber: f64,
    ) -> KwaversResult<CompressedSparseRowMatrix<Complex64>> {
        let num_nodes = fem_mesh.nodes.len();
        let mut coo = CoordinateMatrix::create(num_nodes, num_nodes);

        let grad_ref = [
            Vector3::new(-1.0, -1.0, -1.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        for element in &fem_mesh.elements {
            let n_indices = element.nodes;
            let p0 = Vector3::from(fem_mesh.nodes[n_indices[0]].coordinates);
            let p1 = Vector3::from(fem_mesh.nodes[n_indices[1]].coordinates);
            let p2 = Vector3::from(fem_mesh.nodes[n_indices[2]].coordinates);
            let p3 = Vector3::from(fem_mesh.nodes[n_indices[3]].coordinates);

            let edge1 = p1 - p0;
            let edge2 = p2 - p0;
            let edge3 = p3 - p0;

            let jacobian = Matrix3::from_columns(&[edge1, edge2, edge3]);

            if let Some(inv_j) = jacobian.try_inverse() {
                let inv_j_t = inv_j.transpose();
                let det_j = jacobian.determinant().abs();
                let volume = det_j / 6.0;

                let mut grads = [Vector3::zeros(); 4];
                for k in 0..4 {
                    grads[k] = inv_j_t * grad_ref[k];
                }

                for i in 0..4 {
                    for j in 0..4 {
                        let k_val = grads[i].dot(&grads[j]) * volume;
                        let delta = if i == j { 1.0 } else { 0.0 };
                        let m_val = (1.0 + delta) * volume / 20.0;
                        let val =
                            Complex64::from(k_val) - Complex64::from(wavenumber.powi(2) * m_val);
                        coo.add_triplet(n_indices[i], n_indices[j], val);
                    }
                }
            } else {
                return Err(crate::core::error::KwaversError::Numerical(
                    crate::core::error::NumericalError::SingularMatrix {
                        operation: "element_jacobian".to_string(),
                        condition_number: 0.0,
                    },
                ));
            }
        }

        let penalty = 1.0e14;
        for &node_idx in &self.interface.fem_interface_nodes {
            if node_idx < num_nodes {
                coo.add_triplet(node_idx, node_idx, Complex64::from(penalty));
            }
        }

        Ok(coo.to_csr())
    }

    /// Solve linear system using pre-assembled matrix.
    pub(super) fn solve_linear_system(
        &self,
        matrix: &CompressedSparseRowMatrix<Complex64>,
        fem_field: &mut [Complex64],
    ) -> KwaversResult<()> {
        let num_nodes = matrix.rows;
        let penalty = 1.0e14;
        let mut rhs = Array1::<Complex64>::zeros(num_nodes);

        for &node_idx in &self.interface.fem_interface_nodes {
            if node_idx < num_nodes {
                let prescribed_val = fem_field[node_idx];
                rhs[node_idx] += Complex64::from(penalty) * prescribed_val;
            }
        }

        let config = SolverConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
            preconditioner: Preconditioner::None,
            verbose: false,
        };
        let solver = IterativeSolver::create(config);

        let initial_guess = Array1::from_vec(fem_field.to_vec());
        let solution = solver.bicgstab_complex(matrix, rhs.view(), Some(initial_guess.view()))?;

        for i in 0..num_nodes {
            fem_field[i] = solution[i];
        }

        Ok(())
    }

    /// Get convergence history.
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Check if coupling has converged.
    pub fn has_converged(&self) -> bool {
        if self.convergence_history.is_empty() {
            return false;
        }
        let last_residual = *self.convergence_history.last().unwrap();
        last_residual < self.config.convergence_tolerance
    }

    /// Get number of iterations performed.
    pub fn iterations(&self) -> usize {
        self.iteration_count
    }

    /// Reset convergence tracking.
    pub fn reset(&mut self) {
        self.convergence_history.clear();
        self.iteration_count = 0;
    }
}
