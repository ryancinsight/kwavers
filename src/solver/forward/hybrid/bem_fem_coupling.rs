//! BEM-FEM Coupling Implementation
//!
//! This module implements coupling between Boundary Element Method (BEM)
//! and Finite Element Method (FEM) for problems with complex interior geometries
//! and unbounded exterior domains.
//!
//! ## Mathematical Foundation
//!
//! BEM-FEM coupling is ideal for problems with:
//! - Complex interior geometries requiring FEM discretization
//! - Unbounded exterior domains naturally handled by BEM
//! - Radiation conditions at infinity automatically satisfied by BEM
//!
//! ## Coupling Strategy
//!
//! The coupling enforces continuity across the interface Γ:
//!
//! ```text
//! Interior Domain (FEM):   ∇²u - k²u = f    in Ω₁
//! Exterior Domain (BEM):   ∫_Γ G(x,y) ∂u/∂n(y) ds(y) = u(x)    on Γ
//! Interface Conditions:    u₁ = u₂, ∂u₁/∂n = ∂u₂/∂n    on Γ
//! ```
//!
//! ## Implementation Features
//!
//! - Interface continuity enforcement between FEM and BEM meshes
//! - Conservative field transfer across structured/unstructured interfaces
//! - Automatic radiation boundary conditions through BEM
//! - Support for complex geometries in FEM domain
//!
//! ## Literature References
//!
//! - Wu, T. (2000). "Pre-asymptotic error analysis of BEM and FEM coupling"
//! - Costabel, M. (1987). "Boundary integral operators for the heat equation"
//! - Johnson, C. & Nédélec, J. C. (1980). "On the coupling of boundary integral
//!   and finite element methods"

use crate::core::error::KwaversResult;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use crate::math::numerics::operators::TrilinearInterpolator;
use std::collections::HashMap;

/// Configuration for BEM-FEM coupling
#[derive(Debug, Clone)]
pub struct BemFemCouplingConfig {
    /// Coupling interface tolerance
    pub interface_tolerance: f64,
    /// Maximum coupling iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Relaxation factor for iterative coupling
    pub relaxation_factor: f64,
    /// Enable interface smoothing
    pub interface_smoothing: bool,
}

impl Default for BemFemCouplingConfig {
    fn default() -> Self {
        Self {
            interface_tolerance: 1e-6,
            max_iterations: 50,
            convergence_tolerance: 1e-8,
            relaxation_factor: 0.8,
            interface_smoothing: true,
        }
    }
}

/// Interface definition between FEM and BEM domains
#[derive(Debug, Clone)]
pub struct BemFemInterface {
    /// FEM mesh nodes at interface
    pub fem_interface_nodes: Vec<usize>,
    /// BEM boundary elements at interface
    pub bem_interface_elements: Vec<usize>,
    /// Interface mapping (FEM node index → BEM element index)
    pub node_element_mapping: HashMap<usize, usize>,
    /// Interface quadrature points and weights
    pub quadrature_points: Vec<(f64, f64, f64)>,
    pub quadrature_weights: Vec<f64>,
    /// Interface normals (pointing from FEM to BEM domain)
    pub interface_normals: Vec<(f64, f64, f64)>,
}

impl BemFemInterface {
    /// Create BEM-FEM interface from FEM mesh and BEM boundary
    pub fn new(fem_mesh: &TetrahedralMesh, bem_boundary: &[usize]) -> KwaversResult<Self> {
        // Find FEM nodes that lie on the BEM boundary
        let mut fem_interface_nodes = Vec::new();
        let mut node_element_mapping = HashMap::new();

        // For each FEM node, check if it's on the BEM boundary
        for (node_idx, node) in fem_mesh.nodes.iter().enumerate() {
            // Check if this node is on the BEM interface
            // This would typically involve geometric checks or pre-computed mappings
            let is_interface = Self::is_node_on_interface(node, bem_boundary, fem_mesh);

            if is_interface {
                fem_interface_nodes.push(node_idx);

                // Find corresponding BEM element (simplified)
                // In practice, this would involve more sophisticated geometric queries
                let bem_element =
                    Self::find_corresponding_bem_element(node, bem_boundary, fem_mesh)?;
                node_element_mapping.insert(node_idx, bem_element);
            }
        }

        // Generate interface geometry
        let (quadrature_points, quadrature_weights) =
            Self::generate_interface_quadrature(&fem_interface_nodes, fem_mesh);
        let interface_normals = Self::compute_interface_normals(&fem_interface_nodes, fem_mesh);

        Ok(Self {
            fem_interface_nodes,
            bem_interface_elements: bem_boundary.to_vec(),
            node_element_mapping,
            quadrature_points,
            quadrature_weights,
            interface_normals,
        })
    }

    /// Check if a FEM node lies on the BEM interface
    fn is_node_on_interface(
        _node: &crate::domain::mesh::tetrahedral::MeshNode,
        _bem_boundary: &[usize],
        _fem_mesh: &TetrahedralMesh,
    ) -> bool {
        // TODO: Simplified geometric check - in practice would check if node
        // coordinates lie on the boundary surface defined by BEM elements
        // For now, assume some nodes are on the interface
        false // Placeholder - would need actual geometric computation
    }

    /// Find corresponding BEM element for a FEM node
    fn find_corresponding_bem_element(
        node: &crate::domain::mesh::tetrahedral::MeshNode,
        bem_boundary: &[usize],
        fem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<usize> {
        bem_boundary
            .iter()
            .copied()
            .filter_map(|idx| {
                fem_mesh.nodes.get(idx).map(|bem_node| {
                    let dx = node.coordinates[0] - bem_node.coordinates[0];
                    let dy = node.coordinates[1] - bem_node.coordinates[1];
                    let dz = node.coordinates[2] - bem_node.coordinates[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    (dist_sq, idx)
                })
            })
            .min_by(|(d1, _), (d2, _)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, idx)| idx)
            .ok_or_else(|| {
                crate::core::error::KwaversError::InvalidInput(
                    "Could not find corresponding BEM element: boundary list empty or invalid indices".to_string(),
                )
            })
    }

    /// Generate quadrature points and weights for interface integration
    fn generate_interface_quadrature(
        fem_nodes: &[usize],
        fem_mesh: &TetrahedralMesh,
    ) -> (Vec<(f64, f64, f64)>, Vec<f64>) {
        let mut points = Vec::new();
        let mut weights = Vec::new();

        for &node_idx in fem_nodes {
            if let Some(node) = fem_mesh.nodes.get(node_idx) {
                points.push((
                    node.coordinates[0],
                    node.coordinates[1],
                    node.coordinates[2],
                ));
                weights.push(1.0 / fem_nodes.len() as f64); // Equal weights
            }
        }

        (points, weights)
    }

    /// Compute interface normals
    fn compute_interface_normals(
        fem_nodes: &[usize],
        fem_mesh: &TetrahedralMesh,
    ) -> Vec<(f64, f64, f64)> {
        let mut normals = Vec::new();

        for &node_idx in fem_nodes {
            if let Some(_node) = fem_mesh.nodes.get(node_idx) {
                // TODO: Compute normal vector pointing outward from FEM domain
                // In practice, this would involve surface normal computation
                normals.push((0.0, 0.0, 1.0)); // Placeholder normal
            }
        }

        normals
    }
}

/// BEM-FEM Coupling Solver
#[derive(Debug)]
pub struct BemFemCoupler {
    config: BemFemCouplingConfig,
    interface: BemFemInterface,
    #[allow(dead_code)]
    fem_interpolator: TrilinearInterpolator,
    convergence_history: Vec<f64>,
    iteration_count: usize,
}

impl BemFemCoupler {
    /// Create new BEM-FEM coupler
    pub fn new(
        config: BemFemCouplingConfig,
        fem_mesh: &TetrahedralMesh,
        bem_boundary: &[usize],
    ) -> KwaversResult<Self> {
        let interface = BemFemInterface::new(fem_mesh, bem_boundary)?;
        // Note: TrilinearInterpolator is used as placeholder - would need mesh interpolator
        let fem_interpolator = TrilinearInterpolator::new(0.001, 0.001, 0.001);

        Ok(Self {
            config,
            interface,
            fem_interpolator,
            convergence_history: Vec::new(),
            iteration_count: 0,
        })
    }

    /// Perform coupled BEM-FEM solution
    pub fn solve_coupled(
        &mut self,
        fem_field: &mut Vec<f64>,
        bem_boundary_values: &mut Vec<f64>,
        fem_mesh: &TetrahedralMesh,
        wavenumber: f64,
    ) -> KwaversResult<f64> {
        self.convergence_history.clear();
        let mut residual = f64::INFINITY;

        for iteration in 0..self.config.max_iterations {
            // 1. Extract FEM solution at interface
            let fem_interface_values = self.extract_fem_interface(fem_field.as_slice())?;

            // 2. Apply to BEM boundary conditions
            self.apply_to_bem_boundary(&fem_interface_values, bem_boundary_values.as_mut_slice())?;

            // 3. Solve BEM system (placeholder - would call actual BEM solver)
            self.solve_bem_system(bem_boundary_values.as_mut_slice(), wavenumber)?;

            // 4. Extract BEM solution at interface
            let bem_interface_values =
                self.extract_bem_interface(bem_boundary_values.as_slice())?;

            // 5. Apply to FEM boundary conditions with relaxation
            residual = self.apply_to_fem_boundary(
                &bem_interface_values,
                fem_field.as_mut_slice(),
                fem_mesh,
            )?;

            // 6. Solve FEM system (placeholder - would call actual FEM solver)
            self.solve_fem_system(fem_field.as_mut_slice(), fem_mesh)?;

            self.convergence_history.push(residual);
            self.iteration_count = iteration + 1;

            if residual < self.config.convergence_tolerance {
                break;
            }
        }

        Ok(residual)
    }

    /// Extract FEM field values at interface
    fn extract_fem_interface(&self, fem_field: &[f64]) -> KwaversResult<Vec<f64>> {
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

    /// Apply FEM interface values to BEM boundary
    fn apply_to_bem_boundary(
        &self,
        fem_values: &[f64],
        bem_boundary_values: &mut [f64],
    ) -> KwaversResult<()> {
        // Map FEM interface values to BEM boundary values
        for (i, &fem_value) in fem_values.iter().enumerate() {
            if i < self.interface.fem_interface_nodes.len() {
                let fem_node_idx = self.interface.fem_interface_nodes[i];
                if let Some(&bem_element_idx) =
                    self.interface.node_element_mapping.get(&fem_node_idx)
                {
                    if bem_element_idx < bem_boundary_values.len() {
                        // Apply with relaxation for stability
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

    /// Extract BEM solution at interface
    fn extract_bem_interface(&self, bem_boundary_values: &[f64]) -> KwaversResult<Vec<f64>> {
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

    /// Apply BEM interface values to FEM boundary
    fn apply_to_fem_boundary(
        &self,
        bem_values: &[f64],
        fem_field: &mut [f64],
        _fem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<f64> {
        let mut max_residual: f64 = 0.0;

        // Apply BEM values to FEM interface nodes
        for (i, &bem_value) in bem_values.iter().enumerate() {
            if i < self.interface.fem_interface_nodes.len() {
                let fem_node_idx = self.interface.fem_interface_nodes[i];
                if fem_node_idx < fem_field.len() {
                    let current_value = fem_field[fem_node_idx];
                    let new_value = self.config.relaxation_factor * bem_value
                        + (1.0 - self.config.relaxation_factor) * current_value;

                    // Compute residual for convergence
                    let residual = (new_value - current_value).abs();
                    max_residual = max_residual.max(residual);

                    fem_field[fem_node_idx] = new_value;
                }
            }
        }

        Ok(max_residual)
    }

    /// Solve BEM system (placeholder)
    /// TODO_AUDIT: P2 - Advanced Hybrid Methods - Implement full BEM-FEM coupling with optimized preconditioners and parallel domain decomposition
    /// DEPENDS ON: solver/forward/hybrid/bem_fem/preconditioners.rs, solver/forward/hybrid/bem_fem/domain_decomp.rs, solver/forward/hybrid/bem_fem/optimization.rs
    /// MISSING: Fast multipole method (FMM) for efficient BEM matrix-vector products
    /// MISSING: FETI-DP domain decomposition for parallel BEM-FEM coupling
    /// MISSING: Optimized Schwarz alternating methods for convergence acceleration
    /// MISSING: Adaptive mesh refinement at BEM-FEM interfaces
    /// MISSING: GPU acceleration for large-scale hybrid simulations
    /// THEOREM: Fast multipole method: O(N) complexity vs O(N²) for direct BEM
    /// THEOREM: FETI-DP: Interface problem ensures continuity across subdomains
    /// REFERENCES: Rokhlin (1985) J Comput Phys; Farhat & Roux (1991) Int J Numer Methods Eng
    fn solve_bem_system(
        &self,
        _bem_boundary_values: &mut [f64],
        _wavenumber: f64,
    ) -> KwaversResult<()> {
        // TODO: Placeholder for BEM system solution
        // In practice, this would solve the boundary integral equations
        // using the BEM matrices and the boundary values
        Ok(())
    }

    /// Solve FEM system (placeholder)
    fn solve_fem_system(
        &self,
        _fem_field: &mut [f64],
        _fem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<()> {
        // TODO: Placeholder for FEM system solution
        // In practice, this would solve the finite element system
        // using the FEM stiffness/mass matrices
        Ok(())
    }

    /// Get convergence history
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Check if coupling has converged
    pub fn has_converged(&self) -> bool {
        if self.convergence_history.is_empty() {
            return false;
        }

        let last_residual = *self.convergence_history.last().unwrap();
        last_residual < self.config.convergence_tolerance
    }

    /// Get number of iterations performed
    pub fn iterations(&self) -> usize {
        self.iteration_count
    }

    /// Reset convergence tracking
    pub fn reset(&mut self) {
        self.convergence_history.clear();
        self.iteration_count = 0;
    }
}

/// BEM-FEM Coupled Solver for Helmholtz problems
#[derive(Debug)]
pub struct BemFemSolver {
    config: BemFemCouplingConfig,
    coupler: BemFemCoupler,
    fem_mesh: TetrahedralMesh,
    #[allow(dead_code)]
    bem_boundary_elements: Vec<usize>,
    wavenumber: f64,
}

impl BemFemSolver {
    /// Create new BEM-FEM coupled solver
    pub fn new(
        config: BemFemCouplingConfig,
        fem_mesh: TetrahedralMesh,
        bem_boundary_elements: Vec<usize>,
        wavenumber: f64,
    ) -> KwaversResult<Self> {
        let coupler = BemFemCoupler::new(config.clone(), &fem_mesh, &bem_boundary_elements)?;

        Ok(Self {
            config,
            coupler,
            fem_mesh,
            bem_boundary_elements,
            wavenumber,
        })
    }

    /// Solve the coupled BEM-FEM system
    pub fn solve(
        &mut self,
        fem_initial_guess: Vec<f64>,
        bem_boundary_guess: Vec<f64>,
    ) -> KwaversResult<()> {
        self.coupler.solve_coupled(
            &mut fem_initial_guess.clone(),
            &mut bem_boundary_guess.clone(),
            &self.fem_mesh,
            self.wavenumber,
        )?;

        Ok(())
    }

    /// Get the coupling interface
    pub fn interface(&self) -> &BemFemInterface {
        &self.coupler.interface
    }

    /// Get convergence information
    pub fn convergence_info(&self) -> (bool, usize, &[f64]) {
        (
            self.coupler.has_converged(),
            self.coupler.iterations(),
            self.coupler.convergence_history(),
        )
    }

    /// Get configuration
    pub fn config(&self) -> &BemFemCouplingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::tetrahedral::BoundaryType;

    #[test]
    fn test_bem_fem_interface_creation() {
        // Create simple tetrahedral mesh
        let mut fem_mesh = TetrahedralMesh::new();

        // Add some nodes
        fem_mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
        fem_mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
        fem_mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
        fem_mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);

        // Define BEM boundary elements (placeholder)
        let bem_boundary = vec![0, 1, 2];

        let interface = BemFemInterface::new(&fem_mesh, &bem_boundary);

        // Interface creation should succeed even with simplified geometry
        assert!(interface.is_ok());
    }

    #[test]
    fn test_bem_fem_coupler_creation() {
        let mut fem_mesh = TetrahedralMesh::new();
        fem_mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);

        let bem_boundary = vec![0];
        let config = BemFemCouplingConfig::default();

        let coupler = BemFemCoupler::new(config, &fem_mesh, &bem_boundary);

        assert!(coupler.is_ok());
    }

    #[test]
    fn test_bem_fem_solver_creation() {
        let mut fem_mesh = TetrahedralMesh::new();
        fem_mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);

        let bem_boundary = vec![0];
        let config = BemFemCouplingConfig::default();
        let wavenumber = 2.0 * std::f64::consts::PI * 1e6 / 1482.0; // k = 2πf/c

        let solver = BemFemSolver::new(config, fem_mesh, bem_boundary, wavenumber);

        assert!(solver.is_ok());
    }

    #[test]
    fn test_coupling_config_defaults() {
        let config = BemFemCouplingConfig::default();

        assert_eq!(config.max_iterations, 50);
        assert!(config.convergence_tolerance > 0.0);
        assert!(config.relaxation_factor > 0.0 && config.relaxation_factor <= 1.0);
    }

    #[test]
    fn test_find_corresponding_bem_element() {
        let mut fem_mesh = TetrahedralMesh::new();

        // Node 0: Origin (Query node)
        let n0 = fem_mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);

        // Node 1: (1, 0, 0)
        let n1 = fem_mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);

        // Node 2: (2, 0, 0)
        let n2 = fem_mesh.add_node([2.0, 0.0, 0.0], BoundaryType::Interior);

        // Node 3: (0.5, 0, 0) - Closest
        let n3 = fem_mesh.add_node([0.5, 0.0, 0.0], BoundaryType::Interior);

        // BEM boundary candidates: n1, n2, n3
        let bem_boundary = vec![n1, n2, n3];

        let query_node = fem_mesh.nodes[n0];

        // Access the private function via the associated function on the struct
        // Since we are in a child module, we can access private items of parent
        let closest_idx =
            BemFemInterface::find_corresponding_bem_element(&query_node, &bem_boundary, &fem_mesh);

        assert_eq!(
            closest_idx.unwrap(),
            n3,
            "Should find the node at distance 0.5"
        );
    }
}
