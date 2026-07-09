//! Per-node CPML auxiliary memory variables for the tensor acoustic DG solver.
//!
//! For each GLL node we carry six scalar memory variables `ψ_{q, a}`:
//!
//! | index | variable    | tracks the recursive convolution of |
//! | :---- | :---------- | :---------------------------------- |
//! | 0     | `ψ_{p, x}`  | `K · D_x u_x` contribution to `∂_t p` |
//! | 1     | `ψ_{p, y}`  | `K · D_y u_y` contribution to `∂_t p` |
//! | 2     | `ψ_{p, z}`  | `K · D_z u_z` contribution to `∂_t p` |
//! | 3     | `ψ_{ux, x}` | `(1/ρ) · D_x p` contribution to `∂_t u_x` |
//! | 4     | `ψ_{uy, y}` | `(1/ρ) · D_y p` contribution to `∂_t u_y` |
//! | 5     | `ψ_{uz, z}` | `(1/ρ) · D_z p` contribution to `∂_t u_z` |
//!
//! ## Theorem (CPML auxiliary ODE, Roden-Gedney 2000)
//!
//! Frequency-domain stretching `∂_a → ∂_a / (κ_a + σ_a/(α_a + iω))` is
//! equivalent in time domain to splitting the modified derivative into a
//! κ-rescaled physical derivative and an auxiliary convolution:
//!
//! ```text
//! D̃_a q = (1/κ_a) D_a q + ψ_{q, a}
//! d ψ_{q, a} / d t = -(σ_a / κ_a + α_a) ψ_{q, a} - (σ_a / κ_a²) D_a q
//! ```
//!
//! Under the SSP-RK3 stepper the auxiliary ODE is integrated *jointly* with the
//! field state — i.e., the memory variable is treated as a 7th, 8th, … component
//! of the conserved state and updated by the same RK weights. This avoids the
//! O(dt) splitting error of recursive-convolution post-updates.
//!
//! In the inner physical domain σ = 0 and α = 0 so the auxiliary RHS is
//! identically zero; the memory variable remains at its initial value (zero
//! for cold starts) and does not contribute to the field RHS. The CPML branch
//! therefore degenerates to the standard DG tensor RHS outside the layers.
//!
//! ## Memory layout
//!
//! `Array3<f64>` of shape `(n_elements, nodes_per_element, 6)` matching the
//! field state layout. One ψ component per axis-variable pair; inactive axes
//! carry zeros and are skipped during RHS evaluation.

use leto::Array3;

/// Number of auxiliary CPML memory variables per GLL node.
pub const DG_CPML_MEMORY_VARS: usize = 6;

/// `ψ_{p, axis}` index for the pressure equation.
#[inline]
#[must_use]
pub const fn pressure_memory_index(axis: usize) -> usize {
    axis
}

/// `ψ_{u_axis, axis}` index for the axis-aligned velocity equation.
#[inline]
#[must_use]
pub const fn velocity_memory_index(axis: usize) -> usize {
    3 + axis
}

/// Reusable workspace holding the CPML memory state and per-stage RK buffers.
///
/// `state` carries the persistent ψ values that survive across solver time
/// steps. `original`, `stage`, and `rhs` mirror the SSP-RK3 workspace layout of
/// the field state and are reused across stages to avoid per-step allocation.
#[derive(Debug, Clone)]
pub struct DgCpmlMemoryWorkspace {
    /// Persistent auxiliary state — shape `(n_elements, nodes_per_element, 6)`.
    pub state: Array3<f64>,
    /// Per-stage backup of the original ψ at step entry.
    pub original: Array3<f64>,
    /// Per-stage intermediate ψ.
    pub stage: Array3<f64>,
    /// Per-stage ψ RHS.
    pub rhs: Array3<f64>,
}

impl DgCpmlMemoryWorkspace {
    /// Allocate a workspace sized for `n_elements` × `nodes_per_element` GLL nodes.
    #[must_use]
    pub fn new(n_elements: usize, nodes_per_element: usize) -> Self {
        let dim = (n_elements, nodes_per_element, DG_CPML_MEMORY_VARS);
        Self {
            state: Array3::zeros(dim),
            original: Array3::zeros(dim),
            stage: Array3::zeros(dim),
            rhs: Array3::zeros(dim),
        }
    }

    /// Reshape the workspace if the field state geometry changed; leaves a
    /// freshly-zeroed memory state in that case (cold restart).
    pub fn ensure_dim(&mut self, n_elements: usize, nodes_per_element: usize) {
        let dim = (n_elements, nodes_per_element, DG_CPML_MEMORY_VARS);
        if self.state.shape() != dim {
            *self = Self::new(n_elements, nodes_per_element);
        }
    }

    /// Reset the persistent memory state to zero (cold restart).
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.original.fill(0.0);
        self.stage.fill(0.0);
        self.rhs.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_and_velocity_indices_are_disjoint() {
        let mut seen = [false; DG_CPML_MEMORY_VARS];
        for axis in 0..3 {
            let pi = pressure_memory_index(axis);
            let vi = velocity_memory_index(axis);
            assert!(!seen[pi], "duplicate pressure index {pi}");
            assert!(!seen[vi], "duplicate velocity index {vi}");
            seen[pi] = true;
            seen[vi] = true;
        }
        assert!(seen.iter().all(|x| *x));
    }

    #[test]
    fn new_workspace_is_cold_zero_state() {
        let ws = DgCpmlMemoryWorkspace::new(4, 9);
        assert_eq!(ws.state.shape(), (4, 9, DG_CPML_MEMORY_VARS));
        assert!(ws.state.iter().all(|v| *v == 0.0));
        assert!(ws.original.iter().all(|v| *v == 0.0));
        assert!(ws.stage.iter().all(|v| *v == 0.0));
        assert!(ws.rhs.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn ensure_dim_reshapes_when_geometry_changes() {
        let mut ws = DgCpmlMemoryWorkspace::new(4, 9);
        ws.state[(0, 0, 0)] = 1.5;
        ws.ensure_dim(8, 27);
        assert_eq!(ws.state.shape(), (8, 27, DG_CPML_MEMORY_VARS));
        assert!(ws.state.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn reset_zeros_all_buffers() {
        let mut ws = DgCpmlMemoryWorkspace::new(4, 9);
        ws.state[(0, 0, 0)] = 1.5;
        ws.stage[(1, 2, 3)] = -7.0;
        ws.rhs[(2, 3, 4)] = 11.0;
        ws.reset();
        assert!(ws.state.iter().all(|v| *v == 0.0));
        assert!(ws.stage.iter().all(|v| *v == 0.0));
        assert!(ws.rhs.iter().all(|v| *v == 0.0));
    }
}
