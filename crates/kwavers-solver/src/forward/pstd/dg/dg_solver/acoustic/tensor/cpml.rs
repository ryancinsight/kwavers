//! CPML-aware RHS and SSP-RK3 stepper for the tensor acoustic DG solver.
//!
//! Implements the Roden-Gedney (2000) CPML stretching of the first-order
//! acoustic system, following Lazarov & Warburton (2009) for DG-specific
//! treatment of the per-axis volume + surface operator.
//!
//! # Theorem (DG-CPML field + auxiliary system)
//!
//! Let `D_a q` denote the DG strong-form semi-discrete spatial operator along
//! Cartesian axis `a` applied to grid function `q`, i.e. the sum of the
//! volume derivative `axis_scale · Σ_j (∂φ_j/∂ξ_a)(x_node) q_j` and the
//! interior-face lift residual. The CPML system is:
//!
//! ```text
//! ∂_t p   = -K  Σ_a [(1/κ_a) D_a u_a + ψ_{p, a}]
//! ∂_t u_a = -(1/ρ) [(1/κ_a) D_a p + ψ_{u_a, a}]    (only a-aligned velocity)
//! ∂_t ψ_{p, a}    = -(σ_a/κ_a + α_a) ψ_{p, a}    - (σ_a / κ_a²) D_a u_a
//! ∂_t ψ_{u_a, a}  = -(σ_a/κ_a + α_a) ψ_{u_a, a}  - (σ_a / κ_a²) D_a p
//! ```
//!
//! Outside the PML σ_a = 0, κ_a = 1, α_a = 0: the auxiliary RHS is zero and
//! the modified derivative collapses to `D_a q`, recovering the standard DG
//! tensor RHS bit-for-bit.
//!
//! The exterior face flux still uses [`super::boundary::add_axis_surface_flux`]
//! with the per-axis `boundary_conditions` policy, so any wave residual that
//! survives the absorbing layer is dissipated at the physical boundary rather
//! than reflected back through it.
//!
//! ## Joint SSP-RK3 integration
//!
//! Field state `q = [p, u_x, u_y, u_z]` and memory state
//! `ψ = [ψ_{p,x}, ψ_{p,y}, ψ_{p,z}, ψ_{u_x,x}, ψ_{u_y,y}, ψ_{u_z,z}]` are
//! integrated *jointly* by Shu-Osher SSP-RK3 with identical weights. Treating
//! ψ as additional conserved components avoids the O(dt) splitting error of
//! recursive-convolution post-updates and preserves the formal third-order
//! accuracy of the stepper.

use ndarray::{Array3, Zip};

use super::super::super::core::DGSolver;
use super::boundary::add_axis_surface_flux;
use super::{velocity_var, AcousticDgTensorWorkspace, ACOUSTIC_PRESSURE_VAR};
use crate::forward::pstd::dg::cpml::{
    memory::{pressure_memory_index, velocity_memory_index, DG_CPML_MEMORY_VARS},
    DgCpmlMemoryWorkspace, DgCpmlProfiles,
};
use kwavers_core::error::{KwaversError, KwaversResult};

impl DGSolver {
    /// Compute the CPML-aware tensor acoustic RHS for both field and memory state.
    ///
    /// `state` holds `[p, u_x, u_y, u_z]` at each GLL node; `memory_state` holds
    /// `[ψ_{p,x}, ψ_{p,y}, ψ_{p,z}, ψ_{u_x,x}, ψ_{u_y,y}, ψ_{u_z,z}]`.
    /// `profiles.axes[a].{sigma, kappa, alpha}` must be indexed by the global
    /// per-axis GLL node position (length `element_count[a] * n_nodes`).
    ///
    /// # Errors
    /// Returns an error for state/memory shape mismatches, profile axis-length
    /// mismatches, non-positive density, or non-positive sound speed.
    pub fn compute_acoustic_tensor_rhs_with_cpml_into(
        &self,
        state: &Array3<f64>,
        memory_state: &Array3<f64>,
        density: f64,
        profiles: &DgCpmlProfiles,
        rhs: &mut Array3<f64>,
        memory_rhs: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let topology = super::validate_tensor_state(self, state, density)?;
        let expected_field = state.dim();
        let expected_memory = (expected_field.0, expected_field.1, DG_CPML_MEMORY_VARS);
        if rhs.dim() != expected_field {
            return Err(KwaversError::InvalidInput(format!(
                "DG CPML field RHS shape {:?} does not match state {:?}",
                rhs.dim(),
                expected_field
            )));
        }
        if memory_state.dim() != expected_memory {
            return Err(KwaversError::InvalidInput(format!(
                "DG CPML memory state shape {:?} does not match expected {:?}",
                memory_state.dim(),
                expected_memory
            )));
        }
        if memory_rhs.dim() != expected_memory {
            return Err(KwaversError::InvalidInput(format!(
                "DG CPML memory RHS shape {:?} does not match expected {:?}",
                memory_rhs.dim(),
                expected_memory
            )));
        }
        for (axis, profile) in profiles.axes.iter().enumerate() {
            let expected = topology.element_counts[axis] * self.n_nodes;
            if profile.sigma.len() != expected
                || profile.kappa.len() != expected
                || profile.alpha.len() != expected
            {
                return Err(KwaversError::InvalidInput(format!(
                    "DG CPML profile axis {axis} length mismatch: σ={}, κ={}, α={}, expected {}",
                    profile.sigma.len(),
                    profile.kappa.len(),
                    profile.alpha.len(),
                    expected
                )));
            }
        }

        rhs.fill(0.0);
        memory_rhs.fill(0.0);
        let axis_scales = self.acoustic_axis_scales();
        let bulk = density * self.config().sound_speed * self.config().sound_speed;
        let inv_density = density.recip();
        let nodes_per_element = topology.nodes_per_element;

        // Pre-compute per-axis surface contributions across all elements.
        // `add_axis_surface_flux` only writes to the pressure variable and the
        // axis-aligned velocity variable; we route each axis's output to its own
        // Array3 so the per-axis bracket remains separable for the CPML formula.
        // Allocated once per call (3 × n_elements × nodes_per_element × 4 ×
        // 8 bytes). The CPML branch trades this extra working memory for a
        // clean per-axis separation that the standard RHS does not need.
        let mut surface_rhs_per_axis: [Array3<f64>; 3] = [
            Array3::zeros(state.dim()),
            Array3::zeros(state.dim()),
            Array3::zeros(state.dim()),
        ];
        for (axis, rhs_axis) in surface_rhs_per_axis.iter_mut().enumerate() {
            if !topology.active_axes[axis] {
                continue;
            }
            for elem in 0..topology.n_elements {
                add_axis_surface_flux(
                    self,
                    topology,
                    state,
                    elem,
                    axis,
                    bulk,
                    inv_density,
                    rhs_axis,
                );
            }
        }

        // Per-element scratch storing the per-axis DG strong-form derivative
        // `D_a u_a` and `D_a p` at every GLL node *before* CPML stretching.
        // Indexing: `du[axis * nodes_per_element + node]`. Allocated once per
        // call (size ≤ 3·(p+1)^3 doubles, e.g. 3·27 = 81 for p=2) and
        // overwritten per element.
        let mut axis_du = vec![0.0_f64; nodes_per_element * 3];
        let mut axis_dp = vec![0.0_f64; nodes_per_element * 3];

        for elem in 0..topology.n_elements {
            // 1) Volume per-axis raw derivatives.
            axis_du.iter_mut().for_each(|v| *v = 0.0);
            axis_dp.iter_mut().for_each(|v| *v = 0.0);
            for node in 0..nodes_per_element {
                let node_coords = topology.node_coords(node);
                for axis in 0..3 {
                    if !topology.active_axes[axis] {
                        continue;
                    }
                    let velocity_var = velocity_var(axis);
                    let mut du = 0.0;
                    let mut dp = 0.0;
                    for j in 0..self.n_nodes {
                        let source_node = topology.node_with_axis(node, axis, j);
                        du += self.diff_matrix[[node_coords[axis], j]]
                            * state[(elem, source_node, velocity_var)];
                        dp += self.diff_matrix[[node_coords[axis], j]]
                            * state[(elem, source_node, ACOUSTIC_PRESSURE_VAR)];
                    }
                    axis_du[axis * nodes_per_element + node] = axis_scales[axis] * du;
                    axis_dp[axis * nodes_per_element + node] = axis_scales[axis] * dp;
                }
            }

            // 2) Absorb the per-axis surface contribution into the same scratch.
            // The pre-pass deposited the lift residual into
            // `surface_rhs_per_axis[axis][(elem, node, var)]` with units
            // `axis_scale · (lift · face_res)`. The unmodified strong-form DG
            // equation `∂_t p = -bulk · D_a u_a` lets us recover the boundary
            // contribution to `D_a u_a` as `-surface_p / bulk`, and similarly
            // for `D_a p` from the velocity component.
            for axis in 0..3 {
                if !topology.active_axes[axis] {
                    continue;
                }
                let velocity_var = velocity_var(axis);
                for node in 0..nodes_per_element {
                    let surface_p = surface_rhs_per_axis[axis][(elem, node, ACOUSTIC_PRESSURE_VAR)];
                    let surface_u = surface_rhs_per_axis[axis][(elem, node, velocity_var)];
                    axis_du[axis * nodes_per_element + node] += -surface_p / bulk;
                    axis_dp[axis * nodes_per_element + node] += -surface_u / inv_density;
                }
            }

            // 3) Apply CPML stretching node-by-node.
            for node in 0..nodes_per_element {
                let node_coords = topology.node_coords(node);
                let elem_coords = topology.element_coords(elem);
                for axis in 0..3 {
                    if !topology.active_axes[axis] {
                        continue;
                    }
                    let velocity_var = velocity_var(axis);
                    let global_axis_node = elem_coords[axis] * self.n_nodes + node_coords[axis];
                    let sigma = profiles.axes[axis].sigma[global_axis_node];
                    let kappa = profiles.axes[axis].kappa[global_axis_node];
                    let alpha = profiles.axes[axis].alpha[global_axis_node];
                    let raw_du = axis_du[axis * nodes_per_element + node];
                    let raw_dp = axis_dp[axis * nodes_per_element + node];
                    let pi = pressure_memory_index(axis);
                    let vi = velocity_memory_index(axis);
                    let psi_p = memory_state[(elem, node, pi)];
                    let psi_u = memory_state[(elem, node, vi)];

                    // Field RHS: stretched derivative.
                    rhs[(elem, node, ACOUSTIC_PRESSURE_VAR)] -= bulk * (raw_du / kappa + psi_p);
                    rhs[(elem, node, velocity_var)] -= inv_density * (raw_dp / kappa + psi_u);

                    // Memory RHS: first-order auxiliary ODE.
                    let decay = sigma / kappa + alpha;
                    let drive = sigma / (kappa * kappa);
                    memory_rhs[(elem, node, pi)] = -decay * psi_p - drive * raw_du;
                    memory_rhs[(elem, node, vi)] = -decay * psi_u - drive * raw_dp;
                }
            }
        }
        Ok(())
    }

    /// Advance a tensor-product acoustic state by one CPML-aware SSP-RK3 step.
    ///
    /// Field state and memory state are integrated jointly with the Shu-Osher
    /// weights so the auxiliary ODE inherits the third-order RK accuracy of
    /// the field. Workspace buffers and the memory workspace are reused across
    /// time steps.
    ///
    /// # Errors
    /// Returns an error for shape, density, or profile mismatches.
    pub fn step_acoustic_tensor_ssp_rk3_with_cpml(
        &self,
        state: &mut Array3<f64>,
        density: f64,
        dt: f64,
        workspace: &mut AcousticDgTensorWorkspace,
        memory: &mut DgCpmlMemoryWorkspace,
        profiles: &DgCpmlProfiles,
    ) -> KwaversResult<()> {
        self.step_acoustic_tensor_ssp_rk3_with_cpml_and_source(
            state,
            density,
            dt,
            workspace,
            memory,
            profiles,
            0.0,
            |_, _| {},
        )
    }

    /// CPML-aware SSP-RK3 step with a time-dependent source callback.
    ///
    /// `add_source_rhs` receives the stage physical time and the mutable field
    /// RHS; sources are added before joint advancement. Memory state has no
    /// external forcing, so the source callback never touches it.
    ///
    /// # Errors
    /// Returns an error for shape, density, or profile mismatches.
    #[allow(clippy::too_many_arguments)]
    pub fn step_acoustic_tensor_ssp_rk3_with_cpml_and_source<F>(
        &self,
        state: &mut Array3<f64>,
        density: f64,
        dt: f64,
        workspace: &mut AcousticDgTensorWorkspace,
        memory: &mut DgCpmlMemoryWorkspace,
        profiles: &DgCpmlProfiles,
        t: f64,
        mut add_source_rhs: F,
    ) -> KwaversResult<()>
    where
        F: FnMut(f64, &mut Array3<f64>),
    {
        super::validate_tensor_state(self, state, density)?;
        workspace.ensure_dim(state.dim());
        memory.ensure_dim(state.dim().0, state.dim().1);
        workspace.original.assign(state);
        memory.original.assign(&memory.state);

        // Stage 1: q^{(1)} = q^n + dt·R(q^n)
        self.compute_acoustic_tensor_rhs_with_cpml_into(
            &workspace.original,
            &memory.original,
            density,
            profiles,
            &mut workspace.rhs,
            &mut memory.rhs,
        )?;
        add_source_rhs(t, &mut workspace.rhs);
        Zip::from(&mut workspace.stage)
            .and(&workspace.original)
            .and(&workspace.rhs)
            .for_each(|stage, &q0, &rhs| *stage = q0 + dt * rhs);
        Zip::from(&mut memory.stage)
            .and(&memory.original)
            .and(&memory.rhs)
            .for_each(|stage, &psi0, &rhs| *stage = psi0 + dt * rhs);

        // Stage 2: q^{(2)} = 3/4 q^n + 1/4 (q^{(1)} + dt·R(q^{(1)}))
        self.compute_acoustic_tensor_rhs_with_cpml_into(
            &workspace.stage,
            &memory.stage,
            density,
            profiles,
            &mut workspace.rhs,
            &mut memory.rhs,
        )?;
        add_source_rhs(t + dt, &mut workspace.rhs);
        Zip::from(&mut workspace.stage)
            .and(&workspace.original)
            .and(&workspace.rhs)
            .for_each(|stage, &q0, &rhs| {
                let q1 = *stage;
                *stage = 0.75 * q0 + 0.25 * (q1 + dt * rhs);
            });
        Zip::from(&mut memory.stage)
            .and(&memory.original)
            .and(&memory.rhs)
            .for_each(|stage, &psi0, &rhs| {
                let psi1 = *stage;
                *stage = 0.75 * psi0 + 0.25 * (psi1 + dt * rhs);
            });

        // Stage 3: q^{n+1} = 1/3 q^n + 2/3 (q^{(2)} + dt·R(q^{(2)}))
        self.compute_acoustic_tensor_rhs_with_cpml_into(
            &workspace.stage,
            &memory.stage,
            density,
            profiles,
            &mut workspace.rhs,
            &mut memory.rhs,
        )?;
        add_source_rhs(t + 0.5 * dt, &mut workspace.rhs);
        Zip::from(state)
            .and(&workspace.original)
            .and(&workspace.stage)
            .and(&workspace.rhs)
            .for_each(|q_new, &q0, &q2, &rhs| {
                *q_new = (1.0 / 3.0) * q0 + (2.0 / 3.0) * (q2 + dt * rhs);
            });
        Zip::from(&mut memory.state)
            .and(&memory.original)
            .and(&memory.stage)
            .and(&memory.rhs)
            .for_each(|psi_new, &psi0, &psi2, &rhs| {
                *psi_new = (1.0 / 3.0) * psi0 + (2.0 / 3.0) * (psi2 + dt * rhs);
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests;
