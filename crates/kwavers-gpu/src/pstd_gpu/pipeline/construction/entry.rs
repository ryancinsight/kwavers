//! `GpuPstdSolver::new`: input validation ahead of provider state construction.

use super::super::super::GpuPstdSolver;
use super::super::{AbsorptionArrays, MediumArrays, PmlArrays, SolverParams};
use crate::pstd_gpu::state::PstdStateBuilder;
use kwavers_grid::Grid;

impl<P> GpuPstdSolver<P>
where
    P: PstdStateBuilder,
{
    /// Create a new GPU PSTD solver.
    ///
    /// See `mod.rs` for argument documentation and bind-group layout.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any input array shape is inconsistent with `grid` or
    /// if the provider cannot build its execution state.
    pub fn new(
        context: P::Context,
        grid: &Grid,
        medium: MediumArrays<'_>,
        solver: SolverParams,
        pml: PmlArrays<'_>,
        absorption: AbsorptionArrays<'_>,
    ) -> Result<Self, String> {
        let total = super::super::super::validate_pstd_grid_shape(grid)?;
        validate_solver_params(solver)?;
        validate_input_lengths(total, &medium, &pml, &absorption)?;

        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let SolverParams {
            dt,
            nt,
            nonlinear,
            absorbing,
            ..
        } = solver;
        let state = P::build_state(context, grid, medium, solver, pml, absorption)?;

        Ok(Self {
            nx,
            ny,
            nz,
            nt,
            dt,
            state,
            _provider: std::marker::PhantomData,
            nonlinear,
            absorbing,
        })
    }
}

fn validate_solver_params(solver: SolverParams) -> Result<(), String> {
    if !(solver.dt.is_finite() && solver.dt > 0.0 && (solver.dt as f32) > 0.0) {
        return Err(format!(
            "GPU PSTD time step must be finite, positive, and representable as nonzero f32; got {}",
            solver.dt
        ));
    }
    if solver.nt == 0 {
        return Err("GPU PSTD time-step count must be positive".to_owned());
    }
    u32::try_from(solver.nt).map_err(|_| {
        format!(
            "GPU PSTD time-step count {} exceeds u32 shader addressing",
            solver.nt
        )
    })?;
    if !(solver.c_ref.is_finite() && solver.c_ref > 0.0) {
        return Err(format!(
            "GPU PSTD reference sound speed must be finite and positive; got {}",
            solver.c_ref
        ));
    }
    Ok(())
}

fn validate_input_lengths(
    total: usize,
    medium: &MediumArrays<'_>,
    pml: &PmlArrays<'_>,
    absorption: &AbsorptionArrays<'_>,
) -> Result<(), String> {
    for (name, values) in [
        ("sound speed", medium.c0_flat),
        ("density", medium.rho0_flat),
        ("PML x", pml.x),
        ("PML y", pml.y),
        ("PML z", pml.z),
        ("staggered PML x", pml.sgx),
        ("staggered PML y", pml.sgy),
        ("staggered PML z", pml.sgz),
        ("nonlinearity", absorption.bon_a_flat),
        ("absorption nabla1", absorption.nabla1),
        ("absorption nabla2", absorption.nabla2),
        ("absorption tau", absorption.tau),
        ("absorption eta", absorption.eta),
    ] {
        if values.len() != total {
            return Err(format!(
                "GPU PSTD {name} field has {} values; expected {total}",
                values.len()
            ));
        }
    }

    for (name, values) in [
        ("sound speed", medium.c0_flat),
        ("density", medium.rho0_flat),
    ] {
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || *value <= 0.0)
        {
            return Err(format!(
                "GPU PSTD {name} must be finite and positive at flat index {index}; got {value}"
            ));
        }
    }
    Ok(())
}
