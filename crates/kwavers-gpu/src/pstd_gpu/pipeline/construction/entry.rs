//! `GpuPstdSolver::new`: input validation ahead of provider state construction.

use super::super::super::GpuPstdSolver;
use super::super::{AbsorptionArrays, MediumArrays, PmlArrays, SolverParams};
use crate::pstd_gpu::state::PstdStateBuilder;
use kwavers_core::error::KwaversError;
use kwavers_grid::Grid;

/// Why solver construction refused a request.
pub(in crate::pstd_gpu::pipeline) enum ConstructionError {
    /// An input failed validation, or the provider could not build its state.
    Rejected(String),
    /// The capacity check of the provider refused the plan; its
    /// error is kept whole so a caller can match on its variant.
    Capacity(KwaversError),
}

impl From<String> for ConstructionError {
    fn from(message: String) -> Self {
        Self::Rejected(message)
    }
}

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
    /// Returns `Err` if any input array shape is inconsistent with `grid`, if
    /// the capacity check of the provider refuses the plan (the message then
    /// starts with "Resource limit exceeded"), or if it cannot build its
    /// execution state.
    pub fn new(
        context: P::Context,
        grid: &Grid,
        medium: MediumArrays<'_>,
        solver: SolverParams,
        pml: PmlArrays<'_>,
        absorption: AbsorptionArrays<'_>,
    ) -> Result<Self, String> {
        Self::construct(context, grid, medium, solver, pml, absorption).map_err(|error| match error
        {
            ConstructionError::Rejected(message) => message,
            ConstructionError::Capacity(error) => error.to_string(),
        })
    }

    /// Validate the inputs, run the capacity check of the provider, then
    /// build the provider state.
    pub(in crate::pstd_gpu::pipeline) fn construct(
        context: P::Context,
        grid: &Grid,
        medium: MediumArrays<'_>,
        solver: SolverParams,
        pml: PmlArrays<'_>,
        absorption: AbsorptionArrays<'_>,
    ) -> Result<Self, ConstructionError> {
        let total = super::super::super::validate_pstd_grid_shape(grid)?;
        validate_solver_params(solver)?;
        validate_input_lengths(total, &medium, &pml, &absorption)?;
        P::check_plan_capacity(&context, grid).map_err(ConstructionError::Capacity)?;

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
