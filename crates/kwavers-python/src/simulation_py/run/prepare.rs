//! Run preparation: the pure conversions between the Python-facing
//! `Simulation` state and the solver's request vocabulary. Each function is
//! a total or typed-fallible map with no side effects, so `execute` reads as
//! the pipeline and these carry the rules.

use super::super::ElasticVelocitySource;
use crate::solver_type_bindings::{FftBackend, SolverType};
use kwavers_solver::forward::pstd::extensions::{ElasticPstdSourceMode, ElasticPstdVelocitySource};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Courant number of the default time step. The stability limit of the
/// explicit schemes on a uniform grid is `dt ≤ dx_min / (c_max · √3)` in three
/// dimensions; the default takes a 0.3 fraction of it.
const DEFAULT_CFL: f64 = 0.3;

/// The time step: the caller's, or the CFL default from the finest grid
/// spacing and the medium's fastest sound speed.
pub(super) fn time_step(dt: Option<f64>, dx_min: f64, c_max: f64) -> f64 {
    dt.unwrap_or_else(|| DEFAULT_CFL * dx_min / (c_max * 3.0_f64.sqrt()))
}

/// The solver the run requests; the binding enumerates solvers the runner
/// cannot execute, and those are a typed refusal here rather than a fallback.
pub(super) fn solver_type(solver: SolverType) -> PyResult<kwavers_solver::config::SolverType> {
    use kwavers_solver::config::SolverType as Runner;
    Ok(match solver {
        SolverType::FDTD => Runner::FDTD,
        SolverType::PSTD => Runner::PSTD,
        SolverType::Hybrid => Runner::Hybrid,
        SolverType::Elastic => Runner::Elastic,
        SolverType::ElasticPSTD => Runner::ElasticPSTD,
        SolverType::Helmholtz => Runner::Helmholtz,
        SolverType::BEM => Runner::BEM,
        SolverType::DG => Runner::DG,
        SolverType::RayleighSommerfeld => Runner::RayleighSommerfeld,
        SolverType::Poroelastic => Runner::Poroelastic,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unsupported solver type: {other:?}"
            )))
        }
    })
}

/// The FFT provider the run requests.
pub(super) fn fft_backend(backend: FftBackend) -> kwavers_solver::config::FftBackend {
    match backend {
        FftBackend::Leto => kwavers_solver::config::FftBackend::Leto,
        FftBackend::Hephaestus => kwavers_solver::config::FftBackend::Hephaestus,
    }
}

/// The elastic PSTD velocity source, with its mode string resolved: any
/// value other than `"dirichlet"` is additive, the scheme's default.
pub(super) fn elastic_velocity_source(
    source: ElasticVelocitySource,
) -> Option<ElasticPstdVelocitySource> {
    source.map(|(mask, ux, uy, uz, mode)| ElasticPstdVelocitySource {
        mask,
        ux,
        uy,
        uz,
        mode: if mode == "dirichlet" {
            ElasticPstdSourceMode::Dirichlet
        } else {
            ElasticPstdSourceMode::Additive
        },
    })
}

/// The initial-value-problem axis index for the elastic solver: `x`, `y`,
/// and everything else `z`.
pub(super) fn ivp_axis(axis: Option<&str>) -> Option<usize> {
    axis.map(|axis| match axis {
        "x" => 0,
        "y" => 1,
        _ => 2,
    })
}
