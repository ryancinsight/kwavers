//! Simulation runner — thin solver dispatch orchestration.
//!
//! The runner delegates to per-solver dispatch modules under
//! [`dispatch`](super::dispatch).  All shared types live in
//! [`types`](super::types).

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_source::Source as KwaversSource;
use crate::dispatch;
use kwavers_solver::config::SolverType;

use crate::types::{SimulationRunRequest, SimulationRunResult};

/// Owns solver dispatch for simulation requests.
#[derive(Debug, Default)]
pub struct SimulationRunner;

impl SimulationRunner {
    /// Dispatch and run a simulation based on the request config.
    ///
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] for unsupported solver/config combos.
    /// - Propagates solver creation and runtime errors.
    pub fn run(
        req: &SimulationRunRequest<'_>,
        sources: Vec<Box<dyn KwaversSource>>,
    ) -> KwaversResult<SimulationRunResult> {
        match req.solver_type {
            // ── Time-domain acoustic ──────────────────────────────────────────
            SolverType::FDTD => dispatch::fdtd::run(req, sources),
            SolverType::PSTD | SolverType::Hybrid => {
                if let Some(thermal) = req.thermal {
                    dispatch::pstd::run_with_thermal(req, sources, thermal)
                } else {
                    dispatch::pstd::run(req, sources)
                }
            }
            #[cfg(feature = "gpu")]
            SolverType::PstdGpu => dispatch::pstd::run_gpu_or_fallback(req, sources),
            #[cfg(not(feature = "gpu"))]
            SolverType::PstdGpu => dispatch::pstd::run(req, sources), // fallback to CPU

            // ── Frequency-domain ─────────────────────────────────────────────
            SolverType::Helmholtz => dispatch::helmholtz::run(req),
            SolverType::BEM => dispatch::bem::run(req),

            // ── Discontinuous Galerkin ────────────────────────────────────────
            SolverType::DG => dispatch::dg::run(req),

            // ── Elastic ──────────────────────────────────────────────────────
            SolverType::Elastic => dispatch::elastic::run(req),
            SolverType::ElasticPSTD => dispatch::elastic_pstd::run(req),

            // ── Nonlinear acoustics ───────────────────────────────────────────
            SolverType::Nonlinear => dispatch::nonlinear::run(req),

            // ── Poroelastic ───────────────────────────────────────────────────
            SolverType::Poroelastic => dispatch::poroelastic::run(req),

            // ── Analytical ────────────────────────────────────────────────────
            SolverType::RayleighSommerfeld => dispatch::rayleigh_sommerfeld::run(req),

            // ── Unsupported — these are plugin/external types ─────────────────
            SolverType::KSpace => Err(KwaversError::InvalidInput(
                "Simulation.run expects SolverType::PSTD with k-space correction; use KSpaceCorrectionMode".into(),
            )),
            SolverType::DiscontinuousGalerkin | SolverType::FEM => Err(KwaversError::InvalidInput(
                "Use SolverType::DG (DiscontinuousGalerkin) or SolverType::Helmholtz (FEM)".into(),
            )),
            SolverType::Auto => Err(KwaversError::InvalidInput(
                "SolverType::Auto is not supported — select a concrete solver type".into(),
            )),
        }
    }
}
