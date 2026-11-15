//! Simulation runner — thin solver dispatch orchestration.
//!
//! The runner delegates to per-solver dispatch modules under
//! [`dispatch`]. All shared types live in
//! [`types`](super::types).

use crate::dispatch;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::config::{FftBackend, SolverType};
use kwavers_source::Source as KwaversSource;

use crate::types::{SimulationRunRequest, SimulationRunResult};

/// Owns solver dispatch for simulation requests.
#[derive(Debug, Default)]
pub struct SimulationRunner;

impl SimulationRunner {
    /// Dispatch and run a simulation based on the request config.
    ///
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` for unsupported solver/config combos.
    /// - Returns `KwaversError::FeatureNotAvailable` for `SolverType::Hybrid`
    ///   under every `FftBackend`: the runner has no path that hands a
    ///   request's grid source and sensor recording to `HybridSolver`, and it
    ///   does not answer with PSTD.
    /// - Propagates solver creation and runtime errors.
    pub fn run(
        req: &SimulationRunRequest<'_>,
        sources: Vec<Box<dyn KwaversSource>>,
    ) -> KwaversResult<SimulationRunResult> {
        match req.solver_type {
            // ── Time-domain acoustic ──────────────────────────────────────────
            SolverType::FDTD => dispatch::fdtd::run(req, sources),
            SolverType::PSTD if req.fft_backend == FftBackend::Leto => {
                if let Some(thermal) = req.thermal {
                    dispatch::pstd::run_with_thermal(req, sources, thermal)
                } else {
                    dispatch::pstd::run(req, sources)
                }
            }
            SolverType::Hybrid => Err(KwaversError::FeatureNotAvailable(
                "SolverType::Hybrid is not supported by SimulationRunner: HybridSolver builds \
                 its inner solvers with an empty grid source and keeps them private, so the \
                 request's grid source and sensor recording cannot reach them; select \
                 SolverType::PSTD or SolverType::FDTD"
                    .to_owned(),
            )),
            #[cfg(feature = "gpu")]
            SolverType::PSTD => dispatch::pstd::run_gpu(req, sources),
            #[cfg(not(feature = "gpu"))]
            SolverType::PSTD => Err(KwaversError::FeatureNotAvailable(
                "FftBackend::Hephaestus requires the `gpu` Cargo feature".to_owned(),
            )),

            // ── Frequency-domain ─────────────────────────────────────────────
            SolverType::Helmholtz => dispatch::helmholtz::run(req),
            SolverType::BEM => dispatch::bem::run(req),

            // ── Discontinuous Galerkin ────────────────────────────────────────
            SolverType::DG => dispatch::dg::run(req),

            // ── Elastic ──────────────────────────────────────────────────────
            SolverType::Elastic => dispatch::elastic::run(req),
            SolverType::ElasticPSTD if req.fft_backend == FftBackend::Hephaestus => {
                Err(KwaversError::FeatureNotAvailable(
                    "ElasticPSTD does not yet support FftBackend::Hephaestus".to_owned(),
                ))
            }
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

#[cfg(test)]
mod tests;
