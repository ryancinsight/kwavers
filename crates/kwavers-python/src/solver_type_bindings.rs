//! `SolverType` pyclass — solver backend selection enum.

use pyo3::prelude::*;

/// Solver type selection.
///
/// Mathematical Specifications:
/// - FDTD: Finite Difference Time Domain (2nd/4th/6th order spatial accuracy)
/// - PSTD: Pseudospectral Time Domain (spectral spatial accuracy)
/// - Hybrid: Adaptive switching between FDTD and PSTD
/// - Helmholtz: Frequency-domain Helmholtz FEM solver for monochromatic steady-state fields
/// - BEM: Boundary Element Method solver for exterior/boundary-value problems
/// - DG: Discontinuous Galerkin / Hybrid Spectral-DG solver for high-order shock-capturing
/// - Nonlinear: Westervelt / Kuznetsov / KZK nonlinear acoustic wave solvers
/// - Poroelastic: Biot poroelastic wave solver for fluid-saturated porous media
///
/// References:
/// - Treeby & Cox (2010) for PSTD implementation
/// - Taflove & Hagness (2005) for FDTD fundamentals
/// - Gilmore (1952) for high-amplitude bubble dynamics
/// - Ihlenburg (1998) "Finite Element Analysis of Acoustic Scattering" for Helmholtz FEM
/// - Brebbia & Dominguez (1992) "Boundary Elements" for BEM
/// - Hesthaven & Warburton (2007) "Nodal Discontinuous Galerkin Methods" for DG
#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverType {
    /// Finite Difference Time Domain solver
    FDTD,
    /// Pseudospectral Time Domain solver
    PSTD,
    /// Hybrid FDTD/PSTD solver
    Hybrid,
    /// GPU-resident Pseudospectral Time Domain solver (requires `gpu` feature).
    /// Falls back to CPU PSTD if no GPU adapter is available.
    PstdGpu,
    /// Elastic-wave solver (4th-order FD with velocity-Verlet integration,
    /// PML boundary, supports compressional + shear waves). The Python-level
    /// equivalent of k-Wave's `pstdElastic2D` / `pstdElastic3D`.
    ///
    /// Phase A.2 of ADR 007 — current capability: initial-displacement IVP
    /// only (single component, configurable axis); records the chosen
    /// displacement component at sensor mask positions. Stress / velocity
    /// source masks and multi-component recording land in Phases A.3 and
    /// A.2.5 respectively.
    Elastic,
    /// Pseudospectral elastic solver — drives the canonical PSTD step loop
    /// with the `pstd::extensions::PstdElasticPlugin` for full elastic
    /// (μ ≥ 0) propagation. With μ = 0 reduces exactly to baseline acoustic
    /// PSTD per the plugin's acoustic-fluid-limit theorem.
    ///
    /// Currently velocity-source + sensor-mask only; no PML yet (short-
    /// propagation diagnostics + cross-engine parity validation against
    /// KWave.jl's `pstd_elastic_2d`). Adds boundary absorption in a follow-
    /// up step. See `kwavers_solver::forward::pstd::extensions` and the
    /// canonical solver matrix in `solver::forward` module docs.
    ElasticPSTD,
    /// Frequency-domain Helmholtz FEM solver for steady-state monochromatic fields.
    /// Supports FEM preconditioner types (Jacobi, ILU, AMG) and mesh-based domains.
    /// Canonical path: `kwavers_solver::forward::helmholtz::FemHelmholtzSolver`.
    Helmholtz,
    /// Boundary Element Method solver for exterior scattering and radiation problems.
    /// Canonical path: `kwavers_solver::forward::bem::BemSolver`.
    BEM,
    /// Discontinuous Galerkin / Hybrid Spectral-DG solver for high-order accuracy
    /// with shock-capturing in heterogeneous media. Canonical path:
    /// `kwavers_solver::forward::pstd::dg::HybridSpectralDGSolver`.
    DG,
    /// Nonlinear acoustic wave solvers supporting Westervelt, Kuznetsov, and KZK
    /// formulations for finite-amplitude propagation. Canonical path:
    /// `kwavers_solver::forward::nonlinear`.
    Nonlinear,
    /// Biot poroelastic wave solver for fluid-saturated porous elastic media.
    /// Canonical path: `kwavers_solver::forward::poroelastic`.
    Poroelastic,
    /// Rayleigh-Sommerfeld angular-spectrum solver for transducer field computation.
    /// Uses the Fast Nearfield Method (FNM) to compute pressure fields from
    /// rectangular transducer velocity distributions via the angular spectrum.
    /// Canonical path: `kwavers_solver::analytical::transducer::FastNearfieldSolver`.
    RayleighSommerfeld,
}

#[pymethods]
impl SolverType {
    /// String representation.
    fn __repr__(&self) -> String {
        match self {
            SolverType::FDTD => "SolverType.FDTD".to_string(),
            SolverType::PSTD => "SolverType.PSTD".to_string(),
            SolverType::Hybrid => "SolverType.Hybrid".to_string(),
            SolverType::PstdGpu => "SolverType.PstdGpu".to_string(),
            SolverType::Elastic => "SolverType.Elastic".to_string(),
            SolverType::ElasticPSTD => "SolverType.ElasticPSTD".to_string(),
            SolverType::Helmholtz => "SolverType.Helmholtz".to_string(),
            SolverType::BEM => "SolverType.BEM".to_string(),
            SolverType::DG => "SolverType.DG".to_string(),
            SolverType::Nonlinear => "SolverType.Nonlinear".to_string(),
            SolverType::Poroelastic => "SolverType.Poroelastic".to_string(),
            SolverType::RayleighSommerfeld => "SolverType.RayleighSommerfeld".to_string(),
        }
    }

    /// Human-readable string.
    fn __str__(&self) -> String {
        match self {
            SolverType::FDTD => "FDTD".to_string(),
            SolverType::PSTD => "PSTD".to_string(),
            SolverType::Hybrid => "Hybrid".to_string(),
            SolverType::PstdGpu => "PstdGpu".to_string(),
            SolverType::Elastic => "Elastic".to_string(),
            SolverType::ElasticPSTD => "ElasticPSTD".to_string(),
            SolverType::Helmholtz => "Helmholtz".to_string(),
            SolverType::BEM => "BEM".to_string(),
            SolverType::DG => "DG".to_string(),
            SolverType::Nonlinear => "Nonlinear".to_string(),
            SolverType::Poroelastic => "Poroelastic".to_string(),
            SolverType::RayleighSommerfeld => "RayleighSommerfeld".to_string(),
        }
    }

    /// Equality comparison.
    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}