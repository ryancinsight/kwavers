//! Forward solvers
//!
//! Solvers for forward problems (wave propagation, heat diffusion, etc.) that
//! simulate physical phenomena from causes to effects.
//!
//! # Canonical solver matrix
//!
//! | Domain                      | Spatial scheme        | Time scheme         | Canonical orchestrator                                                                                            |
//! |-----------------------------|-----------------------|---------------------|-------------------------------------------------------------------------------------------------------------------|
//! | Acoustic fluid (μ = 0)      | Pseudospectral (FFT)  | k-space corrected   | [`pstd::PSTDSolver`]                                                                                              |
//! | Acoustic fluid (μ = 0)      | 4th-order FD          | Leapfrog            | [`crate::forward::fdtd::solver::GenericFdtdSolver`]                                                                                       |
//! | Acoustic fluid (μ = 0)      | Pseudospectral (GPU)  | k-space corrected   | `kwavers_gpu::pstd_gpu::GpuPstdSolver` (consolidated into the kwavers-gpu crate) |
//! | Elastic isotropic (μ ≥ 0)   | Pseudospectral (FFT)  | k-space corrected   | [`pstd::PSTDSolver`] **+ `pstd::extensions::PstdElasticPlugin`** — μ = 0 reduces to baseline acoustic PSTD      |
//! | Elastic isotropic (μ ≥ 0)   | 4th-order FD coll.    | Velocity-Verlet     | [`elastic::swe::ElasticWaveSolver`]                                                                               |
//! | Elastic nonlinear           | 4th-order FD          | RK / IMEX           | [`elastic::nonlinear::NonlinearElasticWaveSolver`]                                                                |
//! | Poroelastic                 | FD                    | Leapfrog            | [`poroelastic`]                                                                                                   |
//! | BEM                         | Boundary integral     | —                   | [`bem::BemSolver`]                                                                                                |
//! | Pennes bioheat              | 2nd / 4th-order FD    | Forward Euler       | [`thermal_diffusion::ThermalDiffusionSolver`]                                                                     |
//! | Optical (RTE / diffusion)   | FD / Monte Carlo      | Stationary          | [`optical`]                                                                                                       |
//!
//! # Architecture: elastic-as-PSTD-plugin
//!
//! Pseudospectral acoustic and pseudospectral elastic share the same FFT
//! kernel; only the stress tensor differs (acoustic = isotropic pressure;
//! elastic = full λ/μ stress). `pstd::extensions::PstdElasticPlugin`
//! exposes the spectral elastic stress and velocity primitives so that
//! [`pstd::PSTDSolver`] can be extended in place rather than duplicated.
//!
//! With `μ ≡ 0` the plugin's `apply_stress_update_in_place` reduces
//! mathematically to the baseline acoustic stress update (the shear pass
//! constant-folds to zero — see the theorem on
//! [`pstd::extensions::elastic`]); the elastic plugin therefore strictly
//! generalises rather than competes with the acoustic path.
//!
//! Prior to consolidation, two pseudospectral wave steppers existed in
//! parallel: [`pstd::PSTDSolver`] (acoustic) and a duplicate
//! `solver::forward::elastic_wave::ElasticWave::update_wave` (an
//! `AcousticWaveModel` impl that hard-coded `μ = 0`). That duplicate plus its
//! `μ = 0` `ElasticWavePlugin` were deleted; the genuinely useful spectral
//! primitives now live under [`pstd::extensions`].
//!
//! The `PhysicsModelType::MechanicalStress` capability has since been
//! **re-introduced genuinely** (ADR 021) as
//! [`pstd::extensions::MechanicalStressPlugin`] — a `Plugin` owning a real
//! [`pstd::extensions::ElasticPstdOrchestrator`] (leapfrog λ/μ stress–velocity
//! PSTD) that provides isotropic pressure `p = -⅓ tr(σ)` to the unified field.
//! This is not the deleted `μ = 0` wrapper: the shear pass is active, so the
//! variant supports shear waves the acoustic path cannot.
//!
//! ## Module organisation
//!
//! Two coexisting layouts (per ADR 005):
//!
//! - **Flat layout** (legacy, stable): `solver::forward::{fdtd, pstd, ...}` — the
//!   original 20 sibling modules. All existing internal call sites consume
//!   these paths directly. They remain the supported import path for now.
//! - **Domain-grouped layout** (Phase 1): `solver::forward::{acoustic_solvers,
//!   elastic_solvers, thermal_solvers, optical_solvers, boundary_element,
//!   hybrid_models, ode_methods, multiphysics_bubble, plugin}` — thin
//!   re-export modules that expose the same items grouped by physics domain.
//!
//! Both layouts resolve to the same items (`solver::forward::fdtd::FdtdSolver`
//! and `solver::forward::acoustic_solvers::fdtd::FdtdSolver` are the same
//! type). New code SHOULD prefer the domain-grouped layout.
//!
//! Phase 2 of ADR 005 will deprecate flat paths; Phase 3 will remove them.
//! Until then, both surfaces are stable and `cargo-semver-checks`-clean.

// ── Flat layout (legacy, stable) ─────────────────────────────────────────────
pub mod acoustic_ivp;
pub mod bem;
pub mod bubble_dynamics;
pub mod coupled;
pub mod elastic;
pub mod fdtd;
pub mod helmholtz;
pub mod hybrid;
pub mod nonlinear;
pub mod ode;
pub mod optical;
pub mod plugin_based;
pub mod poroelastic;
pub mod pstd;
pub mod thermal;
pub mod thermal_diffusion;
pub mod viscoacoustic;

// ── Domain-grouped layout (Phase 1 of ADR 005) ───────────────────────────────
//
// Each group module is a pure re-export shell — zero source files, zero new
// types, zero new behaviour. Items resolve to the same `pub mod` declarations
// above; only the navigation path changes.

/// Acoustic-domain solvers: FDTD, PSTD, k-space, IVP, nonlinear (Westervelt /
/// Kuznetsov / KZK), and Helmholtz formulations.
pub mod acoustic_solvers {
    pub use super::acoustic_ivp;
    pub use super::fdtd;
    pub use super::helmholtz;
    pub use super::nonlinear;
    pub use super::pstd;
}

/// Elastic-wave solvers: linear and nonlinear elastic SWE, poroelastic, and
/// the spectral-element method.
pub mod elastic_solvers {
    pub use super::elastic;
    pub use super::poroelastic;
}

/// Thermal-domain solvers: Pennes bioheat and pure thermal diffusion.
pub mod thermal_solvers {
    pub use super::thermal;
    pub use super::thermal_diffusion;
}

/// Optical-domain solvers: photon transport via diffusion approximation.
pub mod optical_solvers {
    pub use super::optical;
}

/// Boundary-element formulations.
pub mod boundary_element {
    pub use super::bem;
}

/// Hybrid and coupled multi-method solvers (BEM-FEM, FDTD-FEM, PSTD-SEM,
/// thermal-acoustic coupling).
pub mod hybrid_models {
    pub use super::coupled;
    pub use super::hybrid;
}

/// ODE integration methods used as inner steppers (explicit Runge-Kutta).
pub mod ode_methods {
    pub use super::ode;
}

/// Multiphysics bubble dynamics (Rayleigh-Plesset, Keller-Miksis, Gilmore).
pub mod multiphysics_bubble {
    pub use super::bubble_dynamics;
}

/// Plugin-orchestrated solver harness.
pub mod plugin {
    pub use super::plugin_based;
}

// ── Top-level re-exports of canonical solver structs ────────────────────────
// These are the curated public-API entry points. Both flat and grouped paths
// also expose them via their own `pub use`s.
//
// ADR 005 Phase 3 (2026-06-02): the deprecated flat-item re-exports
// (`forward::FdtdSolver`, `forward::PSTDSolver`, …) have been removed. Use the
// module paths — flat (`forward::fdtd::FdtdSolver`) or domain-grouped
// (`forward::acoustic_solvers::fdtd::FdtdSolver`); the equivalence of those two
// module groupings is still guaranteed by `path_equivalence_tests` below.

#[cfg(test)]
mod path_equivalence_tests {
    //! Verify that flat and domain-grouped paths resolve to the same item.
    //!
    //! ADR 005 invariant: `solver::forward::<flat>::T` and
    //! `solver::forward::<group>::<flat>::T` MUST refer to the same `T`.
    //! These tests fail to compile if a re-export drifts.

    #[test]
    fn fdtd_solver_path_equivalence() {
        // If both paths name the same type, this assignment compiles.
        let _: fn(
            crate::forward::fdtd::FdtdSolver,
        ) -> crate::forward::acoustic_solvers::fdtd::FdtdSolver = std::convert::identity;
    }

    #[test]
    fn pstd_solver_path_equivalence() {
        let _: fn(
            crate::forward::pstd::PSTDSolver,
        ) -> crate::forward::acoustic_solvers::pstd::PSTDSolver = std::convert::identity;
    }

    #[test]
    fn bem_solver_path_equivalence() {
        let _: fn(
            crate::forward::bem::BemSolver,
        ) -> crate::forward::boundary_element::bem::BemSolver = std::convert::identity;
    }

    #[test]
    fn pennes_solver_path_equivalence() {
        let _: fn(
            crate::forward::thermal::PennesSolver,
        ) -> crate::forward::thermal_solvers::thermal::PennesSolver = std::convert::identity;
    }

    #[test]
    fn plugin_based_solver_path_equivalence() {
        let _: fn(
            crate::forward::plugin_based::PluginBasedSolver,
        ) -> crate::forward::plugin::plugin_based::PluginBasedSolver = std::convert::identity;
    }

    #[test]
    fn elastic_solvers_module_resolves() {
        // Confirms `elastic_solvers::elastic` and the flat path resolve to the
        // same module by exercising a struct re-exported from each.
        let _: fn(
            crate::forward::elastic::nonlinear::HyperelasticModel,
        )
            -> crate::forward::elastic_solvers::elastic::nonlinear::HyperelasticModel =
            std::convert::identity;
    }

    #[test]
    fn optical_solvers_module_resolves() {
        let _: fn(
            crate::forward::optical::DiffusionSolver,
        ) -> crate::forward::optical_solvers::optical::DiffusionSolver = std::convert::identity;
    }

    #[test]
    fn hybrid_models_module_resolves() {
        // `hybrid::HybridSolver` must be reachable via the grouped path.
        let _: fn(
            crate::forward::hybrid::HybridSolver,
        ) -> crate::forward::hybrid_models::hybrid::HybridSolver = std::convert::identity;
    }

    #[test]
    fn multiphysics_bubble_module_resolves() {
        // The grouped path and the flat path must agree on `BubbleDynamicsPlugin`.
        use crate::forward::bubble_dynamics::plugin::BubbleDynamicsPlugin as Flat;
        use crate::forward::multiphysics_bubble::bubble_dynamics::plugin::BubbleDynamicsPlugin as Grouped;
        let _: fn(Flat) -> Grouped = std::convert::identity;
    }
}
