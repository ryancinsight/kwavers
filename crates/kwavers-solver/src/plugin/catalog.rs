//! Physics capability catalog
//!
//! Maps a strongly-typed [`PhysicsConfig`] (a list of enabled physics
//! capabilities) into a populated [`PluginManager`] ready for the solver
//! loop. Each [`PhysicsModelType`] variant resolves to the concrete plugin
//! constructor for that physics domain.
//!
//! ## Design contract
//!
//! - `PhysicsConfig` is the SSOT for "which physics is on". The catalog is the
//!   only sanctioned constructor path that translates that intent into runnable
//!   plugins; users registering plugins manually via [`PluginManager::add_plugin`]
//!   bypass the catalog by design (DIP escape hatch for advanced use).
//! - The catalog needs a build context (`grid`, `medium`, `dt`) because some
//!   plugin constructors require it. Capabilities whose constructors do not
//!   need one of these inputs simply ignore it.
//! - Variants whose plugin adapter is not yet implemented return a structured
//!   [`ConfigError::InvalidValue`] naming the variant and pointing at the
//!   intended workaround. No silent fallback, no placeholder plugins.
//!
//! ## Capability coverage
//!
//! | Variant                                  | Concrete plugin                      |
//! |------------------------------------------|--------------------------------------|
//! | `LinearAcoustics { FDTD }`               | [`FdtdPlugin`]                       |
//! | `LinearAcoustics { PSTD }`               | [`PSTDPlugin`]                       |
//! | `LinearAcoustics { DG }`                 | unsupported via plugin path; use     |
//! |                                          | [`HybridSpectralDGSolver`] directly  |
//! | `NonlinearAcoustics { KZK }`             | [`KzkPlugin`]                        |
//! | `NonlinearAcoustics { Westervelt }`      | [`WesterveltSolverPlugin`]          |
//! | `NonlinearAcoustics { WesterveltFdtd }`  | [`WesterveltFdtdPlugin`]            |
//! | `NonlinearAcoustics { HybridAngularSpectrum }` | [`HybridAngularSpectrumPlugin`] |
//! | `NonlinearAcoustics { Kuznetsov }`       | [`KuznetsovSolverPlugin`]           |
//! | `ThermalDiffusion`                       | [`ThermalDiffusionPlugin`]           |
//! | `BubbleDynamics { KellerMiksis }`       | `BubbleDynamicsPlugin` (adaptive KM ODE) |
//! | `BubbleDynamics { KellerHerring }`      | `BubbleDynamicsPlugin` (KM wrapper path) |
//! | `BubbleDynamics { RayleighPlesset }`   | `BubbleDynamicsPlugin` (KM, compressibility off) |
//! | `BubbleDynamics { Gilmore }`           | `BubbleDynamicsPlugin` (Gilmore/Tait RK4) |
//! | `OpticalPropagation`                     | [`OpticalDiffusionPlugin`]           |
//! | `FractionalViscoacoustic`                | [`FractionalViscoacousticPlugin`]    |
//! | `MechanicalStress { Isotropic }`         | [`MechanicalStressPlugin`] (elastic PSTD, λ/μ) |
//! | `MechanicalStress { Vti }`               | [`VtiElasticPlugin`]                 |
//! | `PhotoacousticTimeReversal { KZK }`      | [`KzkPlugin`] (backward mode)        |
//! | `PhotoacousticTimeReversal { WesterveltFdtd }` | [`WesterveltFdtdPlugin`] (backward mode) |
//!
//! [`HybridSpectralDGSolver`]: crate::pstd::dg::HybridSpectralDGSolver
//! `BubbleDynamicsPlugin`: crate::forward::bubble_dynamics::plugin::BubbleDynamicsPlugin
//! [`MechanicalStressPlugin`]: crate::forward::pstd::extensions::elastic_plugin::MechanicalStressPlugin

use crate::fdtd::FdtdConfig;
use crate::forward::bubble_dynamics::plugin::{BubbleDynamicsConfig, BubbleDynamicsPlugin};
use crate::forward::fdtd::plugin::FdtdPlugin;
use crate::forward::nonlinear::hybrid_angular_spectrum_plugin::HybridAngularSpectrumPlugin;
use crate::forward::nonlinear::kuznetsov_solver_plugin::KuznetsovSolverPlugin;
use crate::forward::nonlinear::kzk::{KzkPlugin, PropagationDirection as KzkPropagationDirection};
use crate::forward::nonlinear::westervelt::PropagationDirection as WesterveltPropagationDirection;
use crate::forward::nonlinear::westervelt_fdtd_plugin::WesterveltFdtdPlugin;
use crate::forward::nonlinear::westervelt_solver_plugin::WesterveltSolverPlugin;
use crate::forward::optical::OpticalDiffusionPlugin;
use crate::forward::pstd::extensions::{MechanicalStressPlugin, VtiElasticPlugin};
use crate::forward::pstd::plugin::PSTDPlugin;
use crate::forward::thermal_diffusion::plugin::ThermalDiffusionPlugin;
use crate::forward::viscoacoustic::FractionalViscoacousticPlugin;
use crate::plugin::Plugin;
use crate::plugin::PluginManager;
use crate::pstd::PSTDConfig;
use aequitas::systems::si::quantities::Time;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_medium::absorption::FractionalLaplacian;
use kwavers_medium::Medium;
use kwavers_physics::factory::config::PhysicsConfig;
use kwavers_physics::factory::models::{
    AcousticSolver, ElasticWaveKind, NonlinearEquation, PhysicsModelType,
};
use kwavers_physics::thermal::diffusion::ThermalDiffusionConfig;

/// Capability-driven plugin catalog.
///
/// The catalog consumes a [`PhysicsConfig`] describing which physics
/// capabilities are enabled and produces a [`PluginManager`] populated with
/// the corresponding concrete plugins. See module documentation for the
/// variant-to-plugin map.
#[derive(Debug, Default)]
pub struct PhysicsCatalog;

impl PhysicsCatalog {
    /// Build a fully wired plugin manager from a validated config.
    ///
    /// `dt` is the global integrator timestep; it is forwarded to plugins
    /// whose construction requires it.
    /// `medium` is borrowed only during construction to extract material
    /// constants; the plugins do not retain the reference.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn build(
        config: &PhysicsConfig,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<PluginManager> {
        config.validate()?;

        let mut manager = PluginManager::new();
        for (idx, model) in config.models.iter().enumerate() {
            if !model.enabled {
                continue;
            }
            let plugin = Self::build_plugin(idx, &model.model_type, grid, medium, dt)?;
            manager.add_plugin(plugin)?;
        }
        Ok(manager)
    }

    /// Translate a single capability variant into its concrete plugin.
    ///
    /// `idx` is the originating index in `PhysicsConfig::models`, used purely
    /// for diagnostic error messages so the caller can locate the offending
    /// entry without re-walking the config.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    fn build_plugin(
        idx: usize,
        kind: &PhysicsModelType,
        grid: &Grid,
        _medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<Box<dyn Plugin>> {
        match kind {
            PhysicsModelType::LinearAcoustics { solver_type, .. } => match solver_type {
                AcousticSolver::FDTD { order } => Ok(Box::new(FdtdPlugin::new(
                    FdtdConfig {
                        spatial_order: usize::from(*order),
                        ..FdtdConfig::default()
                    },
                    grid,
                )?)),
                AcousticSolver::PSTD { .. } => {
                    Ok(Box::new(PSTDPlugin::new(PSTDConfig::default(), grid)?))
                }
                AcousticSolver::DG { polynomial_order } => Err(unsupported(
                    idx,
                    "LinearAcoustics{DG}",
                    &format!(
                        "DG is not exposed via the plugin path; instantiate \
                         crate::forward::pstd::dg::HybridSpectralDGSolver directly \
                         (for example with HybridSpectralDGConfig {{ dg_polynomial_order: {}, ..Default::default() }}).",
                        polynomial_order
                    ),
                )),
            },
            PhysicsModelType::NonlinearAcoustics { equation_type, .. } => match equation_type {
                NonlinearEquation::KZK => Ok(Box::new(KzkPlugin::new())),
                NonlinearEquation::Westervelt => Ok(Box::new(WesterveltSolverPlugin::new())),
                NonlinearEquation::WesterveltFdtd => Ok(Box::new(WesterveltFdtdPlugin::new())),
                NonlinearEquation::HybridAngularSpectrum => {
                    Ok(Box::new(HybridAngularSpectrumPlugin::new()))
                }
                NonlinearEquation::Kuznetsov => Ok(Box::new(KuznetsovSolverPlugin::new())),
            },
            PhysicsModelType::ThermalDiffusion { .. } => Ok(Box::new(ThermalDiffusionPlugin::new(
                ThermalDiffusionConfig::default(),
            ))),
            PhysicsModelType::FractionalViscoacoustic { reference_loss, reference_frequency, exponent } => Ok(Box::new(
                FractionalViscoacousticPlugin::new(
                    FractionalLaplacian::new(*reference_loss,*reference_frequency,*exponent)?,
                    Time::from_base(dt),
                ),
            )),
            PhysicsModelType::BubbleDynamics { model, nucleation } => {
                let config = BubbleDynamicsConfig {
                    model: model.clone(),
                    nucleation: *nucleation,
                    params: kwavers_physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters::default(),
                };
                Ok(Box::new(BubbleDynamicsPlugin::new(config)))
            }
            PhysicsModelType::OpticalPropagation { .. } => Ok(Box::new(OpticalDiffusionPlugin::new())),
            PhysicsModelType::MechanicalStress { wave_kind } => match wave_kind {
                // Isotropic elastic stress–velocity propagation; the orchestrator
                // reads λ/μ/ρ from the medium in `Plugin::initialize`. ADR 021.
                ElasticWaveKind::Isotropic => Ok(Box::new(MechanicalStressPlugin::new(dt))),
                ElasticWaveKind::Vti {
                    c11,
                    c13,
                    c33,
                    c44,
                    c66,
                } => Ok(Box::new(VtiElasticPlugin::new(
                    dt, *c11, *c13, *c33, *c44, *c66,
                ))),
            },
            PhysicsModelType::PhotoacousticTimeReversal { equation } => match equation {
                NonlinearEquation::KZK => Ok(Box::new(KzkPlugin::with_propagation_direction(
                    KzkPropagationDirection::Backward,
                ))),
                NonlinearEquation::WesterveltFdtd => Ok(Box::new(
                    WesterveltFdtdPlugin::with_propagation_direction(
                        WesterveltPropagationDirection::Backward,
                    ),
                )),
                other => Err(unsupported(
                    idx,
                    "PhotoacousticTimeReversal",
                    &format!(
                        "equation {:?} has no backward-time plugin path; use KZK or WesterveltFdtd.",
                        other
                    ),
                )),
            },
        }
    }
}

fn unsupported(idx: usize, variant: &str, hint: &str) -> KwaversError {
    ConfigError::InvalidValue {
        parameter: format!("models[{idx}].model_type"),
        value: variant.to_owned(),
        constraint: format!("PhysicsCatalog cannot build a plugin for this variant. {hint}"),
    }
    .into()
}

#[cfg(test)]
mod tests;
