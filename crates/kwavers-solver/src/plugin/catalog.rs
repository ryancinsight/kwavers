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
//! | `NonlinearAcoustics { KZK }`             | [`KzkSolverPlugin`]                  |
//! | `NonlinearAcoustics { Westervelt }`      | [`WesterveltSolverPlugin`]          |
//! | `NonlinearAcoustics { Kuznetsov }`       | [`KuznetsovSolverPlugin`]           |
//! | `ThermalDiffusion`                       | [`ThermalDiffusionPlugin`]           |
//! | `BubbleDynamics { KellerMiksis }`       | `BubbleDynamicsPlugin` (adaptive KM ODE) |
//! | `BubbleDynamics { KellerHerring }`      | `BubbleDynamicsPlugin` (KM wrapper path) |
//! | `BubbleDynamics { RayleighPlesset }`   | `BubbleDynamicsPlugin` (KM, compressibility off) |
//! | `BubbleDynamics { Gilmore }`           | `BubbleDynamicsPlugin` (Gilmore/Tait RK4) |
//! | `OpticalPropagation`                     | not yet wired                        |
//! | `MechanicalStress { Isotropic }`         | [`MechanicalStressPlugin`] (elastic PSTD, λ/μ) |
//!
//! [`HybridSpectralDGSolver`]: crate::pstd::dg::HybridSpectralDGSolver
//! `BubbleDynamicsPlugin`: crate::forward::bubble_dynamics::plugin::BubbleDynamicsPlugin
//! [`MechanicalStressPlugin`]: crate::forward::pstd::extensions::elastic_plugin::MechanicalStressPlugin

use crate::fdtd::FdtdConfig;
use crate::forward::bubble_dynamics::plugin::{BubbleDynamicsConfig, BubbleDynamicsPlugin};
use crate::forward::fdtd::plugin::FdtdPlugin;
use crate::forward::nonlinear::kuznetsov_solver_plugin::KuznetsovSolverPlugin;
use crate::forward::nonlinear::kzk_solver_plugin::KzkSolverPlugin;
use crate::forward::nonlinear::westervelt_solver_plugin::WesterveltSolverPlugin;
use crate::forward::pstd::extensions::MechanicalStressPlugin;
use crate::forward::pstd::plugin::PSTDPlugin;
use crate::forward::thermal_diffusion::plugin::ThermalDiffusionPlugin;
use crate::plugin::Plugin;
use crate::plugin::PluginManager;
use crate::pstd::PSTDConfig;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use kwavers_grid::Grid;
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
                AcousticSolver::FDTD { .. } => {
                    Ok(Box::new(FdtdPlugin::new(FdtdConfig::default(), grid)?))
                }
                AcousticSolver::PSTD { .. } => {
                    Ok(Box::new(PSTDPlugin::new(PSTDConfig::default(), grid)?))
                }
                AcousticSolver::DG { .. } => Err(unsupported(
                    idx,
                    "LinearAcoustics{DG}",
                    "DG is not exposed via the plugin path; \
                     instantiate HybridSpectralDGSolver directly.",
                )),
            },
            PhysicsModelType::NonlinearAcoustics { equation_type, .. } => match equation_type {
                NonlinearEquation::KZK => Ok(Box::new(KzkSolverPlugin::new())),
                NonlinearEquation::Westervelt => Ok(Box::new(WesterveltSolverPlugin::new())),
                NonlinearEquation::Kuznetsov => Ok(Box::new(KuznetsovSolverPlugin::new())),
            },
            PhysicsModelType::ThermalDiffusion { .. } => Ok(Box::new(ThermalDiffusionPlugin::new(
                ThermalDiffusionConfig::default(),
            ))),
            PhysicsModelType::BubbleDynamics { model, nucleation } => {
                let config = BubbleDynamicsConfig {
                    model: model.clone(),
                    nucleation: *nucleation,
                    params: kwavers_physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters::default(),
                };
                Ok(Box::new(BubbleDynamicsPlugin::new(config)))
            }
            PhysicsModelType::OpticalPropagation { .. } => Err(unsupported(
                idx,
                "OpticalPropagation",
                "no Plugin adapter yet; use physics::optics models directly.",
            )),
            PhysicsModelType::MechanicalStress { wave_kind } => match wave_kind {
                // Isotropic elastic stress–velocity propagation; the orchestrator
                // reads λ/μ/ρ from the medium in `Plugin::initialize`. ADR 021.
                ElasticWaveKind::Isotropic => Ok(Box::new(MechanicalStressPlugin::new(dt))),
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
mod tests {
    use super::*;
    use kwavers_core::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
    use kwavers_medium::HomogeneousMedium;
    use kwavers_physics::factory::models::{PhysicsBoundaryCondition, PhysicsModelConfig};

    fn small_grid() -> Grid {
        Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("grid")
    }

    fn water(grid: &Grid) -> HomogeneousMedium {
        HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, grid)
    }

    #[test]
    fn empty_config_rejected_by_validate() {
        let mut config = PhysicsConfig::new();
        config.models.clear();
        let grid = small_grid();
        let medium = water(&grid);
        let result = PhysicsCatalog::build(&config, &grid, &medium, 1e-7);
        assert!(result.is_err(), "empty model list must fail validate()");
    }

    #[test]
    fn linear_pstd_capability_builds_one_plugin() {
        let mut config = PhysicsConfig::new();
        config.models.clear();
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::LinearAcoustics {
                solver_type: AcousticSolver::PSTD {
                    spectral_accuracy: true,
                },
                boundary_conditions: PhysicsBoundaryCondition::Absorbing { pml_layers: 10 },
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });

        let grid = small_grid();
        let medium = water(&grid);
        let manager = PhysicsCatalog::build(&config, &grid, &medium, 1e-7)
            .expect("PSTD capability should build");
        assert_eq!(
            manager.plugin_count(),
            1,
            "exactly one plugin should be registered for one enabled capability"
        );
    }

    #[test]
    fn disabled_capability_is_skipped() {
        let mut config = PhysicsConfig::new();
        config.models.clear();
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::LinearAcoustics {
                solver_type: AcousticSolver::FDTD { order: 2 },
                boundary_conditions: PhysicsBoundaryCondition::Absorbing { pml_layers: 8 },
            },
            enabled: false,
            parameters: std::collections::HashMap::new(),
        });
        // Add an enabled one alongside so validate() passes (non-empty).
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::ThermalDiffusion {
                bioheat: false,
                perfusion: false,
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });

        let grid = small_grid();
        let medium = water(&grid);
        let manager = PhysicsCatalog::build(&config, &grid, &medium, 1e-7)
            .expect("config with one enabled capability should build");
        assert_eq!(
            manager.plugin_count(),
            1,
            "disabled capability must be skipped"
        );
    }

    #[test]
    fn unsupported_variant_returns_structured_error() {
        let mut config = PhysicsConfig::new();
        config.models.clear();
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::OpticalPropagation {
                scattering: false,
                anisotropy: 0.0,
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });

        let grid = small_grid();
        let medium = water(&grid);
        let err = PhysicsCatalog::build(&config, &grid, &medium, 1e-7)
            .expect_err("unsupported capability must error");

        let msg = format!("{err}");
        assert!(
            msg.contains("OpticalPropagation"),
            "error must name the variant; got: {msg}"
        );
        assert!(
            msg.contains("models[0]"),
            "error must point at the offending index; got: {msg}"
        );
    }

    #[test]
    fn bubble_dynamics_km_builds_one_plugin() {
        use kwavers_physics::factory::models::BubbleModel;

        let mut config = PhysicsConfig::new();
        config.models.clear();
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::BubbleDynamics {
                model: BubbleModel::KellerMiksis,
                nucleation: false,
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });

        let grid = small_grid();
        let medium = water(&grid);
        let manager = PhysicsCatalog::build(&config, &grid, &medium, 1e-7)
            .expect("BubbleDynamics{KellerMiksis} must build a plugin");
        assert_eq!(
            manager.plugin_count(),
            1,
            "exactly one plugin registered for BubbleDynamics capability"
        );
    }

    #[test]
    fn bubble_dynamics_kh_builds_one_plugin() {
        use kwavers_physics::factory::models::BubbleModel;

        let mut config = PhysicsConfig::new();
        config.models.clear();
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::BubbleDynamics {
                model: BubbleModel::KellerHerring,
                nucleation: false,
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });

        let grid = small_grid();
        let medium = water(&grid);
        let manager = PhysicsCatalog::build(&config, &grid, &medium, 1e-7)
            .expect("BubbleDynamics{KellerHerring} must build a plugin");
        assert_eq!(
            manager.plugin_count(),
            1,
            "exactly one plugin registered for BubbleDynamics{{KellerHerring}}"
        );
    }

    #[test]
    fn bubble_dynamics_gilmore_builds_one_plugin() {
        use kwavers_physics::factory::models::BubbleModel;

        let mut config = PhysicsConfig::new();
        config.models.clear();
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::BubbleDynamics {
                model: BubbleModel::Gilmore,
                nucleation: false,
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });

        let grid = small_grid();
        let medium = water(&grid);
        let manager = PhysicsCatalog::build(&config, &grid, &medium, 1e-7)
            .expect("BubbleDynamics{Gilmore} must build a plugin");
        assert_eq!(
            manager.plugin_count(),
            1,
            "exactly one plugin registered for BubbleDynamics{{Gilmore}}"
        );
    }

    #[test]
    fn mechanical_stress_capability_builds_one_plugin() {
        use kwavers_physics::factory::models::ElasticWaveKind;

        let mut config = PhysicsConfig::new();
        config.models.clear();
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::MechanicalStress {
                wave_kind: ElasticWaveKind::Isotropic,
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });

        let grid = small_grid();
        let medium = water(&grid);
        let manager = PhysicsCatalog::build(&config, &grid, &medium, 1e-7)
            .expect("MechanicalStress{Isotropic} must build a plugin");
        assert_eq!(
            manager.plugin_count(),
            1,
            "exactly one plugin registered for MechanicalStress capability"
        );
    }

    #[test]
    fn mechanical_stress_and_bubble_compose_without_field_conflict() {
        // MechanicalStress provides Pressure; BubbleDynamics provides bubble
        // fields — non-overlapping, so the scheduler resolves a valid order.
        use kwavers_physics::factory::models::{BubbleModel, ElasticWaveKind};

        let mut config = PhysicsConfig::new();
        config.models.clear();
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::MechanicalStress {
                wave_kind: ElasticWaveKind::Isotropic,
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::BubbleDynamics {
                model: BubbleModel::KellerMiksis,
                nucleation: false,
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });

        let grid = small_grid();
        let medium = water(&grid);
        let manager = PhysicsCatalog::build(&config, &grid, &medium, 1e-7)
            .expect("MechanicalStress + BubbleDynamics must build and resolve order");
        assert_eq!(
            manager.plugin_count(),
            2,
            "two plugins expected for MechanicalStress + BubbleDynamics config"
        );
    }

    #[test]
    fn bubble_dynamics_and_pstd_compose_correctly() {
        // Theorem 22.2 (scheduling soundness): two plugins with non-overlapping
        // provided_fields must produce a valid topological order.
        // PSTD provides Pressure; BubbleDynamics provides BubbleRadius + BubbleVelocity.
        use kwavers_physics::factory::models::BubbleModel;

        let mut config = PhysicsConfig::new();
        config.models.clear();
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::LinearAcoustics {
                solver_type: AcousticSolver::PSTD {
                    spectral_accuracy: true,
                },
                boundary_conditions: PhysicsBoundaryCondition::Absorbing { pml_layers: 10 },
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::BubbleDynamics {
                model: BubbleModel::KellerMiksis,
                nucleation: false,
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        });

        let grid = small_grid();
        let medium = water(&grid);
        let manager = PhysicsCatalog::build(&config, &grid, &medium, 1e-7)
            .expect("PSTD + BubbleDynamics must build and resolve dependency order");
        assert_eq!(
            manager.plugin_count(),
            2,
            "two plugins expected for PSTD + BubbleDynamics config"
        );
    }

    fn nonlinear_config(equation: NonlinearEquation) -> PhysicsConfig {
        // Nonlinear models require a non-empty parameter map (config validator).
        let mut parameters = std::collections::HashMap::new();
        parameters.insert("source_frequency".to_owned(), 1.0e6);
        let mut config = PhysicsConfig::new();
        config.models.clear();
        config.models.push(PhysicsModelConfig {
            model_type: PhysicsModelType::NonlinearAcoustics {
                equation_type: equation,
                harmonics: 4,
            },
            enabled: true,
            parameters,
        });
        config
    }

    #[test]
    fn westervelt_variant_builds_one_plugin() {
        let grid = small_grid();
        let medium = water(&grid);
        let manager = PhysicsCatalog::build(
            &nonlinear_config(NonlinearEquation::Westervelt),
            &grid,
            &medium,
            1e-7,
        )
        .expect("Westervelt capability should build (plugin path now wired)");
        assert_eq!(manager.plugin_count(), 1, "one Westervelt plugin expected");
    }

    #[test]
    fn kuznetsov_variant_builds_one_plugin() {
        let grid = small_grid();
        let medium = water(&grid);
        let manager = PhysicsCatalog::build(
            &nonlinear_config(NonlinearEquation::Kuznetsov),
            &grid,
            &medium,
            1e-7,
        )
        .expect("Kuznetsov capability should build (plugin path now wired)");
        assert_eq!(manager.plugin_count(), 1, "one Kuznetsov plugin expected");
    }

    /// The wired plugins must drive the REAL solver step: from a centred pressure
    /// pulse in lossless water, a few steps evolve the field to a finite, bounded,
    /// strictly-changed state (not a no-op, input-sensitive). Stability budget:
    /// CFL = c·dt/dx = 1500·5e-8/1e-3 = 0.075.
    fn assert_plugin_evolves_field(mut plugin: Box<dyn Plugin>) {
        use crate::plugin::test_support::{make_context, null_plugin_fields, NullBoundary};
        use leto::Array4;

        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).expect("grid");
        let medium = water(&grid);
        plugin.initialize(&grid, &medium).expect("initialize");

        // One field; PRESSURE_IDX = 0. Centre pulse of 100 kPa.
        let mut fields = Array4::<f64>::zeros((1, grid.nx, grid.ny, grid.nz));
        fields[[0, 8, 8, 8]] = 1.0e5;
        let before = fields.clone();

        let extra = null_plugin_fields(&grid);
        let mut boundary = NullBoundary;
        let mut ctx = make_context(&extra, &mut boundary);

        let dt = 5.0e-8;
        for step in 0..5 {
            plugin
                .update(&mut fields, &grid, &medium, dt, step as f64 * dt, &mut ctx)
                .expect("plugin update must drive the solver");
        }

        assert!(
            fields.iter().all(|v| v.is_finite()),
            "field must remain finite (stable step)"
        );
        let max_change = fields
            .iter()
            .zip(before.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_change > 1.0,
            "plugin must evolve the field (real computation); max change = {max_change} Pa"
        );
    }

    #[test]
    fn westervelt_plugin_evolves_real_field() {
        assert_plugin_evolves_field(Box::new(
            crate::forward::nonlinear::WesterveltSolverPlugin::new(),
        ));
    }

    #[test]
    fn kuznetsov_plugin_evolves_real_field() {
        assert_plugin_evolves_field(Box::new(
            crate::forward::nonlinear::KuznetsovSolverPlugin::new(),
        ));
    }
}
