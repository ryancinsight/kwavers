//! `MechanicalStressPlugin` — elastic stress–velocity propagation as a
//! [`Plugin`](crate::plugin::Plugin) for the capability-driven solver loop.
//!
//! # Why a dedicated plugin (and not the deleted `ElasticWavePlugin`)
//!
//! A prior `ElasticWavePlugin` (and its `PhysicsModelType::MechanicalStress`
//! factory variant) were a `μ ≡ 0` *duplicate* of the acoustic PSTD stepper and
//! were deliberately removed during the elastic-as-PSTD-plugin consolidation
//! (see [`crate::forward`] "Architecture: elastic-as-PSTD-plugin"). This plugin
//! is **not** that wrapper: it owns a genuine
//! [`ElasticPstdOrchestrator`] (leapfrog stress–velocity PSTD with k-space
//! correction and optional PML) and drives real `λ/μ` elastic physics, so the
//! `μ > 0` shear pass is active rather than constant-folded away. ADR 021.
//!
//! # Pipeline coupling
//!
//! The orchestrator is a vector-velocity + rank-2-stress system whose state
//! lives outside the scalar unified field cube. Following the established
//! [`PSTDPlugin`](crate::forward::pstd::plugin) pattern, the plugin holds that
//! state internally and **provides** the isotropic acoustic pressure
//! `p = -⅓ tr(σ)` ([`ElasticPstdOrchestrator::pressure_field`]) into the
//! unified [`UnifiedFieldType::Pressure`] plane each step. It declares no
//! required fields: the elastic state is self-contained and advanced by one
//! genuine leapfrog step per [`Plugin::update`] call, seeded by an initial
//! condition (see [`MechanicalStressPlugin::orchestrator_mut`]) or a configured
//! elastic velocity source. Acoustic-pressure `context.sources` are *not*
//! consumed: an acoustic pressure source is not an elastic velocity source, and
//! conflating them would misrepresent the excitation.

use leto::{
    Array4,
};
use std::any::Any;
use std::fmt::Debug;

use super::elastic_orchestrator::{ElasticPstdMedium, ElasticPstdOrchestrator};
use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;

/// Isotropic-elastic stress–velocity propagation plugin.
///
/// Construct with [`MechanicalStressPlugin::new`] (the global integrator `dt`
/// is captured for the orchestrator's k-space correction). The orchestrator is
/// built lazily in [`Plugin::initialize`] from the medium's Lamé/density
/// fields; [`Plugin::update`] performs exactly one leapfrog step and writes the
/// resulting pressure into the unified field cube.
#[derive(Debug)]
pub struct MechanicalStressPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    dt: f64,
    orchestrator: Option<ElasticPstdOrchestrator>,
}

impl MechanicalStressPlugin {
    /// Create a new elastic-stress plugin for integrator timestep `dt` [s].
    ///
    /// The orchestrator is not built until [`Plugin::initialize`] supplies the
    /// grid and medium.
    #[must_use]
    pub fn new(dt: f64) -> Self {
        Self {
            metadata: PluginMetadata {
                id: "mechanical_stress".to_owned(),
                name: "Mechanical Stress (Elastic PSTD)".to_owned(),
                version: "1.0.0".to_owned(),
                description: "Isotropic elastic stress–velocity propagation (λ/μ), \
                              providing isotropic pressure to the unified field."
                    .to_owned(),
                author: "Kwavers Team".to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            dt,
            orchestrator: None,
        }
    }

    /// Mutable access to the underlying orchestrator once initialized, for
    /// initial-condition seeding (e.g. an isotropic compressional perturbation
    /// or a transverse shear velocity) before stepping. Returns [`None`] before
    /// [`Plugin::initialize`].
    pub fn orchestrator_mut(&mut self) -> Option<&mut ElasticPstdOrchestrator> {
        self.orchestrator.as_mut()
    }

    /// Read access to the underlying orchestrator once initialized.
    #[must_use]
    pub fn orchestrator(&self) -> Option<&ElasticPstdOrchestrator> {
        self.orchestrator.as_ref()
    }
}

impl Plugin for MechanicalStressPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        // Self-contained: elastic state is internal; nothing read from the cube.
        Vec::new()
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let elastic_medium = ElasticPstdMedium {
            lame_lambda: medium.lame_lambda_array().into(),
            lame_mu: medium.lame_mu_array().into(),
            density: medium.density_array().to_contiguous(),
        };
        self.orchestrator = Some(ElasticPstdOrchestrator::new(grid, elastic_medium, self.dt)?);
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        _context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let orchestrator = self.orchestrator.as_mut().ok_or_else(|| {
            KwaversError::InternalError(
                "MechanicalStressPlugin stepped before initialize()".to_owned(),
            )
        })?;

        // One genuine leapfrog stress–velocity step of the elastic field.
        orchestrator.step()?;

        // Provide the isotropic pressure p = -⅓ tr(σ) to the unified cube.
        let pressure = orchestrator.pressure_field();
        let mut pressure_plane = fields.index_axis_mut(0, UnifiedFieldType::Pressure.index());
        let [nx, ny, nz] = pressure.shape();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    pressure_plane[[i, j, k]] = pressure[[i, j, k]];
                }
            }
        }

        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::test_support::{make_context, null_plugin_fields, NullBoundary};
    use kwavers_field::mapping::UnifiedFieldType;
    use kwavers_grid::Grid;
    use kwavers_medium::elastic::lame_from_speeds;
    use kwavers_medium::HomogeneousMedium;
    use leto::{
    Array4,
};

    fn grid() -> Grid {
        Grid::new(24, 24, 1, 1e-3, 1e-3, 1e-3).expect("grid")
    }

    /// Solid medium with genuine shear support (c_s > 0): soft-tissue-like
    /// c_p = 1500, c_s = 80 m/s, ρ = 1000, giving μ = ρc_s² > 0.
    fn solid_medium(grid: &Grid) -> HomogeneousMedium {
        let (lambda, mu) = lame_from_speeds(1500.0, 80.0, 1000.0);
        let mut medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, grid);
        medium
            .set_lame_parameters(lambda, mu)
            .expect("finite non-negative Lamé parameters");
        medium
    }

    #[test]
    fn errors_when_stepped_before_initialize() {
        let g = grid();
        let mut plugin = MechanicalStressPlugin::new(1e-7);
        let mut fields = Array4::<f64>::zeros((16, g.nx, g.ny, g.nz));
        let pf = null_plugin_fields(&g);
        let mut boundary = NullBoundary;
        let mut ctx = make_context(&pf, &mut boundary);
        let medium = solid_medium(&g);
        let err = plugin
            .update(&mut fields, &g, &medium, 1e-7, 0.0, &mut ctx)
            .expect_err("update before initialize must error");
        assert!(
            format!("{err}").contains("before initialize"),
            "error must name the uninitialized cause; got {err}"
        );
    }

    /// A seeded compressional perturbation (σxx=σyy=σzz=−p₀) evolves under real
    /// elastic stepping and the plugin writes the genuine stress-trace pressure
    /// into the unified Pressure plane — value-semantic, and it changes between
    /// steps (the body is not replaceable by a constant / Default).
    #[test]
    fn update_steps_elastic_state_and_writes_genuine_pressure() {
        let g = grid();
        let mut plugin = MechanicalStressPlugin::new(5e-8);
        let medium = solid_medium(&g);
        plugin.initialize(&g, &medium).expect("initialize");

        // Seed a localized isotropic compression at the centre.
        let p0 = 1.0e5;
        {
            let orch = plugin.orchestrator_mut().expect("orchestrator");
            let s = orch.stress_mut();
            let (ci, cj) = (g.nx / 2, g.ny / 2);
            s.txx[[ci, cj, 0]] = -p0;
            s.tyy[[ci, cj, 0]] = -p0;
            s.tzz[[ci, cj, 0]] = -p0;
        }

        let mut fields = Array4::<f64>::zeros((16, g.nx, g.ny, g.nz));
        let pf = null_plugin_fields(&g);
        let mut boundary = NullBoundary;
        let mut ctx = make_context(&pf, &mut boundary);
        let p_idx = UnifiedFieldType::Pressure.index();

        plugin
            .update(&mut fields, &g, &medium, 5e-8, 0.0, &mut ctx)
            .expect("first step");
        let after_one = fields.index_axis(0, p_idx).unwrap().to_owned();
        let energy_one: f64 = after_one.iter().map(|v| v * v).sum();
        assert!(
            energy_one > 0.0,
            "pressure field must be non-trivially populated after one step"
        );

        plugin
            .update(&mut fields, &g, &medium, 5e-8, 5e-8, &mut ctx)
            .expect("second step");
        let after_two = fields.index_axis(0, p_idx).unwrap().to_owned();

        // The wave propagates: the field is different between steps (genuine
        // evolution, not a static write).
        let max_diff = after_one
            .iter()
            .zip(after_two.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-3,
            "pressure must evolve between steps (got max_diff={max_diff})"
        );

        // The written plane equals the orchestrator's stress-trace pressure.
        let direct = plugin
            .orchestrator()
            .expect("orchestrator")
            .pressure_field();
        for (w, d) in after_two.iter().zip(direct.iter()) {
            assert!(
                (w - d).abs() <= 1e-9 * d.abs().max(1.0),
                "unified pressure must equal -⅓tr(σ); {w} vs {d}"
            );
        }
    }

    /// `provided_fields` advertises Pressure and nothing is required — the
    /// plugin composes without demanding fields from upstream.
    #[test]
    fn field_contract_provides_pressure_requires_nothing() {
        let plugin = MechanicalStressPlugin::new(1e-7);
        assert_eq!(plugin.provided_fields(), vec![UnifiedFieldType::Pressure]);
        assert!(plugin.required_fields().is_empty());
    }
}
