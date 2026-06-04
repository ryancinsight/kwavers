use super::engine::BubbleEngine;
use super::*;
use kwavers_core::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use crate::plugin::test_support::{make_context, null_plugin_fields, NullBoundary};
use crate::plugin::{Plugin, PluginState};
use kwavers_physics::acoustics::bubble_dynamics::BubbleParameters;
use kwavers_physics::factory::models::BubbleModel;
use ndarray::{Array4, Axis};

// ── Helpers ───────────────────────────────────────────────────────────────

fn small_grid() -> Grid {
    Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("grid")
}

fn water(grid: &Grid) -> HomogeneousMedium {
    HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, grid)
}

/// Allocate a field array with all required channels initialised to
/// physiologically plausible values.
///
/// Channels: [Pressure(0), Temperature(1), BubbleRadius(2), BubbleVelocity(3),
///            Density(4), SoundSpeed(5)]
fn field_array(grid: &Grid) -> Array4<f64> {
    let n_fields = 6;
    let mut f = Array4::zeros((n_fields, grid.nx, grid.ny, grid.nz));
    f.index_axis_mut(Axis(0), UnifiedFieldType::Pressure.index())
        .fill(50_000.0);
    f.index_axis_mut(Axis(0), UnifiedFieldType::BubbleRadius.index())
        .fill(5e-6);
    f
}

// ── KellerMiksis ─────────────────────────────────────────────────────────

#[test]
fn km_plugin_initialises_and_registers_correct_fields() {
    let config = BubbleDynamicsConfig {
        model: BubbleModel::KellerMiksis,
        nucleation: false,
        params: BubbleParameters::default(),
    };
    let mut plugin = BubbleDynamicsPlugin::new(config);
    let grid = small_grid();
    let medium = water(&grid);

    assert_eq!(plugin.required_fields(), vec![UnifiedFieldType::Pressure]);
    assert_eq!(
        plugin.provided_fields(),
        vec![
            UnifiedFieldType::BubbleRadius,
            UnifiedFieldType::BubbleVelocity
        ]
    );

    plugin
        .initialize(&grid, &medium)
        .expect("KM init must succeed");
    assert_eq!(plugin.state(), PluginState::Initialized);
}

#[test]
fn km_plugin_writes_nonzero_radius_after_update() {
    let config = BubbleDynamicsConfig {
        model: BubbleModel::KellerMiksis,
        nucleation: false,
        params: BubbleParameters::default(),
    };
    let mut plugin = BubbleDynamicsPlugin::new(config);
    let grid = small_grid();
    let medium = water(&grid);
    plugin.initialize(&grid, &medium).expect("init");

    let mut fields = field_array(&grid);
    let extra_fields = null_plugin_fields(&grid);
    let mut null_boundary = NullBoundary;
    let mut ctx = make_context(&extra_fields, &mut null_boundary);
    plugin
        .update(&mut fields, &grid, &medium, 1e-7, 0.0, &mut ctx)
        .expect("KM update must succeed");

    let cx = grid.nx / 2;
    let cy = grid.ny / 2;
    let cz = grid.nz / 2;
    let r = fields[[UnifiedFieldType::BubbleRadius.index(), cx, cy, cz]];
    assert!(
        r > 0.0,
        "BubbleRadius at centre must be positive after KM update; got {r}"
    );
}

// ── RayleighPlesset ───────────────────────────────────────────────────────

#[test]
fn rp_plugin_initialises_and_advances() {
    let mut params = BubbleParameters::default();
    params.driving_amplitude = 30_000.0;
    let config = BubbleDynamicsConfig {
        model: BubbleModel::RayleighPlesset,
        nucleation: false,
        params,
    };
    let mut plugin = BubbleDynamicsPlugin::new(config);
    let grid = small_grid();
    let medium = water(&grid);
    plugin.initialize(&grid, &medium).expect("RP init");

    let mut fields = field_array(&grid);
    let extra_fields = null_plugin_fields(&grid);
    let mut null_boundary = NullBoundary;
    let mut ctx = make_context(&extra_fields, &mut null_boundary);
    plugin
        .update(&mut fields, &grid, &medium, 1e-7, 0.0, &mut ctx)
        .expect("RP update");

    let cx = grid.nx / 2;
    let cy = grid.ny / 2;
    let cz = grid.nz / 2;
    let r = fields[[UnifiedFieldType::BubbleRadius.index(), cx, cy, cz]];
    assert!(r > 0.0, "RP bubble radius must be positive; got {r}");
}

// ── Gilmore ───────────────────────────────────────────────────────────────

#[test]
fn gilmore_plugin_initialises_and_advances() {
    let config = BubbleDynamicsConfig {
        model: BubbleModel::Gilmore,
        nucleation: false,
        params: BubbleParameters::default(),
    };
    let mut plugin = BubbleDynamicsPlugin::new(config);
    let grid = small_grid();
    let medium = water(&grid);
    plugin.initialize(&grid, &medium).expect("Gilmore init");

    let mut fields = field_array(&grid);
    let extra_fields = null_plugin_fields(&grid);
    let mut null_boundary = NullBoundary;
    let mut ctx = make_context(&extra_fields, &mut null_boundary);
    plugin
        .update(&mut fields, &grid, &medium, 1e-7, 0.0, &mut ctx)
        .expect("Gilmore update");

    let cx = grid.nx / 2;
    let cy = grid.ny / 2;
    let cz = grid.nz / 2;
    let r = fields[[UnifiedFieldType::BubbleRadius.index(), cx, cy, cz]];
    assert!(r > 0.0, "Gilmore bubble radius must be positive; got {r}");
}

// ── Nucleation seeding ────────────────────────────────────────────────────

#[test]
fn nucleation_false_seeds_exactly_one_bubble() {
    let config = BubbleDynamicsConfig {
        model: BubbleModel::KellerMiksis,
        nucleation: false,
        params: BubbleParameters::default(),
    };
    let mut plugin = BubbleDynamicsPlugin::new(config);
    let grid = small_grid();
    let medium = water(&grid);
    plugin.initialize(&grid, &medium).expect("init");

    if let Some(BubbleEngine::KmOrRp { field, .. }) = &plugin.engine {
        assert_eq!(
            field.bubbles.len(),
            1,
            "nucleation=false must seed exactly 1 bubble; got {}",
            field.bubbles.len()
        );
    } else {
        panic!("expected KmOrRp engine");
    }
}

#[test]
fn nucleation_true_seeds_multiple_bubbles() {
    let config = BubbleDynamicsConfig {
        model: BubbleModel::KellerMiksis,
        nucleation: true,
        params: BubbleParameters::default(),
    };
    let mut plugin = BubbleDynamicsPlugin::new(config);
    let grid = small_grid();
    let medium = water(&grid);
    plugin.initialize(&grid, &medium).expect("init");

    if let Some(BubbleEngine::KmOrRp { field, .. }) = &plugin.engine {
        assert!(
            field.bubbles.len() > 1,
            "nucleation=true must seed more than 1 bubble; got {}",
            field.bubbles.len()
        );
    } else {
        panic!("expected KmOrRp engine");
    }
}

// ── Multiple update steps ─────────────────────────────────────────────────

#[test]
fn km_plugin_radius_changes_over_three_steps() {
    let config = BubbleDynamicsConfig {
        model: BubbleModel::KellerMiksis,
        nucleation: false,
        params: BubbleParameters::default(),
    };
    let mut plugin = BubbleDynamicsPlugin::new(config);
    let grid = small_grid();
    let medium = water(&grid);
    plugin.initialize(&grid, &medium).expect("init");

    let dt = 1e-7;
    let mut fields = field_array(&grid);
    let extra_fields = null_plugin_fields(&grid);
    let mut null_boundary = NullBoundary;
    let cx = grid.nx / 2;
    let cy = grid.ny / 2;
    let cz = grid.nz / 2;

    let r0 = BubbleParameters::default().r0;
    let mut prev_r = r0;
    let mut any_changed = false;
    for step in 0..3 {
        let mut ctx = make_context(&extra_fields, &mut null_boundary);
        plugin
            .update(&mut fields, &grid, &medium, dt, step as f64 * dt, &mut ctx)
            .expect("update");
        let r = fields[[UnifiedFieldType::BubbleRadius.index(), cx, cy, cz]];
        assert!(r > 0.0, "radius must be positive at step {step}; got {r}");
        if (r - prev_r).abs() > 1e-15 {
            any_changed = true;
        }
        prev_r = r;
    }
    assert!(
        any_changed,
        "bubble radius must change over multiple steps under acoustic driving"
    );
}
